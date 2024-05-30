# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# Revised by Jilai Zheng (zhengjilai@sjtu.edu.cn), for SPOT
# ------------------------------------------------------------------------------
import numpy as np
import torch.nn as nn
from ltr.admin.model_constructor import model_constructor
import torch, random
from ltr.utils import vis_ops
from ltr.models.backbone.ocean_backbone import build_backbone_ocean
from ltr.models.neck.ocean_fusion import AdjustLayer, box_tower, AlignHead
from ltr.models.pre_input import NecessaryPreprocess, IndependentTransform, \
    EllipseRotationAffine, JointTransform, GrayScale
from ltr.models.loss.ocean_label_gen import OceanLabelGenerator
from ltr.utils.tensordict import TensorDict


class Ocean(nn.Module):
    """
    This class implements the SP-Ocean model that performs sparsely-supervised single object tracking

    It inherits from the Original Ocean model: https://github.com/researchmm/TracKit,
            and adds additional features, such as in-network transform/augmentation, IoU prediction, etc.
    """

    def __init__(self, backbone, neck, connect_model, align_head,
                 search_size=255, score_size=25, batch_size=32,
                 training=True, necessary_pre=None, independent_trans=None,
                 joint_trans=None, iou_prediction=False, label_generator=None):

        """
        Initializes the SP-Ocean model, here most inputs are actually network components

        args:
            backbone - Torch module of the backbone to be used. See ocean_backbone.py
            neck - Torch module of the neck of Ocean, see ocean_fusion.py
            connect_model - Torch module of the fusion/relation layers of Ocean, see ocean_fusion.py
            align_head - Torch module of the feature align module of Ocean, see ocean_fusion.py

            search_size, score_size, batch_size - Important hyper-parameters for image / feature size
            training - Boolean tag indicating whether the network is in training mode or testing mode

            necessary_pre - A group of necessary operations for processing image patches, see necessary_preprocess.py
            independent_trans - Independent transform for template patch OR search patch, see transforms_in_net.py
            iou_prediction - Boolean tag indicating whether or not to use IoU filtering
            joint_trans - Joint transform for BOTH template patch AND search patch, see transforms_in_net.py
            label_generator - Generating Ocean cls/reg/align labels in multiple GPUs, see label_generator.py
        """

        super(Ocean, self).__init__()
        self.features = backbone
        self.connect_model = connect_model
        self.align_head = align_head
        self.neck = neck
        self.zf = None
        self.search_size = search_size
        self.score_size = score_size
        self.batch = batch_size if training else 1
        self.training = training

        self.necessary_pre = necessary_pre
        self.independent_trans = independent_trans
        self.joint_trans = joint_trans

        self.iou_prediction = iou_prediction
        self.label_generator = label_generator

        self.grids()

    def feature_extractor(self, x):
        return self.features(x)

    def connector(self, template_feature, search_feature):
        pred_score = self.connect_model(template_feature, search_feature)
        return pred_score

    def grids(self):
        """
        Generate a grid, for each element of feature map on the response map
        The result (grid_to_search_x/y) is shaped B*H*W (the position for each element)
        This grid is useful when generating cls/reg/align labels, or decoding the tracking results from response maps
        """
        sz = self.score_size
        stride = 8

        sz_x = sz // 2
        sz_y = sz // 2

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        self.grid_to_search = {}
        self.grid_to_search_x = x * stride + self.search_size // 2
        self.grid_to_search_y = y * stride + self.search_size // 2

        self.grid_to_search_x = torch.Tensor(self.grid_to_search_x).unsqueeze(0).unsqueeze(0).cuda()
        self.grid_to_search_y = torch.Tensor(self.grid_to_search_y).unsqueeze(0).unsqueeze(0).cuda()

        self.grid_to_search_x = self.grid_to_search_x.repeat(self.batch, 1, 1, 1)
        self.grid_to_search_y = self.grid_to_search_y.repeat(self.batch, 1, 1, 1)

    def pred_to_image(self, bbox_pred):
        """
        Decoding the predicted boxes from response maps to boxes in search patch axis
        """
        self.grid_to_search_x = self.grid_to_search_x.to(bbox_pred.device)
        self.grid_to_search_y = self.grid_to_search_y.to(bbox_pred.device)

        b = bbox_pred.shape[0]
        pred_x1 = self.grid_to_search_x[0:b, :] - bbox_pred[:, 0, ...].unsqueeze(1)  # 17*17
        pred_y1 = self.grid_to_search_y[0:b, :] - bbox_pred[:, 1, ...].unsqueeze(1)  # 17*17
        pred_x2 = self.grid_to_search_x[0:b, :] + bbox_pred[:, 2, ...].unsqueeze(1)  # 17*17
        pred_y2 = self.grid_to_search_y[0:b, :] + bbox_pred[:, 3, ...].unsqueeze(1)  # 17*17

        pred = [pred_x1, pred_y1, pred_x2, pred_y2]

        pred = torch.cat(pred, dim=1)

        return pred

    def offset(self, boxes, featmap_sizes):
        """
        Offsets for align head. Refer to Ocean paper for more details.
        """

        def _shape_offset(boxes, stride):
            ks = 3
            dilation = 1
            pad = (ks - 1) // 2
            idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)
            yy, xx = torch.meshgrid(idx, idx)  # return order matters
            xx = xx.reshape(-1)
            yy = yy.reshape(-1)
            w = (boxes[:, 2] - boxes[:, 0] + 1) / stride
            h = (boxes[:, 3] - boxes[:, 1] + 1) / stride
            w = w / (ks - 1) - dilation
            h = h / (ks - 1) - dilation
            offset_x = w[:, None] * xx  # (NA, ks**2)
            offset_y = h[:, None] * yy  # (NA, ks**2)
            return offset_x, offset_y

        def _ctr_offset(boxes, stride, featmap_size):
            feat_h, feat_w = featmap_size
            image_size = self.search_size

            assert len(boxes) == feat_h * feat_w

            x = (boxes[:, 0] + boxes[:, 2]) * 0.5
            y = (boxes[:, 1] + boxes[:, 3]) * 0.5

            # Different here for Siamese
            # Use center of image as coordinate origin
            x = (x - image_size * 0.5) / stride + feat_w // 2
            y = (y - image_size * 0.5) / stride + feat_h // 2

            # Compute predefine centers
            # Different here for Siamese
            xx = torch.arange(0, feat_w, device=boxes.device)
            yy = torch.arange(0, feat_h, device=boxes.device)
            yy, xx = torch.meshgrid(yy, xx)
            xx = xx.reshape(-1).type_as(x)
            yy = yy.reshape(-1).type_as(y)

            offset_x = x - xx
            offset_y = y - yy
            return offset_x, offset_y

        num_imgs = len(boxes)
        dtype = boxes[0].dtype
        device = boxes[0][0].device

        featmap_sizes = featmap_sizes[2:]

        offset_list = []
        for i in range(num_imgs):
            c_offset_x, c_offset_y = _ctr_offset(boxes[i], 8, featmap_sizes)
            s_offset_x, s_offset_y = _shape_offset(boxes[i], 8)

            # offset = ctr_offset + shape_offset
            offset_x = s_offset_x + c_offset_x[:, None]
            offset_y = s_offset_y + c_offset_y[:, None]

            # offset order (y0, x0, y1, x0, .., y9, x8, y9, x9)
            offset = torch.stack([offset_y, offset_x], dim=-1)
            offset = offset.reshape(offset.shape[0], -1).unsqueeze(0)  # [NA, 2*ks**2]
            offset_list.append(offset)

        offsets = torch.cat(offset_list, 0)
        return offsets

    def template(self, z):
        _, self.zf = self.feature_extractor(z)

        if self.neck is not None:
            _, self.zf = self.neck(self.zf, crop=True)

    def track(self, x):

        _, xf = self.feature_extractor(x)

        if self.neck is not None:
            xf = self.neck(xf)

        bbox_pred, cls_pred, cls_feature, reg_feature, _ = \
            self.connect_model(xf, self.zf, iou_prediction=False)
        bbox_pred_to_img = self.pred_to_image(bbox_pred)
        offsets = self.offset(bbox_pred_to_img.permute(0, 2, 3, 1).reshape(bbox_pred_to_img.shape[0], -1, 4),
                              bbox_pred.shape)
        cls_align = self.align_head(reg_feature, offsets)

        return cls_pred, bbox_pred, cls_align

    def _decode_pairwise_data(self, data_loaded, stage="burnin", prefix=None, vis=False):

        # Burn-in data transform and augmentation
        if data_loaded is not None:
            # First perform necessary transforms (both template and search), including avg filling, norm and jitter
            for p in prefix:
                data_loaded[p + "_images"], data_loaded[p + "_avg"] = \
                    self.necessary_pre(data_loaded, tag=p)

            if stage == "burnin":
                image_groups, anno_groups, avg_groups = \
                    self.joint_trans(data_loaded, tags=prefix, masks=data_loaded['masks'])
                for i in range(len(prefix)):
                    p = prefix[i]
                    data_loaded[p + "_images"], data_loaded[p + "_anno"], data_loaded[p + "_avg"] = \
                        image_groups[i], anno_groups[i], avg_groups[i]

            # Visualize testing after student data transform and augmentation
            if vis:
                for j in range(len(data_loaded[prefix[0] + '_anno'])):
                    rand_str = random.randint(0, 999999)
                    for p in prefix:
                        vis_name = "{}_{:06d}_{}_{}".format(stage, rand_str, j, p[0])
                        vis_ops.vis_patches(data_loaded[p + '_images'][j],
                                            data_loaded[p + '_anno'][j], vis_name, de_norm=True)

        return data_loaded

    def _decode_sp_data(self, sp_data, vis=False):

        # Student data transform and augmentation
        if sp_data is not None:

            # First do necessary transforms (both template and search), including avg filling, norm and jitter
            for s in ['relay', 'aug']:
                sp_data[s + "_images"], sp_data[s + "_avg"] = \
                    self.necessary_pre(sp_data, tag=s)

            # Joint transformation for both template and search (e.g. grayscale)
            # This scheme is used only for student net or baseline ablation.
            if self.joint_trans is not None:
                tags = ['relay', 'aug']
                image_groups, anno_groups, avg_groups = \
                    self.joint_trans(sp_data, tags=tags, masks=sp_data['masks'])
                for i in range(len(tags)):
                    tag = tags[i]
                    sp_data[tag + "_images"], sp_data[tag + "_anno"], sp_data[tag + "_avg"] = \
                        image_groups[i], anno_groups[i], avg_groups[i]

            # Strong augmentation for relay patches (e.g. Ellipse Rotation)
            if self.independent_trans is not None:
                tag = "aug"
                sp_data[tag + "_images"], sp_data[tag + "_anno"] = \
                    self.independent_trans(sp_data, tag=tag)

            # Visualize testing after student data transform and augmentation
            if vis:
                for j in range(len(sp_data['relay_anno'])):
                    tags = ['relay', 'aug']
                    rand_tag = random.randint(0, 999999)
                    for tag in tags:
                        vis_name = "Sp_{:06d}_{}_{}".format(rand_tag, j, tag[0])
                        vis_ops.vis_patches(sp_data[tag + '_images'][j],
                                            sp_data[tag + '_anno'][j], vis_name, de_norm=True)

        return sp_data

    def _assemble_data(self, burnin_data, teacher_data, sp_data):

        all_templates = []
        all_searches = []

        if teacher_data is not None:
            all_searches.append(teacher_data['unlabeled_images'])
            all_templates.append(teacher_data['key_images'])
        if burnin_data is not None:
            all_searches.append(burnin_data['search_images'])
            all_templates.append(burnin_data['template_images'])
        if sp_data is not None:
            all_searches.append(sp_data['aug_images'])
            all_templates.append(sp_data['relay_images'])

        template = torch.cat(all_templates, dim=0).detach()
        search = torch.cat(all_searches, dim=0).detach()

        return template, search

    def _fusion_pipeline(self, xf, zf, pred_to_img_tag=False, b_iou=0):

        bbox_pred, cls_pred, cls_feature, reg_feature, auxi_outputs = \
            self.connect_model(xf, zf, iou_prediction=self.iou_prediction, batch_iou=b_iou)
        bbox_pred_to_img = self.pred_to_image(bbox_pred)
        offsets = self.offset(bbox_pred_to_img.permute(0, 2, 3, 1).reshape(
            bbox_pred_to_img.shape[0], -1, 4), bbox_pred.shape)
        cls_align = self.align_head(reg_feature, offsets)
        out = {'pred_logits': cls_pred, 'pred_boxes': bbox_pred, 'align_logits': cls_align}

        # Attach a branch to predict iou for bounding box regression, predicting pred_bbox IoU with gt_bbox
        # Detach the gradient for IoU head to prevent the regression branch from becoming weaker
        if self.iou_prediction:
            out['pred_ious'] = auxi_outputs['pred_ious']

        # For Ocean, we transform the regression map to the regression results (used in SparseActor)
        if pred_to_img_tag:
            out['bbox_pred_to_img'] = bbox_pred_to_img.detach()

        return out

    def _get_batch_size(self, data):

        if 'template_anno' in data:
            return data['template_anno'].shape[0]
        elif 'relay_anno' in data:
            return data['relay_anno'].shape[0]
        elif 'key_anno' in data:
            return data['key_anno'].shape[0]
        return 0

    def _decode_output(self, outputs, batch_burnin=0):

        outputs_new = TensorDict()
        out_keys = ['pred_logits', 'pred_boxes', 'align_logits', 'pred_ious'] \
            if self.iou_prediction else ['pred_logits', 'pred_boxes', 'align_logits']
        for key in out_keys:
            outputs_new[key + "_sup"] = outputs[key][:batch_burnin]
            if outputs[key].shape[0] > batch_burnin:
                outputs_new[key + '_unsup'] = outputs[key][batch_burnin:]

        return outputs_new

    def _refresh_anno(self, out, data, tag="search"):

        # Note that here ground truth labels have been transformed, so we should return the transformed labels
        if data is not None:
            out[tag + '_anno'] = data[tag + '_anno'].detach()

            # Don't forget to add torch.no_grad(), as these are originally generated in dataloader with no grads
            with torch.no_grad():
                cls_label = self.label_generator({"anno": data[tag + '_anno']}, tag="cls")
                reg_label, reg_weight = self.label_generator({"anno": data[tag + '_anno']}, tag="reg")

            # Fetch pred boxes
            sup_tag = "_sup" if tag == "search" else "_unsup"
            bbox_pred = out['pred_boxes' + sup_tag]
            # It seems that the original Ocean does not detach gradient for generating align labels
            align_cls_label = self.label_generator({"bbox_pred": bbox_pred, "reg_label": reg_label,
                                                    "reg_weight": reg_weight}, tag="align")

            # Weights for IoU prediction, as negative boxes are also included in loss calculation
            if self.iou_prediction:
                with torch.no_grad():
                    out['iou_weight' + sup_tag] = self.label_generator({"anno": data[tag + '_anno']}, tag="iou_w")

            # Record all generated labels for Ocean
            out['reg_label' + sup_tag], out['reg_weight' + sup_tag] = reg_label, reg_weight
            out['cls_label' + sup_tag], out['align_label' + sup_tag] = cls_label, align_cls_label

        return out

    def forward(self, burnin_data=None, teacher_data=None, sp_data=None):
        """
        The forward pass expects two potential input types:

        1. burnin_data: strictly following original Ocean, often used for ablation study / baseline
           Called often from OceanSparseActor (mode 'burnin'), inputting a TensorDict

        2. sp_data: Called often from OceanSparseActor (mode 'spsup'), inputting a TensorDict containing:
           - 'relay_images', 'aug_images': Cropped patches
           - 'relay_anno', 'aug_anno': GT / pseudo labels for target on the search patch
           - 'relay_avg', 'aug_avg': Average channel info for filling the crops
           Input as sp_data means these data are utilized as unsupervised training pairs.
           Thus, strong augmentation may be performed on instances in the batch.

        3. teacher_data: Called often from OceanSparseActor (mode 'inference'), TensorDict structure is the same.
           Input as teacher_data means these data are utilized for inference only (obtaining pseudo labels)
           Thus, strong augmentation is never performed on instances in the batch.

        It returns a TensorDict with the following elements:
           - "pred_logits": The classification logits in response map. Shaped [batch_size, H, W, 1]
           - "pred_boxes": The box offsets related to the center pos on the response map, Shaped [batch_size, H, W, 4]
           - "align_logits": The predicted logits for the feature align branch, Shaped [batch_size, H, W, 1]
           - "pred_ious": The predicted IoU scores for all box outputs, shaped [batch_size, H, W, 4]

           - "search_anno": The transformed annotations after augmentation in network, used as gt
           - "reg_label", "reg_weight": Regression labels for Ocean, see label_generator.py
           - "cls_label", "align_label": Classification labels for Ocean (cls/align branch), see label_generator.py
           - "iou_weight": The positions for calculating loss of IoU predictions, see label_generator.py

           - "bbox_pred_to_img": The predicted boxes in search patch axis, reserved for future IoU calculation
        """

        # Transform and augment the data in multiple GPUs
        burnin_data = self._decode_pairwise_data(burnin_data, "burnin", ['template', 'search'])
        teacher_data = self._decode_pairwise_data(teacher_data, "sparse", ['key', 'unlabeled'])
        sp_data = self._decode_sp_data(sp_data)

        # Assemble template and search patches from burnin_data and sp_data
        template, search = self._assemble_data(burnin_data, teacher_data, sp_data)

        _, zf = self.feature_extractor(template)
        _, xf = self.feature_extractor(search)

        if self.neck is not None:
            _, zf = self.neck(zf, crop=True)
            xf = self.neck(xf, crop=False)

        # Teacher inference
        if teacher_data is not None:
            batch_teacher = self._get_batch_size(teacher_data)
            out = self._fusion_pipeline(xf, zf, pred_to_img_tag=True, b_iou=batch_teacher)

        # Burn-in training and sparsely-supervised training
        else:
            batch_burnin = self._get_batch_size(burnin_data)
            out = self._fusion_pipeline(xf, zf, pred_to_img_tag=True, b_iou=batch_burnin)
            out = self._decode_output(out, batch_burnin=batch_burnin)

        # Refresh the box annotations (for loss calc) if some augmentation is performed in network
        out = self._refresh_anno(out, burnin_data, tag="search")
        out = self._refresh_anno(out, sp_data, tag="aug")

        return out


@model_constructor
def ocean_resnet50(settings, network="student"):
    assert network in ["student", "teacher"], "Network construction type should be in 'student' or 'teacher'."

    # Basic Ocean network structure
    ocean_backbone_net = build_backbone_ocean(settings)
    ocean_neck = AdjustLayer(in_channels=settings.before_neck_dim, out_channels=settings.head_dim)
    ocean_connect_model = box_tower(inchannels=settings.head_dim, outchannels=settings.head_dim,
                                    towernum=settings.head_tower_num, iou_prediction=settings.iou_prediction_tag)
    ocean_align_head = AlignHead(settings.head_dim, settings.head_dim)
    ocean_label_generator = OceanLabelGenerator(settings)

    # Necessary preprocessing for all materials
    necessary_pre = NecessaryPreprocess(b_j=settings.aug_settings['jitter'],
                                        mean=settings.normalize_mean,
                                        std=settings.normalize_std)

    # For teacher network, no independent transform is needed as well
    if network == "teacher":
        joint_trans = None
    else:
        joint_trans = JointTransform([GrayScale(gray_probability=settings.aug_settings['gray'])])

    # Never use in network strong transformation when conducting baseline ablation
    if network == "teacher" or (not settings.strong_augmentation_tag):
        independent_trans = None
    else:
        independent_trans = IndependentTransform([
            EllipseRotationAffine(max_angle=settings.aug_settings['angle'],
                                  lr_flip_prob=settings.aug_settings['lr_f_prob'],
                                  ud_flip_prob=settings.aug_settings['ud_f_prob'])])

    # Build the SP-Ocean model
    model = Ocean(
        backbone=ocean_backbone_net,
        neck=ocean_neck,
        connect_model=ocean_connect_model,
        align_head=ocean_align_head,
        search_size=settings.search_sz,
        score_size=settings.output_sz,
        batch_size=settings.batch_size,
        iou_prediction=settings.iou_prediction_tag,
        necessary_pre=necessary_pre,
        joint_trans=joint_trans,
        independent_trans=independent_trans,
        label_generator=ocean_label_generator
    )

    device = torch.device(settings.device)
    model.to(device)
    return model
