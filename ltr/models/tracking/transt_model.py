import torch.nn as nn
import torch, random
import torch.nn.functional as F

from ltr.admin.model_constructor import model_constructor
from ltr.utils import box_ops
from ltr.utils import vis_ops
from ltr.utils.misc import (NestedTensor, nested_tensor_from_tensor,
                            nested_tensor_from_tensor_2)
from ltr.models.backbone.transt_backbone import build_backbone
from ltr.models.neck.transt_fusion import build_featurefusion_network
from ltr.models.pre_input import NecessaryPreprocess, IndependentTransform, \
    EllipseRotationAffine, JointTransform, GrayScale
from ltr.utils.tensordict import TensorDict


class TransT(nn.Module):
    """
    This class implements the SP-TransT model that performs sparsely-supervised single object tracking

    It inherits from the Original TransT model: https://github.com/chenxin-dlut/TransT,
            and adds additional features, such as in-network transform/augmentation, IoU prediction, etc.
    """

    def __init__(self, backbone, featurefusion_network, num_classes=1,
                 iou_prediction=False, necessary_pre=None,
                 independent_trans=None, joint_trans=None):
        """
        Initializes the SP-TransT model, here most inputs are actually network components

        args:
            backbone - Torch module of the backbone to be used. See transt_backbone.py
            featurefusion_network - Torch module of the featurefusion_network architecture, a variant of transformer.
                                   See featurefusion_network.py

            num_classes - Number of object classes, always 1 for single object tracking
            iou_prediction - Boolean tag indicating whether to use IoU filtering or not

            necessary_pre - A group of necessary operations for processing image patches, see necessary_preprocess.py
            independent_trans - Independent transform for template patch OR search patch, see transforms_in_net.py
            joint_trans - Joint transform for BOTH template patch AND search patch, see transforms_in_net.py
        """

        super().__init__()
        self.featurefusion_network = featurefusion_network
        hidden_dim = featurefusion_network.d_model
        self.class_embed = MLP(hidden_dim, hidden_dim, num_classes + 1, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.necessary_pre = necessary_pre

        self.independent_trans = independent_trans
        self.joint_trans = joint_trans
        # IoU prediction module (for IoU filtering, see paper for details)
        self.iou_embed = MLP(hidden_dim, hidden_dim, 1, 3) if iou_prediction else None

    def _get_batch_size(self, data):

        if 'template_anno' in data:
            return data['template_anno'].shape[0]
        elif 'relay_anno' in data:
            return data['relay_anno'].shape[0]
        elif 'key_anno' in data:
            return data['key_anno'].shape[0]
        return 0

    def _fusion_pipeline(self, src_template, src_search, mask_template,
                         mask_search, pos_template, pos_search, use_iou=False, b_iou=0):
        # Feature fusion network
        hs = self.featurefusion_network(src_template, mask_template, src_search,
                                        mask_search, pos_template, pos_search)
        # Classification and regression decoder
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        # Attach a branch to predict iou for bounding box regression, predicting pred_bbox IoU with gt_bbox
        # Detach the gradient for IoU head to prevent the regression branch from becoming worse
        if self.iou_embed is not None and use_iou:
            outputs_iou = self.iou_embed(hs[:, :b_iou].detach()).sigmoid()
            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_ious': outputs_iou[-1]}
        else:
            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out

    def template(self, z):
        if not isinstance(z, NestedTensor):
            z = nested_tensor_from_tensor_2(z)
        feature_template, pos_template = self.backbone(z)

        src_template, mask_template = feature_template[-1].decompose()
        src_template = self.input_proj(src_template)

        self.zf = src_template
        self.pos_template = pos_template
        self.mask_template = mask_template

    def track(self, search):

        src_template = self.zf
        pos_template = self.pos_template
        mask_template = self.mask_template

        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor_2(search)
        features_search, pos_search = self.backbone(search)
        src_search, mask_search = features_search[-1].decompose()
        src_search = self.input_proj(src_search)
        assert mask_search is not None

        hs = self.featurefusion_network(src_template, mask_template, src_search,
                                        mask_search, pos_template[-1], pos_search[-1])
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out

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

    def _decode_output(self, outputs, batch_burnin=0):

        outputs_new = TensorDict()
        out_keys = ['logits', 'boxes', 'ious'] if self.iou_embed is not None else ['logits', 'boxes']
        for key in out_keys:
            outputs_new[key + "_sup"] = outputs['pred_' + key][:batch_burnin]
            if outputs['pred_' + key].shape[0] > batch_burnin:
                outputs_new[key + '_unsup'] = outputs['pred_' + key][batch_burnin:]

        return outputs_new

    def _refresh_anno(self, outputs, burnin_data, sp_data):

        # Note that here ground truth labels have been transformed, so we should return the transformed labels
        if burnin_data is not None and (self.necessary_pre is not None or self.joint_trans is not None):
            for s in ['search']:
                outputs[s + '_anno'] = burnin_data[s + '_anno'].detach()
        if sp_data is not None and (self.independent_trans is not None or self.joint_trans is not None):
            for s in ['unlabeled', 'aug']:
                if s + '_anno' in sp_data:
                    outputs[s + '_anno'] = sp_data[s + '_anno'].detach()
        return outputs

    def forward(self, burnin_data=None, teacher_data=None, sp_data=None):
        """
        The forward pass expects two potential input types:

        1. burnin_data: strictly following original TransT, often used for ablation study / baseline
           Called often from TranstSparseActor (mode 'burnin'), inputting a TensorDict

        2. sp_data: Called often from TranstSparseActor (mode 'spsup'), inputting a TensorDict containing:
           - 'relay_images', 'aug_images': Cropped patches
           - 'relay_anno', 'aug_anno': GT / pseudo labels for target on the search patch
           - 'relay_avg', 'aug_avg': Average channel info for filling the crops
           Input as student_data means these data are utilized as  unsupervised training pairs.
           Thus, strong augmentation may be performed for instances in the batch.

        3. teacher_data: Called often from TranstSparseActor (mode 'inference'), TensorDict structure is the same.
           Input as teacher_data means these data are utilized as inference only (obtaining pseudo labels)
           Thus, strong augmentation is never performed for instances in the batch.

        It returns a TensorDict with the following elements:
           - "pred_logits": the classification logits for all feature vectors.
                            Shaped [batch_size, num_vectors, (num_classes + 1)]
           - "pred_boxes": The normalized boxes coordinates for all feature vectors, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image.
           - "pred_ious": The predicted IoU scores for all box outputs, shaped [batch_size, num_vectors, 1]
           - "template_anno", "search_anno": The transformed annotations after augmentation, used in SparseActor
        """

        # Transform and augment the data in multiple GPUs
        burnin_data = self._decode_pairwise_data(burnin_data, "burnin", ['template', 'search'])
        teacher_data = self._decode_pairwise_data(teacher_data, "sparse", ['key', 'unlabeled'])
        sp_data = self._decode_sp_data(sp_data)

        # Assemble template and search patches from burnin_data and sp_data
        template, search = self._assemble_data(burnin_data, teacher_data, sp_data)

        # Backbone feature extraction
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor(search)
        if not isinstance(template, NestedTensor):
            template = nested_tensor_from_tensor(template)
        feature_search, pos_search = self.backbone(search)
        feature_template, pos_template = self.backbone(template)
        src_search, mask_search = feature_search[-1].decompose()
        src_template, mask_template = feature_template[-1].decompose()
        src_search = self.input_proj(src_search)
        src_template = self.input_proj(src_template)

        # Teacher inference
        if teacher_data is not None:
            batch_teacher = self._get_batch_size(teacher_data)
            out = self._fusion_pipeline(src_template, src_search, mask_template, mask_search,
                                        pos_template[-1], pos_search[-1], b_iou=batch_teacher, use_iou=True)

        # Burn-in training and Sparsely-supervised training
        else:
            batch_burnin = self._get_batch_size(burnin_data)
            out = self._fusion_pipeline(src_template, src_search, mask_template, mask_search,
                                        pos_template[-1], pos_search[-1], b_iou=batch_burnin, use_iou=True)
            out = self._decode_output(out, batch_burnin=batch_burnin)

        # Refresh the box annotations (for loss calc) if some augmentation is performed in network
        out = self._refresh_anno(out, burnin_data, sp_data)

        return out


class MLP(nn.Module):
    """
    Very simple multi-layer perceptron (also called FFN)
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@model_constructor
def transt_resnet50(settings, network="student"):
    """
    Build and return a SP-TransT model
    """

    assert network in ["student", "teacher", "baseline"], \
        "Network construction type should be in 'student', 'teacher' or 'baseline'."

    # Basic network structure of TransT
    num_classes = settings.num_classes
    backbone_net = build_backbone(settings, backbone_pretrained=True)
    featurefusion_network = build_featurefusion_network(settings)

    # Necessary preprocessing for all materials
    necessary_pre = NecessaryPreprocess(b_j=settings.aug_settings['jitter'],
                                        mean=settings.normalize_mean,
                                        std=settings.normalize_std)

    # For teacher network, no joint transform is needed
    if network == "teacher":
        joint_trans = None
    else:
        joint_trans = JointTransform([GrayScale(gray_probability=settings.aug_settings['gray'])])

    # For teacher network, no independent transform is needed as well
    if network == "teacher" or (not settings.strong_augmentation_tag):
        independent_trans = None
    else:
        independent_trans = IndependentTransform([
            EllipseRotationAffine(max_angle=settings.aug_settings['angle'],
                                  lr_flip_prob=settings.aug_settings['lr_f_prob'],
                                  ud_flip_prob=settings.aug_settings['ud_f_prob'])])

    model = TransT(
        backbone_net,
        featurefusion_network,
        num_classes=num_classes,
        iou_prediction=settings.iou_prediction_tag,
        necessary_pre=necessary_pre,
        joint_trans=joint_trans,
        independent_trans=independent_trans,
    )
    device = torch.device(settings.device)
    model.to(device)
    return model
