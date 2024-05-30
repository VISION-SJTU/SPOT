from . import BaseSparseActor
import torch
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from ltr.utils import TensorDict
from ltr.utils.box_ops import box_xyxy_to_xywh


class OceanSparseActor(BaseSparseActor):
    """ Actor for training Ocean in a sparsely-supervised manner """

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.998):

        """
        Update the teacher with model weights from the student

        args:
            keep_rate  - Weights of updating strategy: tea <-- stu  * (1 - keep_rate) + tea * keep_rate

        """

        # Get current student model weight
        student_model_dict = self.student_net.state_dict()

        # Update teacher model weight with momentum strategy from student model
        new_teacher_dict = OrderedDict()
        for key, value in self.teacher_net.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                        student_model_dict[key] * (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.teacher_net.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _inference_with_teacher(self, teacher_data):
        """
        Inference with the teacher network, and post-process the tracking results on search areas

        args:
            teacher_data - The input data for teacher network inference

        outputs:
            pseudo_label_unc - The forward tracking results with the teacher

        """

        # Inference with teacher model
        outputs = self.teacher_net(teacher_data=teacher_data)

        # Convert the result
        cls_score = F.sigmoid(outputs['pred_logits'])
        cls_align = F.sigmoid(outputs['align_logits'])
        ratio = self.settings.curriculum_settings['cls_ratio']
        score = ratio * cls_score + (1 - ratio) * cls_align

        # Temporally smooth window
        hanning = torch.tensor(np.hanning(self.settings.output_sz),
                               device=score.device, dtype=score.dtype)
        window = torch.outer(hanning, hanning)
        window_flatten = window.unsqueeze(0)
        window_penalty = self.settings.curriculum_settings['window_penalty']

        # Add window penalty
        score_offset = window_penalty * window_flatten[None, :] + (1 - window_penalty) * score

        # For Ocean, the box is already to the image scale, and is in [x0, y0, x1, y1]
        batch_teacher = score.shape[0]
        score_offset = score_offset.reshape(batch_teacher, -1)
        pred_bbox = outputs['bbox_pred_to_img'].reshape(batch_teacher, 4, -1).transpose(1, 2)
        pred_IoU = outputs['pred_ious'].reshape(batch_teacher, -1)

        # Find the bounding box with the highest score
        best_score, best_idx = torch.max(score_offset, dim=1)
        # [x0, y0, x1, y1] --> [x0, y0, w, h]
        bbox_xyxy = pred_bbox[torch.arange(pred_bbox.shape[0]), best_idx]
        bbox_xywh = box_xyxy_to_xywh(bbox_xyxy)
        iou_conf = pred_IoU[torch.arange(pred_IoU.shape[0]), best_idx]

        # Making sure that [x0 >= 0, y0 >= 0, w >= 1, h >= 1] for pseudo label generated
        torch.clamp_min_(bbox_xywh[:, 0:2], min=0)
        torch.clamp_min_(bbox_xywh[:, 2:4], min=1)

        pseudo_label_unc = TensorDict({
            'best_bbox': bbox_xywh.reshape(best_score.shape[0], 4),
            'best_score': best_score,
            'best_conf': iou_conf.reshape(best_score.shape),
        })

        return pseudo_label_unc

    def _burnin_training(self, burnin_data):

        """
        Training the student network with supervised pairs only

        args:
            burnin_data - The input data for student burn-in training

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """

        # Pass data through the student net
        # Use transformed label as ground truth data if strong augmentation is performed in the network
        outputs = self.student_net(burnin_data=burnin_data)

        # Generate labels
        targets = self._build_targets(data=burnin_data, outputs=outputs, tag="search")

        # Compute loss
        # Meaning that it is still in burn-in stage
        loss_dict = self.objective(outputs, targets, tag="sup")
        weight_dict = self.objective.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k + "_sup"] for k in loss_dict.keys() if k + '_sup' in weight_dict)

        # Return training stats
        stats = {'Loss/total': losses.item(),
                 'Loss/ce': loss_dict['loss_ce'].item(),
                 'Loss/bbox': loss_dict['loss_bbox'].item(),
                 'Loss/align': loss_dict['loss_align'].item(),
                 'iou': loss_dict['iou'].item()}

        # IoU prediction stat
        if 'loss_iou_pred' in loss_dict:
            stats['Loss/iou_pred'] = loss_dict['loss_iou_pred'].item()

        return losses, stats

    def _sp_training(self, burnin_data, sp_data):

        """
        Training the student network with the provided data (both burnin_data and sp_data)

        args:
            burnin_data - The input data for student burn-in training (supervised pairs)
            sp_data - The input data for student spot training (unsupervised pairs)

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """

        # Pass data through the student net
        # Use transformed label as ground truth data if strong augmentation is performed in the network
        outputs = self.student_net(burnin_data=burnin_data, sp_data=sp_data)

        # Fetching batch size
        batch_size_sup, batch_size_unsup = burnin_data['search_anno'].shape[0], sp_data['aug_anno'].shape[0]
        batch_size = batch_size_unsup + batch_size_sup

        # Convert search_anno from xywh to cxcywh, in [0,1] scale
        targets_sup = self._build_targets(burnin_data, outputs, tag="search")
        targets_unsup = self._build_targets(sp_data, outputs, tag="aug")

        # Compute loss
        weight_dict = self.objective.weight_dict

        # Loss for fully labeled instances in batch
        outputs_sup = {k: outputs[k] for k in outputs if k.endswith('_sup')}
        loss_dict_sup = self.objective(outputs_sup, targets_sup, tag="sup")
        loss_sup = sum(loss_dict_sup[k] * weight_dict[k + "_sup"]
                       for k in loss_dict_sup.keys() if k + "_sup" in weight_dict)

        # Loss for unsupervised instances in batch
        outputs_unsup = {k: outputs[k] for k in outputs if k.endswith('_unsup')}
        loss_dict_unsup = self.objective(outputs_unsup, targets_unsup, tag="unsup")
        loss_unsup = sum(loss_dict_unsup[k] * weight_dict[k + "_unsup"]
                         for k in loss_dict_unsup.keys() if k + "_unsup" in weight_dict)

        # Total loss (weighted sum of supervised and unsupervised parts)
        losses = (float(batch_size_sup) * loss_sup + float(batch_size_unsup) * loss_unsup) / batch_size

        # Return training stats
        stats = {'Loss/total': losses.item()}
        stat_items = {'loss_ce': 'Loss/ce', 'loss_bbox': 'Loss/bbox', 'loss_align': 'Loss/align',
                      'loss_iou_pred': 'Loss/iou_pred', 'iou': 'iou',
                      'iou_sup': 'iou_sup', 'iou_unsup': 'iou_unsup'}
        for key in stat_items.keys():
            if key.startswith("iou_"):
                link = {'iou_sup': loss_dict_sup, 'iou_unsup': loss_dict_unsup}
                stats[stat_items[key]] = link[key]['iou'].item()
            if key not in loss_dict_sup:
                continue
            elif key not in loss_dict_unsup:
                stats[stat_items[key]] = loss_dict_sup[key].item()
            else:
                stats[stat_items[key]] = (float(batch_size_sup) * loss_dict_sup[key].item() +
                                          float(batch_size_unsup) * loss_dict_unsup[key].item()) / batch_size

        return losses, stats

    def _build_targets(self, data, outputs, tag):

        # Convert search_anno from xywh to cxcywh, in [0,1] scale
        targets = {}
        targets_origin = outputs[tag + '_anno']
        h, w = data[tag + '_images'][0][0].shape
        tgt_cx = (targets_origin[:, 0] + targets_origin[:, 2] / 2) / w
        tgt_cy = (targets_origin[:, 1] + targets_origin[:, 3] / 2) / h
        tgt_w = targets_origin[:, 2] / w
        tgt_h = targets_origin[:, 3] / h
        targets['boxes_cxcywh'] = torch.stack([tgt_cx, tgt_cy, tgt_w, tgt_h]).T

        # Targets format: (x1, y1, x2, x3)
        tgt_x1_orig = targets_origin[:, 0]
        tgt_y1_orig = targets_origin[:, 1]
        tgt_x2_orig = targets_origin[:, 0] + targets_origin[:, 2]
        tgt_y2_orig = targets_origin[:, 1] + targets_origin[:, 3]
        targets['boxes_orig_xyxy'] = torch.stack([tgt_x1_orig, tgt_y1_orig, tgt_x2_orig, tgt_y2_orig]).T
        return targets

    def __call__(self, mode, teacher_data=None, burnin_data=None, sp_data=None, keep_rate=0.998):

        """
        Operations of the Actor for sparsely-supervised learning, including three modes:

            update: update the teacher with model weights from the student
            inference: inference with the teacher based on the loaded data
            burnin: training with supervised pairs only
            spsup: training with supervised and unsupervised pairs both

        args:
            mode - Operation to perform, choose from "update", "inference", "burnin", "spsup"
            For other arguments, please see specific functions.

        """

        if mode == "update":
            self._update_teacher_model(keep_rate=keep_rate)
        elif mode == "inference":
            return self._inference_with_teacher(teacher_data=teacher_data)
        elif mode == "burnin":
            return self._burnin_training(burnin_data=burnin_data)
        elif mode == "spsup":
            return self._sp_training(burnin_data=burnin_data, sp_data=sp_data)
        else:
            assert False, "No such operation for OceanSparseActor"
