import torch.nn as nn
import torch
import torch.nn.functional as F
from ltr.utils import box_ops
from ltr.models.loss.transt_matcher import build_matcher


class SetCriterionTransT(nn.Module):
    """
    This class computes the loss for SP-TransT.
    The process happens in two steps:
        1. we compute assignment between ground truth box and the outputs of the model
        2. we supervise each pair of matched ground-truth / prediction (supervise class, box, and IoU)
    """

    def __init__(self, settings, matcher):
        """
        Create the SP-TransT criterion.

        args:
            settings - The training settings of SP-TransT
            matcher - Module able to compute a matching between target and proposals
        """
        super().__init__()

        self.settings = settings
        self.matcher = matcher
        self.losses = ['labels', 'boxes']

        # Here eos_coef is the relative classification weight applied to the no-object category
        self.eos_coef = settings.eos_coef
        # Here num_classes is the number of object categories, always be 1 for single object tracking.
        self.num_classes = settings.num_classes

        self.weight_dict = settings.weight_dict
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, tag="sup"):
        """
        Classification loss (NLL).
        It targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """

        assert 'logits_' + tag in outputs
        src_logits = outputs['logits_' + tag]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, indices_neg, num_boxes_neg, tag="sup"):
        """
        Compute the losses related to the bounding boxes and IoU prediction with:
               1. L1 regression loss for bounding box regression
               2. GIoU loss for bounding box regression
               3. L1 regression loss for IoU predictions

           The targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4].
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """

        assert 'boxes_' + tag in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['boxes_' + tag][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        # Calculate bbox regression loss for boxes of only positive indices
        losses = {}
        giou, iou = box_ops.generalized_box_iou_pair_xyxy(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes))
        loss_giou = 1 - giou

        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        losses['iou'] = iou.sum() / num_boxes

        # IoU prediction loss for normal head is hacked here
        if indices_neg is not None and 'ious_' + tag in outputs:
            # Calculate IoU prediction loss for positive positions
            src_iou_pred = outputs['ious_' + tag].squeeze(2)[idx]
            loss_iou_pred = F.l1_loss(src_iou_pred, iou.detach(), reduction='none')

            # Calculate IoU prediction loss for negative positions
            idx_neg = self._get_src_permutation_idx(indices_neg)
            src_iou_pred_neg = outputs['ious_' + tag].squeeze(2)[idx_neg]
            src_boxes_neg = outputs['boxes_' + tag][idx_neg]
            target_boxes_neg = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices_neg)], dim=0)

            iou_neg, _ = box_ops.box_iou_pair_xyxy(
                box_ops.box_cxcywh_to_xyxy(src_boxes_neg),
                box_ops.box_cxcywh_to_xyxy(target_boxes_neg))
            loss_iou_pred_neg = F.l1_loss(src_iou_pred_neg, iou_neg.detach(), reduction='none')

            # IoU prediction loss is the loss sum from both positive and negative positions
            losses['loss_iou_pred'] = loss_iou_pred.sum() / num_boxes + \
                                      loss_iou_pred_neg.sum() / num_boxes_neg

        return losses

    def _get_src_permutation_idx(self, indices):

        # Permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):

        # Permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, indices_neg, num_boxes_neg, tag):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes
        }
        assert loss in loss_map, "do you really want to compute {} loss?".format(loss)
        if loss == 'boxes':
            return loss_map[loss](outputs, targets, indices, num_boxes, indices_neg, num_boxes_neg, tag)
        else:
            return loss_map[loss](outputs, targets, indices, num_boxes, tag)

    def forward(self, outputs, targets, tag="sup"):
        """
        This performs the loss computation.
        args:
             outputs - Dict of tensors, see the output specification of the model for the format
             targets - List of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the target
        indices, indices_neg = self.matcher(outputs_without_aux, targets, tag=tag)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes_pos = sum(len(t[0]) for t in indices)
        num_boxes_pos = torch.as_tensor([num_boxes_pos], dtype=torch.float,
                                        device=next(iter(outputs.values())).device)
        num_boxes_pos = torch.clamp(num_boxes_pos, min=1).item()

        # Compute the average number of negative target boxes (for iou prediction) across all nodes
        if 'ious_' + tag in outputs:
            num_boxes_neg = sum(len(t[0]) for t in indices_neg)
            num_boxes_neg = torch.as_tensor([num_boxes_neg], dtype=torch.float,
                                            device=next(iter(outputs.values())).device)
            num_boxes_neg = torch.clamp(num_boxes_neg, min=1).item()
        else:
            num_boxes_neg = None

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes_pos,
                                        indices_neg, num_boxes_neg, tag))

        return losses


def transt_loss(settings):
    """
    Construct and Return the criterion for SP-TransT loss calculation
    """
    matcher = build_matcher()
    criterion = SetCriterionTransT(settings=settings, matcher=matcher)
    device = torch.device(settings.device)
    criterion.to(device)
    return criterion
