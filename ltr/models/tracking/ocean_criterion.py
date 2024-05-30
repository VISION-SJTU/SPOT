# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# Revised by Jilai Zheng (zhengjilai@sjtu.edu.cn), for SPOT
# ------------------------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


class SetCriterionOcean(nn.Module):
    """
    This class computes the loss for SP-Ocean.
    The process supervises each pair of matched ground-truth / prediction (class, box, align, and IoU)
    """

    def __init__(self, settings, losses):
        """
        Create the SP-Ocean criterion.

        args:
            settings - The training settings of SP-Ocean
            losses - List of all the losses to be applied. See get_loss for list of available losses.
        """

        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.weight_dict = settings.weight_dict
        self.losses = losses
        self.search_feature_sz = settings.search_feature_sz
        self.search_sz = settings.search_sz
        self.stride = settings.stride
        self.score_size = settings.output_sz
        self.batch = settings.batch_size
        empty_weight = torch.ones(2) * 0.5
        self.register_buffer('empty_weight', empty_weight)

        self.grids()

    def _cls_loss(self, pred, label, select):
        """
        Calculate BCE loss for those selected indices
        """
        if len(select.size()) == 0:
            return 0
        pred = torch.index_select(pred, 0, select)
        label = torch.index_select(label, 0, select)
        return self.criterion(pred, label)

    def _weighted_BCE(self, pred, label):
        """
        Calculate class weighted BCE loss
        args:
            pred - The predicted binary logits
            label - The pseudo / ground truth label for classification
        """

        pred = pred.view(-1)
        label = label.view(-1)
        pos = label.data.eq(1).nonzero().squeeze().cuda()
        neg = label.data.eq(0).nonzero().squeeze().cuda()

        # Calculate BCE loss for positive positions and negative positions respectively
        loss_pos = self._cls_loss(pred, label, pos)
        loss_neg = self._cls_loss(pred, label, neg)

        return loss_pos * 0.5 + loss_neg * 0.5

    def _IOULoss(self, pred, target, weight=None, return_iou=False):
        """
        Calculate IoU Loss for prediction bounding boxes
        args:
            pred - The predicted bounding box regression map (flattened)
            target - The pseudo / ground truth response map generated from labels (flattened)
            weight - The weight for regression loss calculation (flattened)
            return_iou - Boolean tag indicating whether to calculate and return average IoU for all boxes
        """

        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        ious = (area_intersect + 1.0) / (area_union + 1.0)

        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))

        if weight is not None and weight.sum() > 0:
            iou_loss = (losses * weight).sum() / weight.sum()
            ious = (ious * weight).sum() / weight.sum()
            if return_iou:
                return iou_loss, ious
            else:
                return iou_loss
        else:
            if return_iou:
                return losses.mean(), ious.mean()
            else:
                return losses.mean()

    def add_iouloss(self, bbox_pred, reg_target, reg_weight, return_iou=False):
        """
        Calculate IoU Loss for prediction bounding boxes
        args:
            bbox_pred - The predicted bounding box regression map
            reg_target - The pseudo / ground truth response map generated from labels
            reg_weight - The weight for regression loss calculation
            return_iou - Boolean tag indicating whether to calculate and return average IoU for all boxes
        """

        bbox_pred_flatten = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        reg_target_flatten = reg_target.reshape(-1, 4)
        reg_weight_flatten = reg_weight.reshape(-1)
        pos_inds = torch.nonzero(reg_weight_flatten > 0).squeeze(1)

        bbox_pred_flatten = bbox_pred_flatten[pos_inds]
        reg_target_flatten = reg_target_flatten[pos_inds]

        loss = self._IOULoss(bbox_pred_flatten, reg_target_flatten, return_iou=return_iou)
        if return_iou:
            loss, ious = loss[0], loss[1]

        if loss.isinf() | loss.isnan():
            print("Box loss inf/nan, forcefully setting it to zero.")
            loss = torch.zeros_like(loss)
            if return_iou:
                ious = torch.zeros_like(ious)

        if return_iou:
            return loss, ious
        else:
            return loss

    def grids(self):
        """
        Generate a grid, for each element of feature map on the response map
        The result (grid_to_search_x/y) is shaped B*H*W (the position for each element)
        This grid is useful when generating cls/reg/align labels, or decoding the tracking results from response maps
        """

        # Grid for response map
        sz = self.score_size
        stride = self.stride
        sz_x = sz // 2
        sz_y = sz // 2
        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        self.grid_to_search = {}
        self.grid_to_search_x = x * stride + self.search_sz // 2
        self.grid_to_search_y = y * stride + self.search_sz // 2

        self.grid_to_search_x = torch.Tensor(self.grid_to_search_x).unsqueeze(0).unsqueeze(0).cuda()
        self.grid_to_search_y = torch.Tensor(self.grid_to_search_y).unsqueeze(0).unsqueeze(0).cuda()

        self.grid_to_search_x = self.grid_to_search_x.repeat(self.batch, 1, 1, 1)
        self.grid_to_search_y = self.grid_to_search_y.repeat(self.batch, 1, 1, 1)

    def _pred_offset_to_image_bbox(self, bbox_pred):
        """
        Convert bbox from the predicted response map axis to the image-level axis
        """
        batch = bbox_pred.shape[0]
        self.grid_to_search_x = self.grid_to_search_x[0:batch].to(bbox_pred.device)
        self.grid_to_search_y = self.grid_to_search_y[0:batch].to(bbox_pred.device)

        pred_x1 = self.grid_to_search_x - bbox_pred[:, 0, ...].unsqueeze(1)
        pred_y1 = self.grid_to_search_y - bbox_pred[:, 1, ...].unsqueeze(1)
        pred_x2 = self.grid_to_search_x + bbox_pred[:, 2, ...].unsqueeze(1)
        pred_y2 = self.grid_to_search_y + bbox_pred[:, 3, ...].unsqueeze(1)

        pred = [pred_x1, pred_y1, pred_x2, pred_y2]

        pred = torch.cat(pred, dim=1)

        return pred

    def loss_cls(self, outputs, targets, tag):
        """
        Classification loss
        """
        cls_pred = outputs['pred_logits_' + tag]
        label = outputs['cls_label_' + tag]
        loss_ce = self._weighted_BCE(cls_pred, label)

        if loss_ce.isinf() | loss_ce.isnan():
            print("ce inf/nan")

        loss_ce = torch.where(loss_ce.isinf() | loss_ce.isnan(),
                              torch.zeros_like(loss_ce), loss_ce)

        losses = {'loss_ce': loss_ce}

        return losses

    def loss_align_cls(self, outputs, targets, tag):
        """
        Feature Align Classification loss
        """
        align_label = outputs['align_label_' + tag]
        align_logits = outputs['align_logits_' + tag]
        loss_align = self.criterion(align_logits, align_label.unsqueeze(1))

        if loss_align.isinf() | loss_align.isnan():
            print("align inf/nan")

        loss_align = torch.where(loss_align.isinf() | loss_align.isnan(),
                                 torch.zeros_like(loss_align), loss_align)

        losses = {'loss_align': loss_align}

        return losses

    def loss_boxes(self, outputs, targets, tag):
        """
        Calculate box regression loss and IoU prediction loss
        """

        assert 'pred_boxes_' + tag in outputs
        bbox_pred, reg_label, reg_weight = \
            outputs['pred_boxes_' + tag], outputs['reg_label_' + tag], outputs['reg_weight_' + tag]
        # Calculate bbox regression loss for boxes of only positive indices
        losses = {}

        loss_bbox, ious = self.add_iouloss(bbox_pred, reg_label, reg_weight, return_iou=True)
        losses['loss_bbox'] = loss_bbox
        losses['iou'] = ious

        # IoU prediction loss for normal head is hacked here
        if 'pred_ious_' + tag in outputs:

            # Calculate IoU prediction loss for positive positions
            iou_pred = outputs['pred_ious_' + tag]
            reg_iou_w_gt = outputs['align_label_' + tag]
            iou_weight = outputs['iou_weight_' + tag]
            loss_iou_pred = F.l1_loss(reg_iou_w_gt.detach(), iou_pred.squeeze(1), reduction='none')
            outer_weight = iou_weight - reg_weight
            loss_iou_pred_outer = (loss_iou_pred * outer_weight).sum() / iou_weight.sum()
            loss_iou_pred_inner = (loss_iou_pred * reg_weight).sum() / iou_weight.sum()
            losses['loss_iou_pred'] = loss_iou_pred_outer + loss_iou_pred_inner

        return losses

    def get_loss(self, loss, outputs, targets, tag):
        loss_map = {
            'cls': self.loss_cls,
            'boxes': self.loss_boxes,
            'align_cls': self.loss_align_cls,
        }
        assert loss in loss_map, "do you really want to compute {} loss?".format(loss)
        return loss_map[loss](outputs, targets, tag=tag)

    def forward(self, outputs, targets, tag="sup"):
        """
        This performs the loss computation.
        args:
            outputs - Dict of tensors, see the output specification of the model for the format
            targets - List of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied
            tag - String tag indicating the loss function is for which patches
        """

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, tag=tag))

        return losses


def ocean_loss(settings):
    """
    Construct and Return the criterion for SP-Ocean loss calculation
    """
    losses = ['cls', 'align_cls', 'boxes']
    criterion = SetCriterionOcean(settings=settings, losses=losses)
    device = torch.device(settings.device)
    criterion.to(device)
    return criterion
