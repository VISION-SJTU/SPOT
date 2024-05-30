import numpy as np
import torch.nn as nn
import torch, random
from ltr.utils.box_ops import box_xywh_to_xyxy


class OceanLabelGenerator(nn.Module):
    """
    The class for generating ground truth labels for Ocean, often called in network in multiple GPUs
    """

    def __init__(self, settings):
        super(OceanLabelGenerator, self).__init__()

        self.stride = settings.stride
        self.output_sz = settings.output_sz
        self.batch = settings.batch_size
        self.search_sz = settings.search_sz

        self.grids()

    def forward(self, data_dict=None, tag="cls"):
        """
        Main Logic for generating different kinds of labels

        args:
            data_dict - A dict containing necessary information for generating labels
            tags - The tag indicating which label(s) to generate, e.g. cls, reg, iou_w, align
        returns:
            tensors - Different kinds of labels generated based on input ground truth / pseudo labels
        """

        if tag == "cls":
            return self._construct_cls_labels(data_dict['anno'])
        elif tag == "reg":
            return self._construct_reg_labels(data_dict['anno'])
        elif tag == "iou_w":
            return self._construct_iou_weights(data_dict['anno'])
        elif tag == "align":
            return self._construct_align_labels(data_dict['bbox_pred'],
                                                data_dict['reg_label'],
                                                data_dict['reg_weight'])

    @torch.no_grad()
    def _construct_cls_labels(self, search_anno):
        """
        Generate binary classification labels for Ocean
        """
        search_anno_xyxy = box_xywh_to_xyxy(search_anno)
        return self._dynamic_label(search_anno_xyxy)

    @torch.no_grad()
    def _construct_reg_labels(self, search_anno):
        """
        Generate regression labels for Ocean
        """
        search_anno_xyxy = box_xywh_to_xyxy(search_anno)
        return self._reg_label(search_anno_xyxy)

    @torch.no_grad()
    def _construct_iou_weights(self, search_anno):
        """
        Generate non_zero positions of IoU prediction for Ocean
        """
        search_anno_xyxy = box_xywh_to_xyxy(search_anno)
        return self._iou_weight(search_anno_xyxy)

    def _construct_align_labels(self, bbox_pred, reg_label, reg_weight):
        """
        Generate align branch labels for Ocean, please refer to Ocean paper for details

        Note that, we check the original Ocean implementation, and find the gradient is not detached here
        Thus, this branch in fact has some interaction with the regression branch
        Please refer to TracKit (https://github.com/researchmm/TracKit) for more details
        """

        # Calc predicted box iou (treat it as aligned label)
        pred = bbox_pred.permute(0, 2, 3, 1)  # [B, H, W, 4]
        pred_left = pred[..., 0]
        pred_top = pred[..., 1]
        pred_right = pred[..., 2]
        pred_bottom = pred[..., 3]

        target_left = reg_label[..., 0]
        target_top = reg_label[..., 1]
        target_right = reg_label[..., 2]
        target_bottom = reg_label[..., 3]

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        ious = (area_intersect + 1.0) / (area_union + 1.0)

        # Delete points out of object
        ious = torch.abs(reg_weight * ious)
        ious = torch.clamp(ious, max=1, min=0)

        return ious

    def grids(self):
        """
        Generate a grid, for each element of feature map on the response map
        The result (grid_to_search_x/y) is shaped B*H*W (the position for each element)
        This grid is useful when generating cls/reg labels
        """
        # Response map grid
        sz = self.output_sz
        sz_x = sz // 2
        sz_y = sz // 2
        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))
        self.grid_to_search = {}
        self.grid_to_search_x = torch.tensor(x * self.stride + self.search_sz // 2).unsqueeze(0).cuda()
        self.grid_to_search_y = torch.tensor(y * self.stride + self.search_sz // 2).unsqueeze(0).cuda()
        self.grid_to_search_x = self.grid_to_search_x.repeat(self.batch, 1, 1)
        self.grid_to_search_y = self.grid_to_search_y.repeat(self.batch, 1, 1)

    def _reg_label(self, bbox):
        """
        Generate regression labels for Ocean

        args:
            bbox - A tensor for box labels, shaped a batch of [x1, y1, x2, y2], sized [b, 4]
        returns:
            reg_label - A tensor map for regression labels, batch of [l, t, r, b], sized [b, H, W, 4]
                        Here l, t, r, b denotes the distance from center position to left, top, right, bottom box edge
            inds_nonzero - A tensor map indicating which positions are inside boxes,
                                  and thereby should be viewed as positive instances for regression learning
                        The output tensor is shaped [b, H, W], corresponding to reg_label
        """

        self.grid_to_search_x = self.grid_to_search_x.to(bbox.device)
        self.grid_to_search_y = self.grid_to_search_y.to(bbox.device)

        x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        b = bbox.shape[0]
        l = self.grid_to_search_x[:b, :] - x1[:, None, None]
        t = self.grid_to_search_y[:b, :] - y1[:, None, None]
        r = x2[:, None, None] - self.grid_to_search_x[:b, :]
        b = y2[:, None, None] - self.grid_to_search_y[:b, :]

        l, t, r, b = l.unsqueeze(-1), t.unsqueeze(-1), r.unsqueeze(-1), b.unsqueeze(-1)

        reg_label = torch.cat([l, t, r, b], dim=3)
        reg_label_min = torch.min(reg_label, dim=3).values
        inds_nonzero = (reg_label_min > 0).float()

        return reg_label, inds_nonzero

    def _iou_weight(self, bbox, expand_scale=0.2):
        """
        Generate non_zero positions of IoU prediction for Ocean
        The implementation logic is quite similar to self._reg_label()
        However, we actually expand the area of positive positions with a expand_scale, to cover some negative boxes

        args:
            bbox - A tensor for box labels, shaped a batch of [x1, y1, x2, y2], sized [b, 4]
            expand_scale - A float indicating how much area we expand around gt box for positive IoU prediction
        returns:
            inds_nonzero - A tensor map indicating which positions are inside boxes or slightly around the boxes,
                             and thereby should be viewed as positive instances for iou prediction. Shaped [b, H, W].
        """

        self.grid_to_search_x = self.grid_to_search_x.to(bbox.device)
        self.grid_to_search_y = self.grid_to_search_y.to(bbox.device)

        x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        w, h = torch.clamp_min(x2 - x1, min=1), torch.clamp_min(y2 - y1, min=1)
        x1_outer, x2_outer = x1 - expand_scale * w, x2 + expand_scale * w
        y1_outer, y2_outer = y1 - expand_scale * h, y2 + expand_scale * h

        b = bbox.shape[0]
        l = self.grid_to_search_x[:b, :] - x1_outer[:, None, None]
        t = self.grid_to_search_y[:b, :] - y1_outer[:, None, None]
        r = x2_outer[:, None, None] - self.grid_to_search_x[:b, :]
        b = y2_outer[:, None, None] - self.grid_to_search_y[:b, :]

        l, t, r, b = l.unsqueeze(-1), t.unsqueeze(-1), r.unsqueeze(-1), b.unsqueeze(-1)

        reg_label = torch.cat([l, t, r, b], dim=3)
        reg_label_min = torch.min(reg_label, dim=3).values
        inds_nonzero = (reg_label_min > 0).float()

        return inds_nonzero

    def _dynamic_label(self, bbox, rPos=2):
        """
        Generate binary classification labels for Ocean

        args:
            bbox - A tensor for box labels, shaped a batch of [x1, y1, x2, y2], sized [b, 4]
            rPos - An integer indicating in which radius around the box center the labels should be position
        returns:
            cls_label - The tensor classification label, 1 for positive, 0 for negative, shaped [b, H, W]
        """

        x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        # Center shift compared to center point, typically (127, 127) in Ocean
        shift_x = cx - self.search_sz / 2
        shift_y = cy - self.search_sz / 2

        d_label = self._create_dynamic_logisticloss_label(self.output_sz, shift_x, shift_y, rPos)

        return d_label

    def _create_dynamic_logisticloss_label(self, label_size, shift_x, shift_y, rPos=2):

        sz_x = label_size // 2 + (shift_x / self.stride).long()  # Typically, 8 is the stride
        sz_y = label_size // 2 + (shift_y / self.stride).long()

        # TODO: find non-loop (batch-wise) methods to conduct the following label generation
        feedback = []
        device = shift_x.device
        for i in range(len(shift_x)):
            x, y = torch.meshgrid(torch.arange(0, label_size, device=device) - sz_x[i],
                                  torch.arange(0, label_size, device=device) - sz_y[i])

            dist_to_center = torch.abs(x) + torch.abs(y)  # Block metric
            label = torch.where(dist_to_center <= rPos,
                                torch.ones_like(y, device=device),
                                torch.zeros_like(y, device=device)).transpose(1, 0)
            feedback.append(label)
        cls_label = torch.stack(feedback)

        return cls_label.float()
