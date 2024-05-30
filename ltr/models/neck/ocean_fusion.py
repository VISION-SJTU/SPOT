# -----------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# Revised by Jilai Zheng (zhengjilai@sjtu.edu.cn), for SPOT
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dcn import DeformConv
from ltr.utils.tensordict import TensorDict


class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x, crop=False):
        x_ori = self.downsample(x)
        if x_ori.size(3) < 20 and crop:
            l = 4
            r = -4
            xf = x_ori[:, :, l:r, l:r]

        if not crop:
            return x_ori
        else:
            return x_ori, xf


class box_tower(nn.Module):
    """
    box tower for FCOS reg
    """

    def __init__(self, inchannels=512, outchannels=256, towernum=1, iou_prediction=True):
        super(box_tower, self).__init__()

        # Whether to build an IoU prediction head
        self.iou_prediction = iou_prediction

        # Encode backbone
        tower = []
        cls_tower = []
        self.cls_encode = matrix(in_channels=inchannels, out_channels=outchannels)
        self.reg_encode = matrix(in_channels=inchannels, out_channels=outchannels)
        self.cls_dw = GroupDW(in_channels=inchannels)
        self.reg_dw = GroupDW(in_channels=inchannels)

        # Box pred head
        for i in range(towernum):
            tower.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1))
            tower.append(nn.BatchNorm2d(outchannels))
            tower.append(nn.ReLU())

        # Cls tower
        for i in range(towernum):
            cls_tower.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1))
            cls_tower.append(nn.BatchNorm2d(outchannels))
            cls_tower.append(nn.ReLU())

        self.add_module('bbox_tower', nn.Sequential(*tower))
        self.add_module('cls_tower', nn.Sequential(*cls_tower))

        # Reg head
        self.bbox_pred = nn.Conv2d(outchannels, 4, kernel_size=3, stride=1, padding=1)
        self.cls_pred = nn.Conv2d(outchannels, 1, kernel_size=3, stride=1, padding=1)

        # Adjust scale
        self.adjust = nn.Parameter(0.1 * torch.ones(1))
        self.bias = nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)).cuda())

        if self.iou_prediction:
            iou_pred_tower = []
            # IoU pred head
            for i in range(towernum):
                iou_pred_tower.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1))
                iou_pred_tower.append(nn.BatchNorm2d(outchannels))
                iou_pred_tower.append(nn.ReLU())
            self.add_module('iou_pred_tower', nn.Sequential(*iou_pred_tower))
            self.iou_pred = nn.Conv2d(outchannels, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, search, kernel, iou_prediction=True, batch_iou=0):

        # Encode first
        cls_z, cls_x = self.cls_encode(kernel, search)  # [z11, z12, z13]
        reg_z, reg_x = self.reg_encode(kernel, search)  # [x11, x12, x13]

        # Cls and reg DW
        cls_dw = self.cls_dw(cls_z, cls_x)
        reg_dw = self.reg_dw(reg_z, reg_x)
        x_reg = self.bbox_tower(reg_dw)
        x = self.adjust * self.bbox_pred(x_reg) + self.bias
        x = torch.clamp_max(x, max=4.7)
        x = torch.exp(x)

        # Cls tower
        c = self.cls_tower(cls_dw)
        cls = 0.1 * self.cls_pred(c)

        # Extra outputs
        extra_outputs = TensorDict()

        # IoU prediction tower
        if self.iou_prediction and iou_prediction and batch_iou > 0:
            iou_tower = self.iou_pred_tower(reg_dw.detach()[:batch_iou])
            ious = self.iou_pred(iou_tower)
            extra_outputs['pred_ious'] = ious.sigmoid()

        return x, cls, cls_dw, x_reg, extra_outputs


class matrix(nn.Module):
    """
    encode backbone feature
    """

    def __init__(self, in_channels, out_channels):
        super(matrix, self).__init__()

        # Same size (11)
        self.matrix11_k = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.matrix11_s = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Size: h/2, w
        self.matrix12_k = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, dilation=(2, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.matrix12_s = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, dilation=(2, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Size: w/2, h
        self.matrix21_k = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, dilation=(1, 2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.matrix21_s = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, dilation=(1, 2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, z, x):
        z11 = self.matrix11_k(z)
        x11 = self.matrix11_s(x)

        z12 = self.matrix12_k(z)
        x12 = self.matrix12_s(x)

        z21 = self.matrix21_k(z)
        x21 = self.matrix21_s(x)

        return [z11, z12, z21], [x11, x12, x21]


class AdaptiveConv(nn.Module):
    """
    Adaptive Conv is built based on Deformable Conv
    with precomputed offsets which derived from anchors
    """

    def __init__(self, in_channels, out_channels):
        super(AdaptiveConv, self).__init__()
        self.conv = DeformConv(in_channels, out_channels, 3, padding=1)

    def forward(self, x, offset):
        N, _, H, W = x.shape
        assert offset is not None
        assert H * W == offset.shape[1]
        # reshape [N, NA, 18] to (N, 18, H, W)
        offset = offset.permute(0, 2, 1).reshape(N, -1, H, W)
        x = self.conv(x, offset)

        return x


class AlignHead(nn.Module):
    # Align features and classification score

    def __init__(self, in_channels, feat_channels):
        super(AlignHead, self).__init__()

        self.rpn_conv = AdaptiveConv(in_channels, feat_channels)
        self.rpn_cls = nn.Conv2d(feat_channels, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, offset):
        x = self.relu(self.rpn_conv(x, offset))
        cls_score = self.rpn_cls(x)
        return cls_score


class GroupDW(nn.Module):
    """
    encode backbone feature
    """

    def __init__(self, in_channels=256):
        super(GroupDW, self).__init__()
        self.weight = nn.Parameter(torch.ones(3))

    def forward(self, z, x):
        z11, z12, z21 = z
        x11, x12, x21 = x

        re11 = xcorr_depthwise(x11, z11)
        re12 = xcorr_depthwise(x12, z12)
        re21 = xcorr_depthwise(x21, z21)
        re = [re11, re12, re21]

        weight = F.softmax(self.weight, 0)

        s = 0
        for i in range(3):
            s += weight[i] * re[i]

        return s


def xcorr_depthwise(x, kernel):
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch * channel, x.size(2), x.size(3))
    kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch * channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out
