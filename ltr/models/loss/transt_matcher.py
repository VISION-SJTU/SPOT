# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from torch import nn
import numpy as np


class TrackingMatcher(nn.Module):
    """
    This class computes an assignment between the ground-truth and the predictions of the network.
    The corresponding feature vectors within the ground-truth box are matched as positive samples.
    This class is written and utilized only by the TransT variant.

    To cater for the IoU prediction branch,
        we additionally add some positions around the labeled gt box as positive positions for IoU prediction.
        This actually simulates some bad boxes with low IoUs with the ground truth, hoping for better filtering.
    """

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, outputs, targets, tag="sup"):
        """
        Performs the matching between the ground-truth and the predictions of the network.

        Params:
            outputs - This is a dict that contains entries including:
                 "logits_{tag}": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits

            targets - This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Always tensor([0]) represents the foreground,
                           since single target tracking has only one foreground category
                 "boxes": Tensor of dim [1, 4] containing the target box coordinates

            tag - A str tag indicating whether inputs comes from supervised instances or unsupervised ones.
                  Note that we only perform IoU prediction for supervised instances, see detailed implementation below.

        Returns:
            indices_result - A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order),
                  and it is always 0, because single target tracking has only one target per image
                For each batch element, it holds: len(index_i) = len(index_j)

            indices_neg_result - A similar list as indices_result with the same structure,
                      but corresponds to negative boxes that outside but around the gt boxes.
        """
        indices = []
        indices_neg = []

        bs, num_queries = outputs["logits_" + tag].shape[:2]
        for i in range(bs):
            cx, cy, w, h = targets[i]['boxes'][0]
            cx = cx.item()
            cy = cy.item()
            w = w.item()
            h = h.item()
            xmin = cx - w / 2
            ymin = cy - h / 2
            xmax = cx + w / 2
            ymax = cy + h / 2
            len_feature = int(np.sqrt(num_queries))
            Xmin = max(int(np.ceil(xmin * len_feature)), 0)
            Ymin = max(int(np.ceil(ymin * len_feature)), 0)
            Xmax = min(int(np.ceil(xmax * len_feature)), len_feature)
            Ymax = min(int(np.ceil(ymax * len_feature)), len_feature)
            if Xmin == Xmax:
                Xmax = Xmax + 1
            if Ymin == Ymax:
                Ymax = Ymax + 1
            query_range = np.arange(0, num_queries, 1)
            query_reshape = query_range.reshape([len_feature, len_feature])

            # Pos index, for bbox regression
            index_pos = query_reshape[Ymin:Ymax, Xmin:Xmax].flatten()
            index_zero = np.zeros(len(index_pos), dtype=int)
            indice = (index_pos, index_zero)
            indices.append(indice)

            if tag == 'sup':
                # Negative indices, for IoU prediction only
                # We sample negative bounding boxes around positive bounding boxes for IoU prediction,
                #     as we find that only using positive bounding boxes causes over-fitting to high IoU predictions
                index_neg_collect = []
                gap = 3
                # Main Logic: neg positions are the difference between a larger rectangle and a smaller rectangle
                Ymin_outer, Ymax_outer, Xmin_outer, Xmax_outer = \
                    max(0, Ymin-gap), min(len_feature, Ymax+gap), max(0, Xmin-gap), min(len_feature, Xmax+gap)
                index_neg_list = [(Ymin_outer, Ymax_outer, Xmin_outer, Xmin),
                                  (Ymin_outer, Ymin, Xmin, Xmax),
                                  (Ymax, Ymax_outer, Xmin, Xmax),
                                  (Ymin_outer, Ymax_outer, Xmax, Xmax_outer)]
                for i_n in index_neg_list:
                    index_neg_collect.append(query_reshape[i_n[0]: i_n[1], i_n[2]: i_n[3]].flatten())
                index_neg = np.concatenate(index_neg_collect)

                index_zero_neg = np.zeros(len(index_neg), dtype=int)
                indice_neg = (index_neg, index_zero_neg)
                indices_neg.append(indice_neg)

        indices_result = [(torch.as_tensor(i, dtype=torch.int64),
                           torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        indices_neg_result = [(torch.as_tensor(i, dtype=torch.int64),
                               torch.as_tensor(j, dtype=torch.int64)) for i, j in indices_neg] if tag == 'sup' else None

        return indices_result, indices_neg_result


def build_matcher():
    return TrackingMatcher()
