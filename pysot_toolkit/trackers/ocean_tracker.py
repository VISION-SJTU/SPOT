import os
import cv2
import yaml
import numpy as np
import time
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvisf


class Ocean_Tracker(object):
    def __init__(self, name, net):
        super(Ocean_Tracker, self).__init__()
        self.name = name
        self.net = net
        self.p = OceanConfig()
        self.state = None

    def initialize(self, image, info: dict) -> dict:
        tic = time.time()

        # In: whether input infrared image
        state = dict()
        # Epoch test
        p = OceanConfig()

        state['im_h'] = image.shape[0]
        state['im_w'] = image.shape[1]

        self.grids(p)
        net = self.net
        self.initialize_features()

        bbox = info['init_bbox']
        target_pos = np.array([bbox[0] + bbox[2] / 2,
                               bbox[1] + bbox[3] / 2])
        target_sz = np.array([bbox[2], bbox[3]])

        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))

        avg_chans = np.mean(image, axis=(0, 1))
        z_crop = self.get_subwindow_tracking(image, target_pos, self.p.exemplar_size, s_z, avg_chans)

        # normalize
        z_crop = z_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.inplace = False
        z_crop = tvisf.normalize(z_crop, self.mean, self.std, self.inplace).unsqueeze(0)
        net.template(z_crop.cuda())

        if p.windowing == 'cosine':
            window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))  # [17,17]
        elif p.windowing == 'uniform':
            window = np.ones(int(p.score_size), int(p.score_size))

        state['p'] = p
        state['net'] = net
        state['avg_chans'] = avg_chans
        state['window'] = window
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        self.state = state

        out = {'time': time.time() - tic}

        return out

    def update(self, net, x_crops, target_pos, target_sz, window, scale_z, p):

        cls_score, bbox_pred, cls_align = net.track(x_crops)

        cls_score = F.sigmoid(cls_score).squeeze().cpu().data.numpy()
        cls_align = F.sigmoid(cls_align).squeeze().cpu().data.numpy()
        cls_score = p.ratio * cls_score + (1 - p.ratio) * cls_align

        # bbox to real predict
        bbox_pred = bbox_pred.squeeze().cpu().data.numpy()

        pred_x1 = self.grid_to_search_x - bbox_pred[0, ...]
        pred_y1 = self.grid_to_search_y - bbox_pred[1, ...]
        pred_x2 = self.grid_to_search_x + bbox_pred[2, ...]
        pred_y2 = self.grid_to_search_y + bbox_pred[3, ...]

        # size penalty
        s_c = self.change(self.sz(pred_x2 - pred_x1, pred_y2 - pred_y1) / (self.sz_wh(target_sz)))  # scale penalty
        r_c = self.change((target_sz[0] / target_sz[1]) / ((pred_x2 - pred_x1) / (pred_y2 - pred_y1)))  # ratio penalty

        penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
        pscore = penalty * cls_score

        # window penalty
        pscore = pscore * (1 - p.window_influence) + window * p.window_influence

        # get max
        r_max, c_max = np.unravel_index(pscore.argmax(), pscore.shape)

        # to real size
        pred_x1 = pred_x1[r_max, c_max]
        pred_y1 = pred_y1[r_max, c_max]
        pred_x2 = pred_x2[r_max, c_max]
        pred_y2 = pred_y2[r_max, c_max]

        pred_xs = (pred_x1 + pred_x2) / 2
        pred_ys = (pred_y1 + pred_y2) / 2
        pred_w = pred_x2 - pred_x1
        pred_h = pred_y2 - pred_y1

        diff_xs = pred_xs - p.instance_size // 2
        diff_ys = pred_ys - p.instance_size // 2

        diff_xs, diff_ys, pred_w, pred_h = diff_xs / scale_z, diff_ys / scale_z, pred_w / scale_z, pred_h / scale_z

        target_sz = target_sz / scale_z

        # size learning rate
        lr = penalty[r_max, c_max] * cls_score[r_max, c_max] * p.lr

        # size rate
        res_xs = target_pos[0] + diff_xs
        res_ys = target_pos[1] + diff_ys
        res_w = pred_w * lr + (1 - lr) * target_sz[0]
        res_h = pred_h * lr + (1 - lr) * target_sz[1]

        target_pos = np.array([res_xs, res_ys])
        target_sz = target_sz * (1 - lr) + lr * np.array([res_w, res_h])

        return target_pos, target_sz, cls_score[r_max, c_max]

    def track(self, image, info: dict = None) -> dict:

        state = self.state
        p = state['p']
        net = state['net']
        avg_chans = state['avg_chans']
        window = state['window']
        target_pos = state['target_pos']
        target_sz = state['target_sz']

        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = p.exemplar_size / s_z
        d_search = (p.instance_size - p.exemplar_size) / 2  # slightly different from rpn++
        pad = d_search / scale_z
        s_x = s_z + 2 * pad

        x_crop = self.get_subwindow_tracking(image, target_pos, p.instance_size, self.python2round(s_x), avg_chans)
        # Normalize
        x_crop = x_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        x_crop = tvisf.normalize(x_crop, self.mean, self.std, self.inplace).unsqueeze(0)

        target_pos, target_sz, cls_score = self.update(net, x_crop.cuda(), target_pos,
                                                       target_sz * scale_z, window, scale_z, p)

        target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
        target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
        target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
        target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['p'] = p

        self.state = state

        out_bbox = [target_pos[0] - target_sz[0] / 2,
                    target_pos[1] - target_sz[1] / 2,
                    target_sz[0], target_sz[1]]

        out = {'target_bbox': out_bbox,
               'best_score': cls_score}

        return out

    def grids(self, p):
        """
        each element of feature map on input search image
        :return: H*W*2 (position for each element)
        """
        sz = p.score_size

        # the real shift is -param['shifts']
        sz_x = sz // 2
        sz_y = sz // 2

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        self.grid_to_search_x = x * p.total_stride + p.instance_size // 2
        self.grid_to_search_y = y * p.total_stride + p.instance_size // 2

    def change(self, r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(self, wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    def im_to_torch(self, img):
        img = np.transpose(img, (2, 0, 1))  # C*H*W
        img = torch.from_numpy(img).float()
        return img

    def python2round(self, f):
        """
        use python2 round function in python3
        """
        if round(f + 1) - round(f) != 1:
            return f + abs(f) / f * 0.5
        return round(f)

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.net.initialize()
        self.features_initialized = True

    def get_subwindow_tracking(self, im, pos, model_sz, original_sz, avg_chans, out_mode='torch'):
        """
        SiamFC type cropping
        """
        if isinstance(pos, float):
            pos = [pos, pos]

        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        context_xmin = round(pos[0] - c)
        context_xmax = context_xmin + sz - 1
        context_ymin = round(pos[1] - c)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)

            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch_original = te_im[int(context_ymin):int(context_ymax + 1),
                                int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
        else:
            im_patch = im_patch_original

        if out_mode == "torch":
            return self.im_to_torch(im_patch.copy())
        else:
            return im_patch


class OceanConfig(object):
    penalty_k = 0.062
    window_influence = 0.38
    lr = 0.765
    windowing = 'cosine'
    exemplar_size = 127
    instance_size = 255
    total_stride = 8
    score_size = (instance_size - exemplar_size) // total_stride + 1 + 8
    context_amount = 0.5
    ratio = 0.94
