import cv2
import numpy as np
import os
import torch

from config.local import EnvironmentSettings

default_setting = EnvironmentSettings()
vis_dataloader_path = os.path.join(default_setting.var_dir, 'loader_test')


def vis_patches(image, anno, name, de_norm):
    """
    Draw cropped template/search patches for debugging, see ltr/models/tracking/xxx_model.py for usage
    The drawn patches with corresponding annotations can be found in 'var/loader_test'
    """
    if de_norm:
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        image = (image * std + mean) * 255
    is_tensor_input = torch.is_tensor(image)
    if is_tensor_input:
        image = image.cpu().numpy()
    image = image.astype(np.uint8)
    image = np.clip(image, 0, 255)
    if is_tensor_input:
        draw_image = np.transpose(image, (1, 2, 0))[:, :, ::-1]
    else:
        draw_image = image[:, :, ::-1]
    draw_image = np.ascontiguousarray(draw_image, dtype=np.uint8)

    if anno is not None:
        x1, y1, w, h = map(lambda x: round(int(x)), anno)
        x2, y2 = x1 + w, y1 + h
        cv2.rectangle(draw_image, (x1, y1), (x2, y2), (255, 215, 0), 2)

    if not os.path.exists(vis_dataloader_path):
        os.makedirs(vis_dataloader_path)
    store_path = os.path.join(vis_dataloader_path, "{}.jpg".format(name))
    cv2.imwrite(store_path, draw_image)
