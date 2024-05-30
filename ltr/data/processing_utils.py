import torch
import math
import cv2 as cv
import torchvision.transforms.functional as trf
import numpy as np


def sample_target(im, target_bb, search_area_factor, output_sz=None, avg=None, device=None, fill_avg=True):
    """
    Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area
    Here is a numpy version where the input images is in Numpy format, often used in dataloader

    args:
        im - Numpy formatted cv image
        target_bb - Target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square).

        avg - The (pre-computed) average channel info pre-computed for filling the paddings in cropped images
        device - The device of the boxes (e.g. cpu, cuda)
        fill_avg - Boolean tag indicating whether to fill average channel in paddings here or not
                    (deprecated, should always be true)

    returns:

        im_patch - Extracted crop
        resize_factor - Float, the factor by which the crop has been resized to make the crop size equal output_size
    """

    assert fill_avg == True, "The fill_avg flag should always be true, previous late filling is deprecated"
    if avg is None:
        avg_chans = np.mean(im, axis=(0, 1))
    else:
        avg_chans = np.array(avg)
    x, y, w, h = target_bb.tolist()

    # Crop image
    w_z = w + (search_area_factor - 1) * ((w + h) * 0.5)
    h_z = h + (search_area_factor - 1) * ((w + h) * 0.5)
    crop_sz = math.ceil(math.sqrt(w_z * h_z))

    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    pos = [x + 0.5 * w, y + 0.5 * h]
    sz = crop_sz
    im_sz = im.shape
    c = (crop_sz + 1) / 2

    context_xmin = np.floor(pos[0] - c + 0.5)
    context_xmax = context_xmin + sz - 1
    context_ymin = np.floor(pos[1] - c + 0.5)
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
        size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
        te_im = np.zeros(size, np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                   int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch = im[int(context_ymin):int(context_ymax + 1),
                   int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(output_sz, crop_sz):
        asset_shape = im_patch.shape
        assert asset_shape[0] > 0 and asset_shape[1] > 0, "Data labeling error emerges!"
        im_patch = cv.resize(im_patch, (output_sz, output_sz))
    resize_factor = output_sz / crop_sz

    return im_patch, resize_factor


def sample_target_tensor(im, target_bb, search_area_factor, output_sz=None, avg=None, device="cpu", fill_avg=True):
    """
    Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area
    Here is a tensor version where the input images is in tensor format, often used in MiddleWare

    args:
        im - Numpy formatted cv image
        target_bb - Target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square).

        avg - The (pre-computed) average channel info pre-computed for filling the paddings in cropped images
        device - The device of the boxes (e.g. cpu, cuda)
        fill_avg - Boolean tag indicating whether to fill average channel in paddings here or not
                    (deprecated, should always be true)

    returns:

        im_patch - Extracted crop
        resize_factor - Float, the factor by which the crop has been resized to make the crop size equal output_size
        fill_info - The top/bottom/left/right padding length, for later average channel filling in GPUs
    """

    assert fill_avg == True, "The fill_avg flag should always be true, previous late filling is deprecated"
    if avg is None:
        im_float = im.float()
        avg_chans = torch.mean(im_float, dim=(1, 2))
    else:
        avg_chans = avg

    x, y, w, h = target_bb.tolist()

    # Crop image
    w_z = w + (search_area_factor - 1) * ((w + h) * 0.5)
    h_z = h + (search_area_factor - 1) * ((w + h) * 0.5)
    crop_sz = math.ceil(math.sqrt(w_z * h_z))

    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    pos = [x + 0.5 * w, y + 0.5 * h]
    sz = crop_sz
    c = (crop_sz + 1) / 2

    context_xmin = np.floor(pos[0] - c + 0.5)
    context_xmax = context_xmin + sz - 1
    context_ymin = np.floor(pos[1] - c + 0.5)
    context_ymax = context_ymin + sz - 1

    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))

    k, r, c = im.shape
    right_pad = int(max(0., context_xmax - c + 1))
    bottom_pad = int(max(0., context_ymax - r + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    if any([top_pad, bottom_pad, left_pad, right_pad]):
        size = (k, r + top_pad + bottom_pad, c + left_pad + right_pad)
        te_im = torch.zeros(size, dtype=torch.uint8, device=device)
        te_im[:, top_pad:top_pad + r, left_pad:left_pad + c] = im
        if top_pad:
            te_im[:, 0:top_pad, left_pad:left_pad + c] = avg_chans[:, None, None]
        if bottom_pad:
            te_im[:, r + top_pad:, left_pad:left_pad + c] = avg_chans[:, None, None]
        if left_pad:
            te_im[:, :, 0:left_pad] = avg_chans[:, None, None]
        if right_pad:
            te_im[:, :, c + left_pad:] = avg_chans[:, None, None]
        im_patch = te_im[:, int(context_ymin):int(context_ymax + 1),
                   int(context_xmin):int(context_xmax + 1)]
    else:
        im_patch = im[:, int(context_ymin):int(context_ymax + 1),
                   int(context_xmin):int(context_xmax + 1)]

    if not np.array_equal(output_sz, crop_sz):
        im_patch = cv.resize(im_patch, (output_sz, output_sz))
    resize_factor = output_sz / crop_sz

    return im_patch, resize_factor


def transform_image_to_crop(box_in, box_extract, resize_factor, crop_sz):
    """
    Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image

    args:
        box_in - Tensor, the box in original images, for which the co-ordinates are to be transformed
        box_extract - Tensor, the box about which the image crop has been extracted
        resize_factor - Float of the ratio between the original image scale and the scale of the image crop
        crop_sz - Tensor, size of the cropped image

    returns:
        box_out - transformed co-ordinates of box_in, in cropped image coordinate
    """

    box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]

    box_in_center = box_in[0:2] + 0.5 * box_in[2:4]

    box_out_center = (crop_sz - 1) / 2 + (box_in_center - box_extract_center) * resize_factor
    box_out_wh = box_in[2:4] * resize_factor

    box_out = torch.cat((box_out_center - 0.5 * box_out_wh, box_out_wh))
    return box_out


def recover_bbox_to_original(box_crop, box_extract, resize_factor, crop_sz):
    """
    Transform the box co-ordinates from the co-ordinates of the cropped image,
        back to the original image co-ordinates. All bboxes are formatted [x, y, w, h].

    args:
        box_crop - Tensor, the box for which the co-ordinates are to be reversely transformed to original image
        box_extract - Tensor, the box about which the image crop has been extracted
        resize_factor - Float of the ratio between the original image scale and the scale of the image crop
        crop_sz - Tensor, size of the cropped image

    returns:
        box_original - Tensor, transformed co-ordinates in the original image
    """

    box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]
    box_crop_center = box_crop[0:2] + 0.5 * box_crop[2:4]

    box_original_wh = box_crop[2:4] / resize_factor
    box_original_center = (box_crop_center - (crop_sz - 1) / 2) / resize_factor + box_extract_center

    box_original = torch.cat((box_original_center - 0.5 * box_original_wh, box_original_wh))
    box_original = torch.clamp(box_original, min=0.0)

    return box_original


def jittered_center_crop(frames, box_extract, box_gt, search_area_factor, output_sz, avg=None,
                         input_type="numpy", device="cpu", recoverable=False, fill_avg=True):
    """
    For each frame in frames, extracts a square crop centered at box_extract, of area search_area_factor^2
    times box_extract area. The extracted crops are then resized to output_sz.
    Further, the co-ordinates of the box box_gt are transformed to the image crop co-ordinates.
    Beyond above, we also add features to:
        1. enable to recover the boxes from search area coordinate to original image coordinate

    args:
        frames - List of frames
        box_extract - List of boxes of same length as frames. The crops are extracted using anno_extract
        box_gt - List of boxes of same length as frames. The co-ordinates of these boxes are transformed from
                    image co-ordinates to the crop co-ordinates
        search_area_factor - The area of the extracted crop is search_area_factor^2 times box_extract area
        output_sz - The size to which the extracted crops are resized
        avg - The (pre-computed) average channel info pre-computed for filling the paddings in cropped images

        input_type - String indicating whether the input format is in "numpy" or "tensor"
        device - The device of the boxes (e.g. cpu, cuda)

        recoverable - Boolean tag indicating whether to return recover_info or not
        fill_avg - Boolean tag indicating whether to fill average channel in paddings here or not
                   (deprecated, should always be true)


    returns:
        frames_crop - List of image crops
        box_crop - List of box (gt box or coarse box) location in the crop co-ordinates
        recover_info - Info for recover the box in unlabeled frames,
                       for decoding the tracking results from boxes in search areas to boxes in original images
        """

    assert fill_avg == True, "The fill_avg flag should always be true, previous late filling is deprecated"
    sample_target_function = sample_target_tensor if input_type == "tensor" else sample_target

    if avg is not None:
        crops_resize_factors = [sample_target_function(f, a, search_area_factor, output_sz,
                                                       device=device, avg=vg, fill_avg=fill_avg)
                                for f, a, vg in zip(frames, box_extract, avg)]
    else:
        crops_resize_factors = [sample_target_function(f, a, search_area_factor, output_sz,
                                                       device=device, fill_avg=fill_avg)
                                for f, a in zip(frames, box_extract)]
    frames_crop, resize_factors = zip(*crops_resize_factors)
    crop_sz = torch.tensor([output_sz, output_sz])

    # Find the bounding box location in the cropped patch
    box_crop = [transform_image_to_crop(a_gt, a_ex, rf, crop_sz)
                for a_gt, a_ex, rf in zip(box_gt, box_extract, resize_factors)]

    if not recoverable:
        return frames_crop, box_crop, None
    else:
        # Recover_info helps recover boxes from crop image coordinate to original image coordinate
        recover_info = {
            'box_extract': box_extract,
            'resize_factors': [torch.tensor(rf, device=device) for rf in resize_factors]
        }
        return frames_crop, box_crop, recover_info


def transform_box_to_crop(box, crop_box, crop_sz):
    """
    Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image

    args:
        box - Tensor, the box for which the co-ordinates are to be transformed
        crop_box - Tensor, bounding box defining the crop in the original image
        crop_sz - Tensor, size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """

    box_out = box.clone()
    box_out[:2] -= crop_box[:2]

    scale_factor = crop_sz / crop_box[2:]

    box_out[:2] *= scale_factor
    box_out[2:] *= scale_factor
    return box_out


def get_jittered_box(box, mode, scale_jitter_factor, center_jitter_factor, device="cpu"):
    """
    Jitter the input box

    args:
        box - input bounding box
        mode - string 'search' or 'template' indicating search or template data
        center_jitter_factor - The amount of jittering to be applied to
                                  the target center before extracting the search region.
        scale_jitter_factor - The amount of jittering to be applied to
                                  the target size before extracting the search region.
        device - The device of the input boxes (e.g. cpu, cuda)

    returns:
        torch.Tensor - jittered box
    """
    jittered_size = box[2:4] * torch.exp(torch.randn(2, device=device) * scale_jitter_factor[mode])
    max_offset = (jittered_size.sum() * 0.5 * torch.tensor(center_jitter_factor[mode]).float())
    jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2, device=device) - 0.5)
    return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)
