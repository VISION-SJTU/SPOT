import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class IndependentTransform(nn.Module):
    """
    Independent transform for template patch OR search patch
    Typically, for IndependentTransform, there should exist no public random sampling for both patches.

    The following example help you figure out the difference between IndependentTransform and JointTransform.
        1. EllipseRotation is typically performed on search patches only ---> IndependentTransform
        2. Grayscale augmentation should be consistent to both search and template patches ---> JointTransform
    """

    def __init__(self, aug_list):
        """
        args:
            aug_list: A List of nn.Module, for independently augmenting the patches in multiple GPUs
                      Please refer to the following examples to implement your own augmentations.
        """

        super().__init__()

        self.independent_augs = nn.ModuleList([])
        for aug in aug_list:
            self.independent_augs.append(aug)

    @torch.no_grad()
    def forward(self, data, tag='template', masks=None):
        """
        args:
            data - A TensorDict containing information for patches, should at least including:

                'masks': A Boolean tensor sized [batch], indicating whether the certain instance needs aug or not.
                         For example, for batch = 3, and data['mask'] = Tensor[True, True, False],
                                      then only first two instance in batch performs augmentation.
                         The mask field is often used to distinguish supervised and unsupervised pairs,
                                      as in our design, strong augmentation is performed only on unsupervised pairs.
                '{}_images': Cropped patches
                '{}_anno': Annotations of target object. Coarse boxes for unlabeled frames
                '{}_avg': Average channel info for filling the crops
                Note that here brace should be filled with a string tag, e.g. template, search

            tag - A string indicating which patch to perform augmentation, e.g. template, search

        returns:
            image, bbox - Augmented image patches and labels
        """

        # Fetch all materials
        mask = data['masks'] if masks is None else masks
        image, bbox, avg = data['{}_images'.format(tag)], \
                           data['{}_anno'.format(tag)], data['{}_avg'.format(tag)]
        # Conduct augmentations
        for aug in self.independent_augs:
            image, bbox, avg, mask = aug(image, bbox, avg, mask)

        return image, bbox


class JointTransform(nn.Module):
    """
    Joint transform for BOTH template patch AND search patch
    Typically, for JointTransform, there should exist public random sampling for both patches.

    The following example help you figure out the difference between IndependentTransform and JointTransform.
        1. EllipseRotation is typically performed on search patches only ---> IndependentTransform
        2. Grayscale augmentation should be consistent to both search and template patches ---> JointTransform
    """

    def __init__(self, aug_list):
        """
        args:
            aug_list: A List of nn.Module, for jointly augmenting the patches in multiple GPUs
                      Please refer to the following examples to implement your own augmentations.
        """

        super().__init__()

        self.joint_augs = nn.ModuleList([])
        for aug in aug_list:
            self.joint_augs.append(aug)

    @torch.no_grad()
    def forward(self, data, tags=None, masks=None):
        """
        args:
            data - A TensorDict containing information for patches, should at least including:

                'masks': A Boolean tensor sized [batch], indicating whether the certain instance needs aug or not.
                         For example, for batch = 3, and data['mask'] = Tensor[True, True, False],
                                      then only first two instance in batch performs augmentation.
                         The mask field is often used to distinguish supervised and unsupervised pairs,
                                      as in our design, strong augmentation is performed only on unsupervised pairs.
                'template_images', 'search_images': Cropped patches
                'template_anno', 'search_anno': Annotations of target object. Coarse boxes for unlabeled frames
                'template_avg', 'search_avg': Average channel info for filling the crops

        returns:
            image_t, bbox_t, avg_t, image_s, bbox_s, avg_s - Augmented image patches, labels and average channels
        """

        # Fetch all materials
        if tags is None:
            tags = ["template", "search"]
        image_groups = [data[s + "_images"] for s in tags]
        anno_groups = [data[s + "_anno"] for s in tags]
        avg_groups = [data[s + "_avg"] for s in tags]
        mask = data['masks'] if masks is None else masks

        # Conduct augmentations
        for aug in self.joint_augs:
            image_groups, anno_groups, avg_groups, mask = \
                aug(image_groups, anno_groups, avg_groups, mask)

        return image_groups, anno_groups, avg_groups


class EllipseRotationAffine(nn.Module):
    """
        EllipseRotationAffine Transform, An independent transform combination of:

        1. Probabilistic Left / right Flips
        2. Probabilistic Up / down Flips
        3. Ellipse Rotation Transform, using Ellipse for bbox conversion, suggested by:

            Kalra, Agastya, et al. "Towards Rotation Invariance in Object Detection."
            Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.

        Thanks for their great work!

        In SPOT, This augmentation is typically performed only on unsupervised pairs only.
    """

    def __init__(self, max_angle=0, lr_flip_prob=0.0, ud_flip_prob=0.0):
        """
        args:
            max_angle - The maximum rotation degree (typically, the rotation should not be too large)
            lr_flip_prob - The probability of left / right flip
            ud_flip_prob - The probability of up / down flip
        """

        super().__init__()
        self.max_angle = abs(int(max_angle))
        self.lr_flip_prob = lr_flip_prob
        self.ud_flip_prob = ud_flip_prob

    def forward(self, image, bbox, avg, mask):

        # No strong augmentation is needed (typically for baseline ablation)
        if not any(mask):
            return image, bbox, avg, mask

        batch, c, h, w = image.shape
        device = image.device

        # Sample an angle to rotate. A Trick: Up/down flip is a kind of 180 degree rotation
        # In SPOT, all strong augmentation is conducted on unsupervised pairs only
        base_angle = 180 * torch.where(torch.rand([batch], device=device) <= self.ud_flip_prob,
                                       torch.ones([batch], device=device),
                                       torch.zeros([batch], device=device))
        angle = torch.randint(-self.max_angle, self.max_angle + 1, [batch], device=device) + base_angle
        # Sample whether to conduct Left/right flip
        lr_flip_tag = torch.rand([batch], device=device) <= self.lr_flip_prob

        # Transform image
        for i in range(batch):
            # Strong augmentation only for unsupervised instances in SPOT, so we need the key mask here.
            # TODO: find some non-loop methods to do such affine transformation
            if mask[i]:
                flip_tag = [0.0, 0.0]
                if lr_flip_tag[i]:
                    image[i] = image[i].flip((2,))
                image[i] = self._affine_with_certain_center(image[i], angle=int(angle[i]), center=[0.5, 0.5],
                                                            translate=flip_tag, scale=1.0,
                                                            fill=list(avg[i]), shear=[0, 0])

        # Now begin to transform bbox, following ellipse methodology (suggested by ICCV2021 paper)
        rwrh = 1 / 2 * bbox[:, 2:]
        cxcy = bbox[:, :2] + rwrh
        # Transform the center cx, if a left/right flip is sampled
        lr_flip_cx_cy = (w - 1) - cxcy[:, 0]
        cxcy[:, 0] = torch.where(lr_flip_tag, lr_flip_cx_cy, cxcy[:, 0])

        # Sample all points on the Ellipse
        a_range = torch.arange(0, 360, 1, device=device) * (math.pi / 180)
        sin_a_range = torch.sin(a_range).expand([batch, 360])
        cos_a_range = torch.cos(a_range).expand([batch, 360])

        # Ellipse axis for x and y, ranging from angle 0 to 360
        x_axis_ellipse = rwrh[:, 0][:, None] * sin_a_range + cxcy[:, 0][:, None]
        y_axis_ellipse = rwrh[:, 1][:, None] * cos_a_range + cxcy[:, 1][:, None]

        # Affine matrix (rotate around the center [0.5, 0.5] of the patch)
        center = torch.tensor([[w * 0.5, h * 0.5]], device=device, dtype=rwrh.dtype).expand(batch, 2)
        matrix = self._get_rotation_matrix(center, angle, 1.0)
        axis_xy = torch.cat([x_axis_ellipse.unsqueeze(1), y_axis_ellipse.unsqueeze(1),
                             torch.ones(x_axis_ellipse.shape, device=device).unsqueeze(1)], dim=1).float()

        # Prepare the vector to be transformed
        transformed_axis = torch.bmm(matrix, axis_xy)

        # Ellipse rotation advised by the ICCV2021 paper
        trans_max = transformed_axis.max(dim=2).values
        trans_min = transformed_axis.min(dim=2).values

        # Transform back to [x, y, w, h] format, with batch dim before that
        new_bbox = torch.stack([trans_min[:, 0], trans_min[:, 1], trans_max[:, 0] - trans_min[:, 0],
                                trans_max[:, 1] - trans_min[:, 1]], dim=1)
        # Use the transformed boxes only for those instances with mask[i] == True
        output_bbox = torch.where(mask[:, None], new_bbox, bbox)
        output_bbox = torch.clamp_min(output_bbox, min=0.0)

        return image, output_bbox, avg, mask

    def _get_rotation_matrix(self, center, angle, scale):

        # Angle
        angle_rad = angle * (math.pi / 180)
        alpha = torch.cos(angle_rad) * scale
        beta = torch.sin(angle_rad) * scale

        # Unpack the center to x, y coordinates
        x, y = center[:, 0], center[:, 1]

        # Create output tensor
        batch_size, _ = center.shape
        M = torch.zeros(batch_size, 2, 3, device=center.device).float()
        M[:, 0, 0] = alpha
        M[:, 0, 1] = beta
        M[:, 0, 2] = (1. - alpha) * x - beta * y
        M[:, 1, 0] = -beta
        M[:, 1, 1] = alpha
        M[:, 1, 2] = beta * x + (1. - alpha) * y

        return M

    def _affine_with_certain_center(self, img, angle, translate, scale, shear, center,
                                    interpolation="nearest", fill=None):

        # Rotate around the center ([0.5, 0.5])
        matrix = self._get_inverse_affine_matrix(center, -angle, translate, scale, shear)
        return self.affine(img, matrix=matrix, interpolation=interpolation, fill=fill)

    def _get_inverse_affine_matrix(self, center, angle, translate, scale, shear):

        # Helper method to compute inverse matrix for affine transformation (copied from tvisf)
        rot = math.radians(angle)
        sx, sy = [math.radians(s) for s in shear]

        cx, cy = center
        tx, ty = translate

        # RSS without scaling
        a = math.cos(rot - sy) / math.cos(sy)
        b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
        c = math.sin(rot - sy) / math.cos(sy)
        d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

        # Inverted rotation matrix with scale and shear
        # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
        matrix = [d, -b, 0.0, -c, a, 0.0]
        matrix = [x / scale for x in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
        matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += cx
        matrix[5] += cy

        return matrix

    def _gen_affine_grid(self, theta, w, h, ow, oh):

        d = 0.5
        base_grid = torch.empty(1, oh, ow, 3, dtype=theta.dtype, device=theta.device)
        x_grid = torch.linspace(-ow * 0.5 + d, ow * 0.5 + d - 1, steps=ow, device=theta.device)
        base_grid[..., 0].copy_(x_grid)
        y_grid = torch.linspace(-oh * 0.5 + d, oh * 0.5 + d - 1, steps=oh, device=theta.device).unsqueeze_(-1)
        base_grid[..., 1].copy_(y_grid)
        base_grid[..., 2].fill_(1)

        rescaled_theta = theta.transpose(1, 2) / torch.tensor([0.5 * w, 0.5 * h], dtype=theta.dtype,
                                                              device=theta.device)
        output_grid = base_grid.view(1, oh * ow, 3).bmm(rescaled_theta)
        return output_grid.view(1, oh, ow, 2)

    def affine(self, img, matrix, interpolation="nearest", fill=None):

        dtype = img.dtype if torch.is_floating_point(img) else torch.float32
        theta = torch.tensor(matrix, dtype=dtype, device=img.device).reshape(1, 2, 3)
        shape = img.shape

        # Grid will be generated on the same device as theta and img
        grid = self._gen_affine_grid(theta, w=shape[-1], h=shape[-2], ow=shape[-1], oh=shape[-2])
        return self._apply_grid_transform(img, grid, interpolation, fill=fill)

    def _apply_grid_transform(self, img, grid, mode, fill):

        img, need_cast, need_squeeze, out_dtype = self._cast_squeeze_in(img, [grid.dtype, ])

        if img.shape[0] > 1:
            # Apply same grid to a batch of images
            grid = grid.expand(img.shape[0], grid.shape[1], grid.shape[2], grid.shape[3])

        # Append a dummy mask for customized fill colors, should be faster than grid_sample() twice
        if fill is not None:
            dummy = torch.ones((img.shape[0], 1, img.shape[2], img.shape[3]), dtype=img.dtype, device=img.device)
            img = torch.cat((img, dummy), dim=1)

        img = F.grid_sample(img, grid, mode=mode, padding_mode="zeros", align_corners=False)

        # Fill with required color
        if fill is not None:
            mask = img[:, -1:, :, :]  # N * 1 * H * W
            img = img[:, :-1, :, :]  # N * C * H * W
            mask = mask.expand_as(img)
            len_fill = len(fill) if isinstance(fill, (tuple, list)) else 1
            fill_img = torch.tensor(fill, dtype=img.dtype, device=img.device).view(1, len_fill, 1, 1).expand_as(img)
            if mode == 'nearest':
                mask = mask < 0.5
                img[mask] = fill_img[mask]
            else:  # 'bilinear'
                img = img * mask + (1.0 - mask) * fill_img

        img = self._cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
        return img

    def _cast_squeeze_in(self, img, req_dtypes):
        need_squeeze = False

        # Make image NxCxHxW
        if img.ndim < 4:
            img = img.unsqueeze(dim=0)
            need_squeeze = True

        out_dtype = img.dtype
        need_cast = False
        if out_dtype not in req_dtypes:
            need_cast = True
            req_dtype = req_dtypes[0]
            img = img.to(req_dtype)
        return img, need_cast, need_squeeze, out_dtype

    def _cast_squeeze_out(self, img, need_cast, need_squeeze, out_dtype):
        if need_squeeze:
            img = img.squeeze(dim=0)

        if need_cast:
            if out_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                # It is better to round before cast
                img = torch.round(img)
            img = img.to(out_dtype)

        return img


class GrayScale(nn.Module):
    """
    GrayScale Transform, a typical JointTransform for both the template and search patch
    This augmentation is performed on both supervised and unsupervised pairs,
         as the original TransT baseline also adopts it in the dataloader.
    """

    def __init__(self, gray_probability=0.05):
        """
        args:
            gray_probability - The probability of performing gray scale transform
        """

        super().__init__()
        self.gray_prob = gray_probability

    def forward(self, image_groups, bbox_groups, avg_groups, masks):

        # Groups of batched instances, with search patches and template patches
        batch = image_groups[0].shape[0]
        device = image_groups[0].device

        color_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device).float()
        gray_tag = torch.rand([batch], device=device) <= self.gray_prob

        for i in range(batch):
            # TODO: find some non-loop methods to do such gray scale transformation
            # Note: Gray scale is a joint transform for both template and search patches
            if gray_tag[i]:
                for g in range(len(avg_groups)):
                    # Gray_scale transform to template patches
                    image_groups[g][i].mul_(color_weights[:, None, None])
                    cvt_res_t = avg_groups[g][i][0] * color_weights[0] + \
                                avg_groups[g][i][1] * color_weights[1] + avg_groups[g][i][2] * color_weights[2]
                    avg_groups[g][i] = torch.tensor([cvt_res_t, cvt_res_t, cvt_res_t], device=device)

        return image_groups, bbox_groups, avg_groups, masks
