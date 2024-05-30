import torch
import torchvision.transforms as transforms
import ltr.data.transforms as tfm
from ltr.utils import TensorDict
import ltr.data.processing_utils as prutils
import os
import cv2


def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseSparseProcessing:
    """
    Base class for Processing for sparsely-supervised tracking.

    Processing class is used to process the data returned by a dataset, before passing it
        through the network. For example, it can be used to crop a search region around the object,
        apply various data augmentations, etc.

    In our implementation, we use complex augmentations such as Ellipse Rotation,
        so some of the cropping and augmentation works are moved to multiple GPUs.
        The transform feature here are reserved only for future features and compatibility with LTR framework.
    """

    def __init__(self, transform=transforms.ToTensor(), search_transform=None, template_transform=None,
                 joint_transform=None, key_transform=None, unlabeled_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if search_transform or
                                template_transform is None.
            search_transform - The set of transformations to be applied on the search images in burn_in stage.
                                 If None, the 'transform' argument is used instead.
            template_transform - The set of transformations to be applied on the template images in burn_in stage.
                                 If None, the 'transform' argument is used instead.
            key_transform - The set of transformations to be applied on the labeled images in sparse-sup stage.
                                Key transform is often weak as inference by teacher should be as easy as possible.
                                If None, the 'transform' argument is used instead.
            unlabeled_transform - The set of transformations to be applied on the unlabeled images in sparse-sup stage.
                                Unlabeled transform is often weak as inference by teacher should be as easy as possible.
                                If None, the 'transform' argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the search and template images.
                              In our implementation, joint transform is moved to multiple GPUs.
        """
        self.transform = {'search': transform if search_transform is None else search_transform,
                          'template': transform if template_transform is None else template_transform,
                          'unlabeled': transform if unlabeled_transform is None else unlabeled_transform,
                          'key': transform if key_transform is None else key_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class SparseProcessingImpl(BaseSparseProcessing):
    """
    The processing class implemented for training in a sparsely-supervised manner
    """

    def __init__(self, search_area_factor, template_area_factor, search_sz,
                 temp_sz, center_jitter_factor, scale_jitter_factor,
                 mode='sequence', burn_in=False, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region relative to the target size.
            template_area_factor - The size of the template region relative to the template target size.

            search_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            temp_sz - An integer, denoting the size to which the template region is resized. The search region is always
                      square.

            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.

            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
            burn_in - Boolean tag indicating whether the sampler works as burn_in stage or sparsely-supervised stage

        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.template_area_factor = template_area_factor
        self.search_sz = search_sz
        self.temp_sz = temp_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode

        # whether in burn_in stage
        self.burn_in = burn_in
        self.to_tensor_tsf = tfm.ToTensor()

    def __call__(self, data: TensorDict):

        if self.burn_in:
            return self._burn_in_preprocessing(data)
        else:
            return self._sparse_preprocessing(data)

    def _sparse_preprocessing(self, data: TensorDict):
        """
        Basic processing logic for unsupervised instances in sparsely-supervised training
        The item key should be in ['key', 'aug', 'unlabeled', 'relay'],
             but the data source of 'aug' patch is 'unlabeled'.
        Here 'key' and 'relay' will be cropped as template, while 'aug' and 'unlabeled' as search patches.

        args:

            data - The input data, should contain the following fields:

                'XXX_images': Raw images loaded from original dataset
                'XXX_anno': Annotation info for raw images

                'XXX_avg': Average channel info for original images
                'XXX_record_keys': Other useful info for those frames, e.g. sequence/frame id

        returns:
            data - output data block with following fields:

                'XXX_images': Cropped images
                'XXX_anno': Annotations of target object. Coarse boxes for unlabeled frames.
                'XXX_avg' : Average channel info for filling the crops

            middle_info - auxiliary information reserved for middleware, a dict containing:

                'XXX_record_keys': Useful info for those frames, e.g. sequence/frame id
                'XXX_box_extract', 'XXX_resize_factors': Info for recovering the box in cropped frames.

        """

        # Note: No joint augmentation for generating pseudo label (teacher net)
        # Here we store the original loaded images and other auxiliary info in a dict middle_info,
        #      they will be used in SparseMiddleWare for further cropping patches from results of teacher models.
        middle_info = dict()
        # Reserve meta info already recorded in sampler (moving them into the dict middle_info)
        for s in ['key', 'unlabeled', 'relay']:
            res_key = s + '_record_keys'
            if res_key in data:
                middle_info[res_key] = data[res_key]
                del data[res_key]

        # Now begin to crop the template/search patches
        for s in ['key', 'aug', 'unlabeled', 'relay']:
            assert self.mode == 'sequence', "Currently, SparseProcessing only supports sequence mode."

            # Crop image region centered at jittered_anno box
            if s == 'key':
                # Add a uniform noise to the center pos
                jittered_anno = [prutils.get_jittered_box(a, "template", self.scale_jitter_factor,
                                                          self.center_jitter_factor) for a in data[s + '_anno']]
                # Crop template patches of labeled key frames with jittered center
                crops, boxes, _ = \
                    prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                 self.template_area_factor, self.temp_sz, avg=data[s + '_avg'],
                                                 input_type="numpy", recoverable=False, fill_avg=True)

            elif s == 'relay':
                # The first unlabeled frame should be used to build a relay template
                jittered_anno = [prutils.get_jittered_box(a, "template", self.scale_jitter_factor,
                                                          self.center_jitter_factor) for a in data[s + '_anno']]
                # Crop template patches of labeled key frames with jittered center
                crops, boxes, _ = \
                    prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                 self.template_area_factor, self.temp_sz, avg=data[s + '_avg'],
                                                 input_type="numpy", recoverable=False, fill_avg=True)

            elif s == 'unlabeled':
                # Add a uniform noise to the center pos
                jittered_anno = [prutils.get_jittered_box(a, s, self.scale_jitter_factor,
                                                          self.center_jitter_factor) for a in data[s + '_anno']]
                # Always crop search patches of unlabeled key frames with jittered center
                # Here we additionally collect two middle_info including 'box_extract' and 'resize_factors'.
                # Teacher inference only gives tracking results on cropped patches,
                #     so we need to reserve these info and recover the tracking results to boxes on raw images.
                crops, boxes, recover_info = \
                    prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                 self.search_area_factor, self.search_sz, avg=data[s + '_avg'],
                                                 input_type="numpy", recoverable=True, fill_avg=True)
                middle_info[s + '_box_extract'] = torch.stack(recover_info['box_extract'])
                middle_info[s + '_resize_factors'] = torch.stack(recover_info['resize_factors'])

            elif s == 'aug':
                # Add a uniform noise to the center pos
                s_source = 'unlabeled'
                jittered_anno = [prutils.get_jittered_box(a, s, self.scale_jitter_factor,
                                                          self.center_jitter_factor) for a in data[s_source + '_anno']]
                # Always crop search patches of unlabeled key frames with jittered center
                crops, boxes, recover_info = \
                    prutils.jittered_center_crop(data[s_source + '_images'], jittered_anno, data[s_source + '_anno'],
                                                 self.search_area_factor, self.search_sz, avg=data[s_source + '_avg'],
                                                 input_type="numpy", recoverable=True, fill_avg=True)
                data[s + '_avg'] = data[s_source + '_avg']
                # Here we also collect two middle_info including 'box_extract' and 'resize_factors',
                #    These are used to calculate precise box anno on the augmented pair in SparseMiddleWare
                middle_info[s + '_box_extract'] = torch.stack(recover_info['box_extract'])
                middle_info[s + '_resize_factors'] = torch.stack(recover_info['resize_factors'])

            else:
                raise NotImplementedError

            # Now we apply transforms in multiple GPUs, instead of in dataloader
            data[s + '_images'], data[s + '_anno'] = \
                [self.to_tensor_tsf.transform_image(crop) for crop in crops], boxes

        # Prepare output
        data = data.apply(stack_tensors)
        # Squeeze output
        key_pre, key_suf = ['key', 'unlabeled', 'relay', 'aug'], ['images', 'anno', 'avg']
        for kp in key_pre:
            for ks in key_suf:
                squeeze_key = '{}_{}'.format(kp, ks)
                data[squeeze_key] = data[squeeze_key].squeeze()

        return data, middle_info

    def _burn_in_preprocessing(self, data: TensorDict):
        """
        Basic processing logic for supervised instances in burn-in / sparsely-supervised training
        The item key should be in ['template', 'search']
        Here 'template' will be cropped as template, while 'search' as search patches.

        args:
            data - The input data, should contain the following fields:

                'XXX_images': Raw images loaded from original dataset
                'XXX_anno': Annotation info for raw images

                'XXX_avg': Average channel info for original images
                'XXX_record_keys': Useful info for those frames, e.g. sequence/frame id

        returns:
            data - output data block with following fields:

                'XXX_images: Cropped images
                'XXX_anno': Annotations of target object. Coarse boxes for unlabeled frames.
                'XXX_avg': Average channel info for filling the crops

        """

        # Note: Joint augmentation (e.g. grayscale) here is moved to GPUs
        for s in ['search', 'template']:
            assert self.mode == 'sequence', "Currently, SparseProcessing only supports sequence mode."

            if s == 'template':
                # Having no check_key means that the loaded data from sampler is raw images / annotations
                # Add a uniform noise to the center pos
                jittered_anno = [prutils.get_jittered_box(a, s, self.scale_jitter_factor,
                                                          self.center_jitter_factor) for a in data[s + '_anno']]
                # Crop template patches of labeled key frames with jittered center
                crops, boxes, _ = \
                    prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                 self.template_area_factor, self.temp_sz, avg=data[s + '_avg'],
                                                 input_type="numpy", recoverable=False, fill_avg=True)
                if s + '_record_keys' in data.keys():
                    del data[s + '_record_keys']

            elif s == 'search':
                # Add a uniform noise to the center pos
                jittered_anno = [prutils.get_jittered_box(a, s, self.scale_jitter_factor,
                                                          self.center_jitter_factor) for a in data[s + '_anno']]
                # Always crop search patches of unlabeled key frames with jittered center
                crops, boxes, _ = \
                    prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                 self.search_area_factor, self.search_sz, avg=data[s + '_avg'],
                                                 input_type="numpy", recoverable=False, fill_avg=True)
            else:
                raise NotImplementedError

            # We need to transpose the image patches into [C, H, W] format as a tensor
            crops = [self.to_tensor_tsf.transform_image(img_np) for img_np in crops]
            data[s + '_images'], data[s + '_anno'] = crops, boxes

        # Prepare output
        data = data.apply(stack_tensors)
        # Squeeze output
        key_pre, key_suf = ['template', 'search'], ['images', 'anno', 'avg']
        for kp in key_pre:
            for ks in key_suf:
                squeeze_key = '{}_{}'.format(kp, ks)
                data[squeeze_key] = data[squeeze_key].squeeze()

        return data

