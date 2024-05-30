import torch.nn as nn
import torch
import torchvision.transforms.functional as tvisf


class NecessaryPreprocess(nn.Module):
    """
    NecessaryPreprocess is a group of necessary operations for processing the image patches.
    Conventionally, these operations should be performed in the Dataloader
    However, since the need to construct and augment pairs based on the tracking results of teacher inference,
             performing those operations in Dataloader is no longer feasible.
    Thus, we move them into the network, and operate on multiple GPUs to save time.

    Typically, NecessaryPreprocess includes:
        1. NormJitter: Normalize the image to [0,1] and perform color jitter, necessary in baseline TransT.
        2. Normalize: Normalize the image with mean and std for channels, necessary in baseline TransT.
    """

    def __init__(self, b_j=0.2, mean=None, std=None, inplace=True):
        super().__init__()
        self.norm_jitter = NormJitter(brightness_jitter=b_j)
        self.channel_norm = Normalize(mean=mean, std=std, inplace=inplace)

    def forward(self, data, tag="template"):
        """
        args:
            data - A TensorDict containing information for patches, should at least including:
                '{}_images': Cropped patches
                '{}_avg': Average channel info for filling the crops
                Note that here brace should be filled with a string tag, e.g. template, search
            tag - A string indicating which patch to perform augmentation, e.g. template, search

        returns:
             im_out - The preprocessed images
             avg_out - The average channels of preprocessed images, reserved for future augmentation
        """
        im_jit, avg_jit = self.norm_jitter(data[tag + "_images"], data[tag + "_avg"])
        im_out, avg_out = self.channel_norm(im_jit, avg_jit)
        return im_out, avg_out

class NormJitter(nn.Module):
    """
    Normalize the image to [0,1] and perform color jitter, necessary in baseline TransT.
    """

    def __init__(self, brightness_jitter=0.2):
        """
        args:
            brightness_jitter - The random brightness jitter scale factor
        """
        super().__init__()
        self.brightness_jitter = brightness_jitter

    def forward(self, images, avg):

        batch = images.shape[0]
        device = images.device

        # Random brightness factor
        brightness_factor = torch.zeros([batch], device=device).uniform_(
            max(0, 1 - self.brightness_jitter), 1 + self.brightness_jitter)

        # Normalization
        norm_images = images.float().mul((brightness_factor / 255.0)[:, None, None, None]).clamp(0.0, 1.0)
        norm_avg = avg.mul((brightness_factor / 255.0)[:, None]).clamp(0.0, 1.0)

        return norm_images, norm_avg


class Normalize(nn.Module):
    """
    Normalize the image with mean and std for channels, necessary in baseline TransT.
    """

    def __init__(self, mean, std, inplace=True):
        """
        args:
            mean, std - Channel mean and std
            inplace - Boolean tag for whether to normalize the image inspace
        """
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, image, avg):

        image = tvisf.normalize(image, self.mean, self.std, self.inplace)
        b, c = avg.shape
        avg = tvisf.normalize(avg.view(b, c, 1, 1), self.mean, self.std, self.inplace).view(b, c)
        return image, avg
