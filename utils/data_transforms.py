import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
import random


class Compose(object):
    """ Composes several co_transforms together.
    For example:
    >>> transforms.Compose([
    >>>     transforms.CenterCrop(10),
    >>>     transforms.ToTensor(),
    >>>  ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class ColorJitter(object):
    def __init__(self, color_adjust_para):
        """brightness [max(0, 1 - brightness), 1 + brightness] or the given [min, max]"""
        """contrast [max(0, 1 - contrast), 1 + contrast] or the given [min, max]"""
        """saturation [max(0, 1 - saturation), 1 + saturation] or the given [min, max]"""
        """hue [-hue, hue] 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5"""
        '''Ajust brightness, contrast, saturation, hue'''
        '''Input: PIL Image, Output: PIL Image'''
        self.brightness, self.contrast, self.saturation, self.hue = color_adjust_para

    def __call__(self, img, mask):
        brightness_factor = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        contrast_factor = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        saturation_factor = np.random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
        hue_factor = np.random.uniform(-self.hue, self.hue)
        img = Image.fromarray(np.uint8(img))
        if self.brightness > 0:
            img = F.adjust_brightness(img, brightness_factor)
        if self.contrast > 0:
            img = F.adjust_contrast(img, contrast_factor)
        if self.saturation > 0:
            img = F.adjust_saturation(img, saturation_factor)
        if self.hue > 0:
            img = F.adjust_hue(img, hue_factor)

        img = np.asarray(img)
        img = img.clip(0, 255)

        return img, mask


class RandomColorChannel(object):
    def __call__(self, img, mask):
        random_order = np.random.permutation(3)
        img = img[:, :, random_order]
        return img, mask


# class RandomGaussianNoise(object):
#     def __init__(self, gaussian_para):
#         self.mu = gaussian_para[0]
#         self.std_var = gaussian_para[1]
#
#     def __call__(self, img, mask):
#         shape = cfg.DATASET.CROP_IMG_SIZE + [3]
#         gaussian_noise = np.random.normal(self.mu, self.std_var, shape)
#         # only apply to blurry images
#         img = (img + gaussian_noise).clip(0, 1)
#
#         return img, mask


class Normalize(object):
    def __call__(self, img, mask):
        img = np.clip(img - np.median(img)+127, 0, 255)/255.
        mask = mask/255.
        return img.astype(np.float32), mask.astype(np.float32)


class RandomCrop(object):

    def __init__(self, crop_size_h, crop_size_w):
        """Set the height and weight before and after cropping"""
        self.crop_size_h = crop_size_h
        self.crop_size_w = crop_size_w

    def __call__(self, img, mask):
        input_size_h, input_size_w, _ = img.shape  # TODO:right?
        x_start = random.randint(0, input_size_w - self.crop_size_w)
        y_start = random.randint(0, input_size_h - self.crop_size_h)
        img = img[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w]
        mask = mask[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w]

        return img, mask


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5 left-right"""

    def __call__(self, img, mask):
        if random.random() < 0.5:
            '''Change the order of 0 and 1, for keeping the net search direction'''
            img = np.copy(np.fliplr(img))
            mask = np.copy(np.fliplr(mask))

        return img, mask


class RandomVerticalFlip(object):
    """Randomly vertically flips the given PIL.Image with a probability of 0.5  up-down"""

    def __call__(self, img, mask):
        if random.random() < 0.5:
            img = np.copy(np.flipud(img))
            mask = np.copy(np.flipud(mask))
        return img, mask


class ToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, img, mask):
        # handle numpy array
        img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).expand(2, -1, -1).float()
        return img_tensor, mask_tensor
