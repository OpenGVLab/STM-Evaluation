"""
data transform modules for invariance analysis
"""

import torch
from torchvision import transforms
from torchvision.transforms.functional import rotate
import numpy as np
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image


def standard_transform(img_size=224, crop_ratio=0.875):
    size = int(img_size / crop_ratio)
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])


def position_jitter_transform(img_size=224, crop_ratio=0.875, jitter_strength=0):
    return PositionJitterTransform(input_size=img_size,
                                   resizing_size=int(img_size / crop_ratio),
                                   jitter_strength=jitter_strength)


def rotate_transform(img_size=224, crop_ratio=0.875, angle=0, pre_rotate=False):
    return RotateTransform(input_size=img_size,
                           resizing_size=int(img_size / crop_ratio),
                           angle=angle,
                           pre_rotate=pre_rotate)


class PositionJitterTransform:
    def __init__(self, jitter_strength=0, resizing_size=256, input_size=224):
        self.resizing_size = resizing_size
        self.input_size = input_size
        self.jitter_strength = jitter_strength
        self.resize = transforms.Resize(self.resizing_size,
                                        interpolation=transforms.InterpolationMode.BICUBIC)
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    def __call__(self, image):
        resized_image = self.resize(image)
        resized_image = self.to_tensor(resized_image)
        resized_image = self.positon_jitter_crop(resized_image)

        resized_image = self.norm(resized_image)

        return resized_image

    def positon_jitter_crop(self, image):
        mode = np.random.randint(7)
        center = [image.shape[-2] // 2, image.shape[-1] // 2]
        h, w = (self.input_size // 2, self.input_size // 2)

        # randomly choose one direction from 8 directions.
        if mode == 0:
            center[0] = center[0] + self.jitter_strength
        elif mode == 1:
            center[1] = center[1] + self.jitter_strength
        elif mode == 2:
            center[0] = center[0] - self.jitter_strength
        elif mode == 3:
            center[1] = center[1] - self.jitter_strength
        elif mode == 4:
            center[0] = center[0] + self.jitter_strength
            center[1] = center[1] + self.jitter_strength
        elif mode == 5:
            center[0] = center[0] - self.jitter_strength
            center[1] = center[1] + self.jitter_strength
        elif mode == 6:
            center[0] = center[0] - self.jitter_strength
            center[1] = center[1] - self.jitter_strength
        elif mode == 7:
            center[0] = center[0] + self.jitter_strength
            center[1] = center[1] - self.jitter_strength

        top = min(center[0]+h, image.shape[-2])
        bottom = max(center[0]-h, 0)
        right = min(center[1]+w, image.shape[-1])
        left = max(center[1]-w, 0)

        cropped_image = image[:, bottom:top, left:right]

        image_height, image_width = cropped_image.shape[-2:]
        crop_height, crop_width = self.input_size, self.input_size
        if cropped_image.shape[1] < self.input_size or cropped_image.shape[0] < self.input_size:
            padding_lrtb = [
                (crop_width - image_width) // 2 if crop_width > image_width else 0,
                (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
                (crop_height - image_height) // 2 if crop_height > image_height else 0,
                (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
            ]
            cropped_image = torch.nn.functional.pad(cropped_image, padding_lrtb)

        return cropped_image


class RotateTransform:
    def __init__(self, angle=0, resizing_size=256, input_size=224, pre_rotate=False):
        self.angle = angle
        self.resizing_size = resizing_size
        self.input_size = input_size
        self.pre_rotate = pre_rotate

        self.resize = transforms.Resize(self.resizing_size,
                                        interpolation=transforms.InterpolationMode.BICUBIC)
        self.crop = transforms.CenterCrop(input_size)
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    def __call__(self, image):
        resized_image = self.resize(image)
        if self.pre_rotate:
            resized_image = self.rotate(resized_image)
        resized_image = self.crop(resized_image)
        if not self.pre_rotate:
            resized_image = self.rotate(resized_image)
        resized_image = self.to_tensor(resized_image)
        resized_image = self.norm(resized_image)

        return resized_image

    def rotate(self, image):
        return rotate(image,
                      angle=self.angle,
                      interpolation=transforms.InterpolationMode.BICUBIC)
