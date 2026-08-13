# Siddharth Mehta, CS5330 PRCV, Final Project
# The image transforms used for training and evaluation. Both arms use the same
# ones so that the only difference between them stays the masking.

from __future__ import annotations

import torchvision.transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# Returns the training or evaluation transform pipeline. Every arm
# uses the same one.
def build_transforms(image_size: int = 224, train: bool = True) -> T.Compose:
    if train:
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.75, 1.333)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    return T.Compose([
        T.Resize(int(image_size * 256 / 224)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# Undoes normalisation so a tensor can be viewed as an image.
def denormalize(tensor):
    import torch

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)
