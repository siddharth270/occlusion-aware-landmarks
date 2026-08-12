"""Image augmentation pipelines.

torchvision rather than albumentations, deliberately: masking is applied to the
numpy image *before* augmentation (it is a data-cleaning step on the source
photo, not a geometric transform), so no mask-aware augmentation is needed, and
torchvision's API is stable across the versions Kaggle ships.

Both arms use identical transforms. The masking flag is the only thing that
differs between them -- anything else would confound the comparison.
"""
from __future__ import annotations

import torchvision.transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int = 224, train: bool = True) -> T.Compose:
    """Training or evaluation pipeline for a PIL RGB image.

    The cache stores images at 256px short side, so `Resize(256)` is a no-op for
    almost every image and only upscales the handful that were already smaller.
    """
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


def denormalize(tensor):
    """Undo normalisation for visualising what the model actually sees."""
    import torch

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)
