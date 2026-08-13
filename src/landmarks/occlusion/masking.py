# Siddharth Mehta, CS5330 PRCV, Final Project
# The different ways of filling in a masked region. Plain black is avoided by
# default, because the shape of a black patch is itself something a network can
# learn to recognise.

from __future__ import annotations

import cv2
import numpy as np

STRATEGIES = ("black", "mean_fill", "blur", "inpaint_telea")


# Removes the masked region using the chosen fill, and leaves the
# image untouched if the mask covers too much of it.
def apply_mask(
    image: np.ndarray,
    mask: np.ndarray,
    strategy: str = "mean_fill",
    max_mask_fraction: float = 0.85,
    blur_kernel: int = 31,
) -> np.ndarray:
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")

    binary = mask > 0
    if not binary.any():
        return image
    if binary.mean() > max_mask_fraction:
        return image

    out = image.copy()

    if strategy == "black":
        out[binary] = 0

    elif strategy == "mean_fill":
        fill = image[~binary].reshape(-1, image.shape[2]).mean(axis=0)
        out[binary] = fill.astype(image.dtype)

    elif strategy == "blur":
        k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        blurred = cv2.GaussianBlur(image, (k, k), 0)
        out[binary] = blurred[binary]

    elif strategy == "inpaint_telea":
        out = cv2.inpaint(image, (binary * 255).astype(np.uint8), 3, cv2.INPAINT_TELEA)

    return out


# Fraction of the image that the mask covers.
def mask_fraction(mask: np.ndarray) -> float:
    return float((mask > 0).mean()) if mask.size else 0.0
