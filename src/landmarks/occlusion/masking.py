from __future__ import annotations

import cv2
import numpy as np

STRATEGIES = ("black", "mean_fill", "blur", "inpaint_telea")


def apply_mask(
    image: np.ndarray,
    mask: np.ndarray,
    strategy: str = "mean_fill",
    max_mask_fraction: float = 0.85,
    blur_kernel: int = 31,
) -> np.ndarray:
    """Remove masked regions from `image`.

    Args:
        image: HxWx3 uint8 (BGR or RGB -- strategy is channel-agnostic).
        mask: HxW uint8, non-zero where content should be removed.
        max_mask_fraction: if the mask covers more than this, return the image
            unchanged. Erasing 95% of a photo leaves nothing to learn from and
            would silently poison training with near-blank samples.
    """
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
        # Mean of the KEPT pixels only -- including masked pixels would bias
        # the fill colour toward the occluder we are trying to remove.
        fill = image[~binary].reshape(-1, image.shape[2]).mean(axis=0)
        out[binary] = fill.astype(image.dtype)

    elif strategy == "blur":
        k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        blurred = cv2.GaussianBlur(image, (k, k), 0)
        out[binary] = blurred[binary]

    elif strategy == "inpaint_telea":
        out = cv2.inpaint(image, (binary * 255).astype(np.uint8), 3, cv2.INPAINT_TELEA)

    return out


def mask_fraction(mask: np.ndarray) -> float:
    return float((mask > 0).mean()) if mask.size else 0.0
