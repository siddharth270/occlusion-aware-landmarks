"""Pre-resized image cache and binary occlusion masks.

Training decodes 64k images per epoch. Doing that from the original ~800px
JPEGs on a mounted 98GB dataset is I/O-bound and re-attaches a dataset we
otherwise never need. Caching at 256px short side cuts decode cost ~10x and
shrinks the working set to ~1.6GB, which fits comfortably as a Kaggle Dataset.

Masks are stored as separate binary PNGs rather than baked into the images, so
the fill strategy stays a runtime choice and both arms read the same cache.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from landmarks.utils.io import ensure_dir


def cache_path(image_id: str, root: str | Path, kind: str, ext: str) -> Path:
    """Two-level hex fan-out: ~312 files per directory at 80k images."""
    return Path(root) / kind / image_id[0] / image_id[1] / f"{image_id}.{ext}"


def resize_short_side(img: np.ndarray, short_side: int = 256) -> np.ndarray:
    """Downscale preserving aspect ratio. Never upscales -- a few GLDv2 images
    are already smaller than the target and enlarging them only wastes bytes."""
    h, w = img.shape[:2]
    if min(h, w) <= short_side:
        return img
    scale = short_side / min(h, w)
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def save_image(img: np.ndarray, path: str | Path, quality: int = 90) -> None:
    ensure_dir(Path(path).parent)
    # cv2.imwrite returns False instead of raising; an unchecked failure here
    # would silently produce a cache with missing entries.
    if not cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise IOError(f"failed to write image: {path}")


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    """PNG, max compression. Binary masks compress to 1-5 KB."""
    ensure_dir(Path(path).parent)
    if not cv2.imwrite(str(path), mask, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
        raise IOError(f"failed to write mask: {path}")


def load_image(image_id: str, root: str | Path) -> np.ndarray:
    """Cached BGR image. Raises if absent -- a silent black image would poison training."""
    p = cache_path(image_id, root, "images", "jpg")
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cached image missing or unreadable: {p}")
    return img


def load_mask(image_id: str, root: str | Path, shape: tuple[int, int]) -> np.ndarray:
    """Cached mask, or an all-zero mask when none was written (no detections)."""
    p = cache_path(image_id, root, "masks", "png")
    if not p.exists():
        return np.zeros(shape, dtype=np.uint8)
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros(shape, dtype=np.uint8)
    if m.shape != shape:                       # defensive: JPEG rounding drift
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m
