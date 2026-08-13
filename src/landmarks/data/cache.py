# Siddharth Mehta, CS5330 PRCV, Final Project
# Reads and writes the cached images and masks. Caching at a smaller size keeps
# training fast and means the huge original dataset is only needed once.

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from landmarks.utils.io import ensure_dir


# Builds the path for a cached file, spread over two levels so no
# single directory gets too large.
def cache_path(image_id: str, root: str | Path, kind: str, ext: str) -> Path:
    return Path(root) / kind / image_id[0] / image_id[1] / f"{image_id}.{ext}"


# Shrinks an image so its shorter side hits the target. Images that are
# already smaller are left alone.
def resize_short_side(img: np.ndarray, short_side: int = 256) -> np.ndarray:
    h, w = img.shape[:2]
    if min(h, w) <= short_side:
        return img
    scale = short_side / min(h, w)
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


# Writes a cached JPEG and raises on failure, since OpenCV only
# returns False and would fail silently.
def save_image(img: np.ndarray, path: str | Path, quality: int = 90) -> None:
    ensure_dir(Path(path).parent)
    if not cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise IOError(f"failed to write image: {path}")


# Writes a mask as a compressed PNG and raises if the write fails.
def save_mask(mask: np.ndarray, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    if not cv2.imwrite(str(path), mask, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
        raise IOError(f"failed to write mask: {path}")


# Reads a cached image and raises if it is missing, so a blank image
# never reaches training.
def load_image(image_id: str, root: str | Path) -> np.ndarray:
    p = cache_path(image_id, root, "images", "jpg")
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cached image missing or unreadable: {p}")
    return img


# Reads a cached mask, returning an empty one when the image had
# no detections.
def load_mask(image_id: str, root: str | Path, shape: tuple[int, int]) -> np.ndarray:
    p = cache_path(image_id, root, "masks", "png")
    if not p.exists():
        return np.zeros(shape, dtype=np.uint8)
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros(shape, dtype=np.uint8)
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m
