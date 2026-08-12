"""Dataset reading the cached images, with masking as a switchable hook.

`apply_masking` is a constructor argument rather than something read from the
config's arm. That is what makes the 2x2 cross-condition evaluation possible: a
model trained on masked data can be evaluated on raw validation images, and vice
versa, using the same class with a different flag.

Masking is applied to the decoded image BEFORE augmentation, because it models a
data-cleaning step on the source photo rather than a geometric transform.
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from landmarks.data.cache import load_image, load_mask
from landmarks.occlusion.masking import apply_mask


class LandmarkDataset(Dataset):
    """Cached landmark images with optional transient-region masking.

    Args:
        frame: rows from the subset manifest, already merged with the occlusion
            index (needs columns: id, label, has_mask).
        cache_root: directory containing images/ and masks/.
        transform: torchvision transform applied to a PIL RGB image.
        apply_masking: whether to remove transient regions at all.
        strategy: fill strategy, see landmarks.occlusion.masking.STRATEGIES.
        apply_prob: probability of masking a given sample. 1.0 is deterministic
            cleaning; values below 1.0 turn masking into a stochastic
            augmentation (the `maskaug` arm).
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        cache_root: str | Path,
        transform,
        apply_masking: bool = False,
        strategy: str = "mean_fill",
        apply_prob: float = 1.0,
        max_mask_fraction: float = 0.85,
    ) -> None:
        required = {"id", "label", "has_mask"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"frame is missing columns {sorted(missing)}")

        self.cache_root = Path(cache_root)
        self.transform = transform
        self.apply_masking = apply_masking
        self.strategy = strategy
        self.apply_prob = apply_prob
        self.max_mask_fraction = max_mask_fraction

        # Materialise as plain lists: DataFrame row access in __getitem__ is
        # slow enough to starve the GPU at this batch size.
        self.ids: list[str] = frame["id"].astype(str).tolist()
        self.labels: list[int] = frame["label"].astype(int).tolist()
        self.has_mask: list[bool] = frame["has_mask"].astype(bool).tolist()

    def __len__(self) -> int:
        return len(self.ids)

    def _should_mask(self, index: int) -> bool:
        if not self.apply_masking or not self.has_mask[index]:
            return False
        if self.apply_prob >= 1.0:
            return True
        # Seeded per worker via landmarks.utils.seed.worker_init_fn.
        return random.random() < self.apply_prob

    def __getitem__(self, index: int):
        image_id = self.ids[index]
        img = load_image(image_id, self.cache_root)          # BGR uint8
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self._should_mask(index):
            mask = load_mask(image_id, self.cache_root, img.shape[:2])
            img = apply_mask(img, mask, self.strategy, self.max_mask_fraction)

        return self.transform(Image.fromarray(img)), self.labels[index]


def build_frame(
    subset: pd.DataFrame,
    occlusion_index: pd.DataFrame,
    split: str | None = None,
) -> pd.DataFrame:
    """Join the split manifest to the occlusion index.

    Inner join on purpose: an image present in the manifest but absent from the
    cache is a build error, and the assertion below surfaces it rather than
    letting training run on a silently truncated set.
    """
    cols = ["id", "occlusion_ratio", "has_mask", "cache_h", "cache_w"]
    frame = subset.merge(occlusion_index[cols], on="id", how="inner")

    if len(frame) != len(subset):
        raise ValueError(
            f"{len(subset) - len(frame):,} manifest rows have no cache entry; "
            "the cache and the subset manifest are out of sync"
        )

    if split is not None:
        frame = frame[frame.split == split].reset_index(drop=True)
    return frame


def load_occlusion_index(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"id": str})
