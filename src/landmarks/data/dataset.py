# Siddharth Mehta, CS5330 PRCV, Final Project
# The PyTorch dataset. Masking is a switch passed in when the dataset is built, so
# the same class can serve raw or masked images to any of the trained models.

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

    # Sets up the dataset and copies the columns into lists, since
    # DataFrame lookups are too slow to do once per item.
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

        self.ids: list[str] = frame["id"].astype(str).tolist()
        self.labels: list[int] = frame["label"].astype(int).tolist()
        self.has_mask: list[bool] = frame["has_mask"].astype(bool).tolist()

    # Number of images in this split.
    def __len__(self) -> int:
        return len(self.ids)

    # Decides whether this sample gets masked, which is random for
    # the maskaug arm.
    def _should_mask(self, index: int) -> bool:
        if not self.apply_masking or not self.has_mask[index]:
            return False
        if self.apply_prob >= 1.0:
            return True
        return random.random() < self.apply_prob

    # Loads one image, masks it if required, then applies the transforms.
    def __getitem__(self, index: int):
        image_id = self.ids[index]
        img = load_image(image_id, self.cache_root)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self._should_mask(index):
            mask = load_mask(image_id, self.cache_root, img.shape[:2])
            img = apply_mask(img, mask, self.strategy, self.max_mask_fraction)

        return self.transform(Image.fromarray(img)), self.labels[index]


# Joins the split manifest to the occlusion index and fails if the
# two disagree about which images exist.
def build_frame(
    subset: pd.DataFrame,
    occlusion_index: pd.DataFrame,
    split: str | None = None,
) -> pd.DataFrame:
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


# Reads the occlusion index, keeping image ids as strings.
def load_occlusion_index(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"id": str})
