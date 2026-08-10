
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from landmarks.utils.io import ensure_dir, gld_image_path, write_json


def load_train_csv(competition_root: str | Path) -> pd.DataFrame:
    """Read the full competition label file (id, landmark_id)."""
    df = pd.read_csv(Path(competition_root) / "train.csv", dtype={"id": str})
    expected = {"id", "landmark_id"}
    if not expected.issubset(df.columns):
        raise ValueError(f"train.csv missing columns {expected - set(df.columns)}")
    return df


def select_classes(
    train_df: pd.DataFrame,
    n_classes: int,
    min_images_per_class: int,
) -> pd.DataFrame:

    counts = train_df.groupby("landmark_id").size().rename("n_available").reset_index()
    eligible = counts[counts.n_available >= min_images_per_class]

    if len(eligible) < n_classes:
        raise ValueError(
            f"only {len(eligible)} classes have >= {min_images_per_class} images, "
            f"but n_classes={n_classes} was requested"
        )

    # Sort by count desc, then landmark_id asc to break ties deterministically.
    selected = (
        eligible.sort_values(["n_available", "landmark_id"], ascending=[False, True])
        .head(n_classes)
        .reset_index(drop=True)
    )
    selected["label"] = np.arange(len(selected), dtype=np.int64)
    return selected[["landmark_id", "label", "n_available"]]


def sample_images(
    train_df: pd.DataFrame,
    class_map: pd.DataFrame,
    max_images_per_class: int,
    seed: int,
) -> pd.DataFrame:
    """Take up to `max_images_per_class` images from each selected class.

    Shuffling once before the per-class head() keeps the choice random but
    fully reproducible from `seed`.
    """
    df = train_df.merge(class_map[["landmark_id", "label"]], on="landmark_id", how="inner")
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df = df.groupby("landmark_id", sort=False).head(max_images_per_class)
    return df.sort_values(["label", "id"]).reset_index(drop=True)


def verify_images_exist(
    df: pd.DataFrame,
    competition_root: str | Path,
    workers: int = 16,
) -> tuple[pd.DataFrame, int]:
    """Drop rows whose JPEG is missing on disk.

    Cheap insurance: a handful of missing files would otherwise surface as
    crashes deep inside a 90-minute training run.
    """
    paths = [gld_image_path(i, competition_root) for i in df["id"]]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        exists = list(tqdm(ex.map(Path.exists, paths), total=len(paths), desc="verify files"))

    mask = np.asarray(exists, dtype=bool)
    return df[mask].reset_index(drop=True), int((~mask).sum())


def stratified_split(
    df: pd.DataFrame,
    ratios: tuple[float, float, float],
    seed: int,
) -> pd.DataFrame:
    """Split within each class so every class appears in every split.

    A global random split can starve a class of training images entirely.
    Splitting per class guarantees train coverage and keeps the val/test class
    distributions identical to train's.
    """
    train_r, val_r, _ = ratios
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"split_ratios must sum to 1.0, got {ratios}")

    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    splits = np.empty(len(shuffled), dtype=object)

    for _, idx in shuffled.groupby("label", sort=False).indices.items():
        n = len(idx)
        n_train = max(1, int(np.floor(n * train_r)))     # never starve train
        n_val = min(int(np.floor(n * val_r)), n - n_train)  # never overflow

        splits[idx[:n_train]] = "train"
        splits[idx[n_train : n_train + n_val]] = "val"
        splits[idx[n_train + n_val :]] = "test"

    shuffled["split"] = splits
    return shuffled.sort_values(["label", "split", "id"]).reset_index(drop=True)


def validate_manifest(subset: pd.DataFrame, class_map: pd.DataFrame) -> None:
    """Fail loudly now rather than silently corrupting every later result."""
    assert subset["id"].is_unique, "duplicate image ids in subset"
    assert subset["split"].isin(["train", "val", "test"]).all(), "bad split value"

    n_classes = len(class_map)
    assert subset["label"].nunique() == n_classes, "some classes have no images"

    train_labels = set(subset.loc[subset.split == "train", "label"])
    assert len(train_labels) == n_classes, "some classes are absent from train"

    assert subset["label"].min() == 0 and subset["label"].max() == n_classes - 1, (
        "labels are not contiguous"
    )


def build_subset(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end: full train.csv -> written, validated manifests."""
    sub = cfg.subset

    train_df = load_train_csv(cfg.paths.competition)
    print(f"full train.csv: {len(train_df):,} images / {train_df.landmark_id.nunique():,} classes")

    class_map = select_classes(train_df, sub.n_classes, sub.min_images_per_class)
    print(
        f"selected {len(class_map):,} classes  "
        f"(available images/class: min={class_map.n_available.min():,} "
        f"max={class_map.n_available.max():,})"
    )

    subset = sample_images(train_df, class_map, sub.max_images_per_class, cfg.seed)
    print(f"sampled {len(subset):,} images (cap {sub.max_images_per_class}/class)")

    if sub.verify_files_exist:
        subset, n_missing = verify_images_exist(subset, cfg.paths.competition)
        print(f"missing files dropped: {n_missing:,} -> {len(subset):,} remain")

    subset = stratified_split(subset, sub.split_ratios, cfg.seed)

    # Recount after sampling/verification so class_map reflects what we actually have.
    n_sampled = subset.groupby("label").size().rename("n_sampled")
    class_map = class_map.merge(n_sampled, on="label", how="left")
    class_map["n_sampled"] = class_map["n_sampled"].fillna(0).astype(int)

    validate_manifest(subset, class_map)

    out_dir = ensure_dir(cfg.paths.manifests_out)
    subset[["id", "landmark_id", "label", "split"]].to_csv(
        out_dir / "subset_splits.csv", index=False
    )
    class_map.to_csv(out_dir / "class_map.csv", index=False)

    counts = subset["split"].value_counts().to_dict()
    write_json(
        {
            "seed": cfg.seed,
            "n_classes": int(len(class_map)),
            "n_images": int(len(subset)),
            "split_counts": {k: int(v) for k, v in counts.items()},
            "min_images_per_class": sub.min_images_per_class,
            "max_images_per_class": sub.max_images_per_class,
            "split_ratios": list(sub.split_ratios),
        },
        out_dir / "subset_summary.json",
    )

    print(f"\nsplit counts: {counts}")
    print(f"written to {out_dir}")
    return subset, class_map


def load_subset(cfg, split: str | None = None) -> pd.DataFrame:
    """Read the frozen manifest. Every downstream step uses this, never build_subset."""
    path = Path(cfg.paths.manifests) / "subset_splits.csv"
    if not path.exists():                      # fall back to the writable copy
        path = Path(cfg.paths.manifests_out) / "subset_splits.csv"
    df = pd.read_csv(path, dtype={"id": str})
    return df[df.split == split].reset_index(drop=True) if split else df


def load_class_map(cfg) -> pd.DataFrame:
    path = Path(cfg.paths.manifests) / "class_map.csv"
    if not path.exists():
        path = Path(cfg.paths.manifests_out) / "class_map.csv"
    return pd.read_csv(path)
