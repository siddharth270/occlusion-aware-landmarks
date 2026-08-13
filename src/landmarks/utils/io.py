# Siddharth Mehta, CS5330 PRCV, Final Project
# Small helpers for building file paths and for reading and writing files.

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# Builds the path to a source image, which GLDv2 nests by the first
# three characters of its id.
def gld_image_path(image_id: str, root: str | Path, split: str = "train") -> Path:
    return Path(root) / split / image_id[0] / image_id[1] / image_id[2] / f"{image_id}.jpg"


# Builds the path to a cached image.
def cached_image_path(image_id: str, cache_root: str | Path) -> Path:
    return Path(cache_root) / image_id[0] / image_id[1] / f"{image_id}.jpg"


# Creates a directory if it is not there already.
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# Writes an object to JSON, creating the parent directory first.
def write_json(obj, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# Reads a JSON file.
def read_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


# Writes a table, choosing parquet or CSV from the file extension.
def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
