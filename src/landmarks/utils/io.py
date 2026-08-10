
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def gld_image_path(image_id: str, root: str | Path, split: str = "train") -> Path:

    return Path(root) / split / image_id[0] / image_id[1] / image_id[2] / f"{image_id}.jpg"


def cached_image_path(image_id: str, cache_root: str | Path) -> Path:
    
    return Path(cache_root) / image_id[0] / image_id[1] / f"{image_id}.jpg"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def write_table(df: pd.DataFrame, path: str | Path) -> None:

    path = Path(path)
    ensure_dir(path.parent)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
