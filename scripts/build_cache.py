# Siddharth Mehta, CS5330 PRCV, Final Project
# Runs the detector over the selected images once and saves everything the rest
# of the project needs: a resized copy of each image, a mask of the transient
# objects found in it, all the raw detections, and an occlusion score per image.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landmarks.config import get_config
from landmarks.data.cache import cache_path, resize_short_side, save_image, save_mask
from landmarks.data.subset import load_subset
from landmarks.occlusion.detector import TransientDetector, free_gpu
from landmarks.occlusion.metrics import (
    detections_to_rows,
    occlusion_ratio,
    render_union_mask,
)
from landmarks.occlusion.taxonomy import describe_taxonomy
from landmarks.utils.io import ensure_dir, gld_image_path
from landmarks.utils.seed import set_seed


# Reads the command line options for the cache build.
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the image cache + occlusion index.")
    p.add_argument("--out", type=str, default="/kaggle/working/cache")
    p.add_argument("--chunk-size", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="process only the first N images")
    p.add_argument("--weights", type=str, default=None)
    p.add_argument("--device", type=int, default=0)

    return p.parse_args()


# Runs the detector over one chunk of images, then writes the resized
# image, its mask, and a row of stats for each one.
def process_chunk(
    det,
    cfg,
    chunk: pd.DataFrame,
    out_dir: Path,
    batch_size: int,
) -> tuple[list[dict], list[dict]]:
    paths = [gld_image_path(i, cfg.paths.competition) for i in chunk["id"]]
    expected_ids = set(chunk["id"])
    index_rows: list[dict] = []
    det_rows: list[dict] = []

    for r in det.predict(paths, batch_size=batch_size, keep_images=True):
        if r.orig_img is None:
            continue

        if r.image_id not in expected_ids:
            raise ValueError(
                f"detector returned unexpected image_id {r.image_id!r}; "
                "Result.path is not per-image and the cache would be corrupt"
            )

        small = resize_short_side(r.orig_img, cfg.cache.short_side)
        h, w = small.shape[:2]

        mask = render_union_mask(
            r.detections, h, w,
            dilate_px=cfg.masking.dilate_px,
            subject_guard=cfg.masking.subject_guard,
            subject_min_area=cfg.masking.subject_min_area,
        )
        ratio = occlusion_ratio(mask)

        save_image(small, cache_path(r.image_id, out_dir, "images", "jpg"),
                   cfg.cache.jpeg_quality)

        has_mask = bool(mask.any())
        if has_mask:
            save_mask(mask, cache_path(r.image_id, out_dir, "masks", "png"))

        index_rows.append({
            "id": r.image_id,
            "occlusion_ratio": round(ratio, 6),
            "n_detections": r.n_transient,
            "has_mask": has_mask,
            "cache_h": h,
            "cache_w": w,
            "orig_h": r.orig_h,
            "orig_w": r.orig_w,
        })
        det_rows.extend(detections_to_rows(r))

    return index_rows, det_rows


# Builds the whole cache, picking up where an interrupted run stopped,
# then merges the shards and checks the output against the manifest.
def main() -> None:
    args = parse_args()

    cfg = get_config("masked")
    if args.weights:
        cfg.detection.weights = args.weights
    set_seed(cfg.seed, cfg.deterministic)

    out_dir = ensure_dir(args.out)
    shard_dir = ensure_dir(out_dir / "_shards")

    print(describe_taxonomy())
    print(f"detector: {cfg.detection.weights} @ {cfg.detection.imgsz}px, "
          f"conf={cfg.detection.conf}, max_det={cfg.detection.max_det}")

    subset = load_subset(cfg)
    if args.limit:
        subset = subset.head(args.limit)
    print(f"{len(subset):,} images to process")

    chunks = [subset.iloc[i:i + args.chunk_size]
              for i in range(0, len(subset), args.chunk_size)]

    det = TransientDetector(
        weights=cfg.detection.weights,
        imgsz=cfg.detection.imgsz,
        conf=cfg.detection.conf,
        iou=cfg.detection.iou,
        device=args.device,
        max_det=cfg.detection.max_det,
    )

    for ci, chunk in enumerate(tqdm(chunks, desc="chunks")):
        idx_path = shard_dir / f"index_{ci:05d}.csv"
        if idx_path.exists():
            continue

        index_rows, det_rows = process_chunk(det, cfg, chunk, out_dir, args.batch_size)

        pd.DataFrame(index_rows).to_csv(idx_path, index=False)
        pd.DataFrame(det_rows).to_parquet(
            shard_dir / f"dets_{ci:05d}.parquet", index=False
        )
        free_gpu()

    index = pd.concat(
        [pd.read_csv(p, dtype={"id": str}) for p in sorted(shard_dir.glob("index_*.csv"))],
        ignore_index=True,
    )
    dets = pd.concat(
        [pd.read_parquet(p) for p in sorted(shard_dir.glob("dets_*.parquet"))],
        ignore_index=True,
    )

    index.to_csv(out_dir / "occlusion_index.csv", index=False)
    dets.to_parquet(out_dir / "detections.parquet", index=False)
    cfg.save(out_dir / "cache_config.json")

    r = index.occlusion_ratio.values
    print(f"\ncached {len(index):,} images, {len(dets):,} detections")
    print(f"occlusion ratio: mean={r.mean():.4f} median={np.median(r):.4f} "
          f"zero={np.mean(r == 0):.1%} max={r.max():.3f}")

    n_img_files = sum(1 for _ in (out_dir / "images").rglob("*.jpg"))
    n_mask_files = sum(1 for _ in (out_dir / "masks").rglob("*.png"))
    n_expect_mask = int(index.has_mask.sum())

    print(f"index rows={len(index):,} unique ids={index['id'].nunique():,}")
    print(f"image files on disk={n_img_files:,} (expected {len(index):,})")
    print(f"mask files on disk={n_mask_files:,} (expected {n_expect_mask:,})")

    problems = []
    if not index["id"].is_unique:
        problems.append(f"duplicate ids in index: {len(index) - index['id'].nunique():,}")
    if n_img_files != len(index):
        problems.append(f"image file count {n_img_files:,} != index rows {len(index):,}")
    if n_mask_files != n_expect_mask:
        problems.append(f"mask file count {n_mask_files:,} != expected {n_expect_mask:,}")
    if problems:
        raise RuntimeError("cache integrity check failed:\n  " + "\n  ".join(problems))

    print("integrity checks passed")


if __name__ == "__main__":
    main()
