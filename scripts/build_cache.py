"""Step 3: one GPU pass producing the cache, masks, detections and occlusion index.

Detection and caching are fused deliberately. Ultralytics hands back the decoded
image on each result (`orig_img`), so resizing from that array costs no extra
disk read -- important when the source is a 98GB mounted dataset.

Work is chunked and each chunk's outputs are flushed to shards, so a killed
session resumes from the last completed chunk instead of restarting.

Usage:
    python scripts/build_cache.py
    python scripts/build_cache.py --limit 2000 --chunk-size 500   # dry run
"""
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the image cache + occlusion index.")
    p.add_argument("--out", type=str, default="/kaggle/working/cache")
    p.add_argument("--chunk-size", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="process only the first N images")
    p.add_argument("--weights", type=str, default=None)
    p.add_argument("--device", type=int, default=0)

    return p.parse_args()


def process_chunk(det, cfg, chunk: pd.DataFrame, out_dir: Path) -> tuple[list[dict], list[dict]]:
    """Detect, cache and mask one chunk. Returns (index_rows, detection_rows)."""
    paths = [gld_image_path(i, cfg.paths.competition) for i in chunk["id"]]
    index_rows: list[dict] = []
    det_rows: list[dict] = []

    # Consume the stream incrementally: holding 80k decoded images would need
    # ~100GB of RAM.
    for r in det.predict(paths, batch_size=ARGS.batch_size, keep_images=True):
        if r.orig_img is None:
            continue

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
        if mask.any():                       # ~75% of images need no mask file
            save_mask(mask, cache_path(r.image_id, out_dir, "masks", "png"))

        index_rows.append({
            "id": r.image_id,
            "occlusion_ratio": round(ratio, 6),
            "n_detections": r.n_transient,
            "cache_h": h,
            "cache_w": w,
            "orig_h": r.orig_h,
            "orig_w": r.orig_w,
        })
        det_rows.extend(detections_to_rows(r))

    return index_rows, det_rows


def main() -> None:
    global ARGS
    ARGS = parse_args()

    cfg = get_config("masked")
    if ARGS.weights:
        cfg.detection.weights = ARGS.weights
    set_seed(cfg.seed, cfg.deterministic)

    out_dir = ensure_dir(ARGS.out)
    shard_dir = ensure_dir(out_dir / "_shards")

    print(describe_taxonomy())
    print(f"detector: {cfg.detection.weights} @ {cfg.detection.imgsz}px, "
          f"conf={cfg.detection.conf}, max_det={100}")

    subset = load_subset(cfg)
    if ARGS.limit:
        subset = subset.head(ARGS.limit)
    print(f"{len(subset):,} images to process")

    chunks = [subset.iloc[i:i + ARGS.chunk_size]
              for i in range(0, len(subset), ARGS.chunk_size)]

    det = TransientDetector(
        weights=cfg.detection.weights,
        imgsz=cfg.detection.imgsz,
        conf=cfg.detection.conf,
        iou=cfg.detection.iou,
        device=ARGS.device,
        max_det=cfg.detection.max_det,
    )


    for ci, chunk in enumerate(tqdm(chunks, desc="chunks")):
        idx_path = shard_dir / f"index_{ci:05d}.csv"
        if idx_path.exists():                 # completed in an earlier session
            continue

        index_rows, det_rows = process_chunk(det, cfg, chunk, out_dir)

        pd.DataFrame(index_rows).to_csv(idx_path, index=False)
        pd.DataFrame(det_rows).to_parquet(
            shard_dir / f"dets_{ci:05d}.parquet", index=False
        )
        free_gpu()

    # ---- merge shards -------------------------------------------------------
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
    print(f"mask files written: {sum(1 for _ in (out_dir / 'masks').rglob('*.png')):,}")


if __name__ == "__main__":
    main()
