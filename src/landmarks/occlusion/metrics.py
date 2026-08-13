# Siddharth Mehta, CS5330 PRCV, Final Project
# Turns detections into a mask and into the occlusion score that every result in
# the report is grouped by.

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from landmarks.occlusion.detector import Detection, ImageDetections


# Flags a detection that is large and centred, which usually means it
# is the subject rather than something in the way.
def is_probable_subject(
    box: tuple[float, float, float, float],
    min_area: float = 0.25,
    center_tol: float = 0.30,
) -> bool:
    x1, y1, x2, y2 = box
    if (x2 - x1) * (y2 - y1) < min_area:
        return False
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return abs(cx - 0.5) < center_tol and abs(cy - 0.5) < center_tol


# Draws every kept detection into one binary mask at the size asked for.
def render_union_mask(
    dets: list[Detection],
    height: int,
    width: int,
    tiers: tuple[str, ...] | None = None,
    conf_threshold: float | None = None,
    dilate_px: int = 4,
    use_segmentation: bool = True,
    subject_guard: bool = True,
    subject_min_area: float = 0.25,
) -> np.ndarray:
    from landmarks.occlusion.taxonomy import conf_threshold_for, transient_class_ids

    keep = transient_class_ids(tiers) if tiers is not None else transient_class_ids()
    mask = np.zeros((height, width), dtype=np.uint8)

    for d in dets:
        if d.class_id not in keep:
            continue
        thr = conf_threshold if conf_threshold is not None else conf_threshold_for(d.class_name)
        if d.conf < thr:
            continue
        if subject_guard and is_probable_subject(d.box, min_area=subject_min_area):
            continue

        if use_segmentation and d.polygon is not None:
            pts = d.polygon.copy()
            pts[:, 0] *= width
            pts[:, 1] *= height
            cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
        else:
            x1, y1, x2, y2 = d.box
            cv2.rectangle(
                mask,
                (int(x1 * width), int(y1 * height)),
                (int(x2 * width), int(y2 * height)),
                255,
                thickness=-1,
            )

    if dilate_px > 0 and mask.any():
        k = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel)

    return mask


# Fraction of the image covered by transient content.
def occlusion_ratio(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float((mask > 0).sum()) / float(mask.size)


# Maps a continuous occlusion ratio to its reporting stratum.
def assign_bin(ratio: float, bins: tuple[float, ...], labels: tuple[str, ...]) -> str:
    if len(labels) != len(bins) - 1:
        raise ValueError(f"need {len(bins) - 1} labels for {len(bins)} bin edges")
    idx = int(np.digitize(ratio, bins[1:-1], right=False))
    return labels[min(idx, len(labels) - 1)]


# Flattens one image's detections into rows ready for storage.
def detections_to_rows(img_dets: ImageDetections) -> list[dict]:
    import json

    rows = []
    for d in img_dets.detections:
        rows.append(
            {
                "id": img_dets.image_id,
                "class_id": d.class_id,
                "class_name": d.class_name,
                "conf": round(d.conf, 4),
                "x1": round(d.box[0], 5),
                "y1": round(d.box[1], 5),
                "x2": round(d.box[2], 5),
                "y2": round(d.box[3], 5),
                "box_area": round((d.box[2] - d.box[0]) * (d.box[3] - d.box[1]), 6),
                "polygon": (
                    json.dumps(np.round(d.polygon, 4).tolist())
                    if d.polygon is not None else None
                ),
            }
        )
    return rows


# Counts how many images fall into each occlusion stratum.
def summarise_occlusion(
    index: pd.DataFrame,
    bins: tuple[float, ...],
    labels: tuple[str, ...],
) -> pd.DataFrame:
    index = index.copy()
    index["bin"] = index.occlusion_ratio.apply(lambda r: assign_bin(r, bins, labels))
    summary = (
        index.groupby("bin").agg(n=("id", "size"), mean_ratio=("occlusion_ratio", "mean"))
    )
    summary["pct"] = (100 * summary.n / len(index)).round(2)
    return summary.reindex(list(labels)).fillna(0)
