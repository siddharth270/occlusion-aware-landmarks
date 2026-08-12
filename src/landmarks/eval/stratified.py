"""GAP broken down by occlusion level.

This is the analysis the project exists for. An overall GAP difference between
arms is easy to dismiss as noise or as an artefact of the fill colour; a
*monotonic* divergence across occlusion strata is much harder to explain any
other way.

GAP within a stratum is computed over that stratum alone (M = stratum size), so
the numbers are comparable across strata but are not a decomposition of overall
GAP -- ranking is global in the full metric and local here. That is the standard
way stratified GAP is reported, and it is stated in the report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from landmarks.eval.gap import global_average_precision
from landmarks.occlusion.metrics import assign_bin


def add_occlusion_bin(
    frame: pd.DataFrame,
    bins: tuple[float, ...],
    labels: tuple[str, ...],
) -> pd.DataFrame:
    out = frame.copy()
    out["occlusion_bin"] = out.occlusion_ratio.apply(lambda r: assign_bin(r, bins, labels))
    return out


def stratified_gap(
    frame: pd.DataFrame,
    preds: np.ndarray,
    confs: np.ndarray,
    bins: tuple[float, ...],
    labels: tuple[str, ...],
) -> pd.DataFrame:
    """Per-stratum GAP, top-1 and support.

    Args:
        frame: evaluation rows in the SAME ORDER as preds/confs (guaranteed by
            using shuffle=False in the eval DataLoader).
        preds, confs: predicted class and its confidence per image.
    """
    if not (len(frame) == len(preds) == len(confs)):
        raise ValueError("frame, preds and confs must be aligned and equal length")

    df = add_occlusion_bin(frame, bins, labels)
    df = df.assign(_pred=preds, _conf=confs)

    rows = []
    for label in labels:
        g = df[df.occlusion_bin == label]
        if len(g) == 0:
            rows.append({"occlusion_bin": label, "n": 0, "gap": np.nan,
                         "top1": np.nan, "mean_occlusion": np.nan})
            continue
        rows.append({
            "occlusion_bin": label,
            "n": len(g),
            "gap": global_average_precision(g.label.values, g._pred.values, g._conf.values),
            "top1": float((g._pred.values == g.label.values).mean()),
            "mean_occlusion": float(g.occlusion_ratio.mean()),
        })

    overall = {
        "occlusion_bin": "ALL",
        "n": len(df),
        "gap": global_average_precision(df.label.values, df._pred.values, df._conf.values),
        "top1": float((df._pred.values == df.label.values).mean()),
        "mean_occlusion": float(df.occlusion_ratio.mean()),
    }
    return pd.DataFrame(rows + [overall])


def cross_condition_table(results: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """Pivot {(arm, eval_condition): stratified_df} into the 2x2 GAP matrix.

    The off-diagonal cells are the informative ones: they separate "masking
    helps the model learn better features" from "masking merely makes train and
    test look alike".
    """
    rows = []
    for (arm, condition), df in results.items():
        overall = df[df.occlusion_bin == "ALL"].iloc[0]
        rows.append({"train_arm": arm, "eval_condition": condition,
                     "gap": overall.gap, "top1": overall.top1, "n": int(overall.n)})
    return pd.DataFrame(rows).pivot(index="train_arm", columns="eval_condition", values="gap")
