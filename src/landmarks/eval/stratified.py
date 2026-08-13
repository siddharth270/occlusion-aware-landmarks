# Siddharth Mehta, CS5330 PRCV, Final Project
# Breaks the results down by how occluded each image is. This is where the project
# expects to see a difference, since most images have no occlusion at all.

from __future__ import annotations

import numpy as np
import pandas as pd

from landmarks.eval.gap import global_average_precision
from landmarks.occlusion.metrics import assign_bin


# Labels each row with the occlusion stratum it belongs to.
def add_occlusion_bin(
    frame: pd.DataFrame,
    bins: tuple[float, ...],
    labels: tuple[str, ...],
) -> pd.DataFrame:
    out = frame.copy()
    out["occlusion_bin"] = out.occlusion_ratio.apply(lambda r: assign_bin(r, bins, labels))
    return out


# GAP, top-1 and support for every occlusion stratum and for the
# set as a whole.
def stratified_gap(
    frame: pd.DataFrame,
    preds: np.ndarray,
    confs: np.ndarray,
    bins: tuple[float, ...],
    labels: tuple[str, ...],
) -> pd.DataFrame:
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


# Reshapes the per cell results into the training arm by evaluation
# input matrix.
def cross_condition_table(results: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (arm, condition), df in results.items():
        overall = df[df.occlusion_bin == "ALL"].iloc[0]
        rows.append({"train_arm": arm, "eval_condition": condition,
                     "gap": overall.gap, "top1": overall.top1, "n": int(overall.n)})
    return pd.DataFrame(rows).pivot(index="train_arm", columns="eval_condition", values="gap")
