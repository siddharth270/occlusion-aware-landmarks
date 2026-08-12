"""Per-stratum bootstrap CIs and paired deltas from the saved prediction files.

evaluate.py reports stratified GAP as point estimates only. With 123 test images
in the `high` stratum, a bare point estimate is not interpretable -- this adds
the interval, and the paired within-stratum delta against the baseline.

Runs on CPU from preds_*.parquet alone; no GPU, no model, no cache needed.

    python scripts/stratified_ci.py --results /kaggle/working/artifacts/results
    python scripts/stratified_ci.py --results ~/Downloads/results --iters 2000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landmarks.eval.gap import global_average_precision
from landmarks.eval.stats import paired_bootstrap_delta

BINS = (0.0, 0.02, 0.10, 0.25, 1.01)
LABELS = ("none", "low", "medium", "high")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-stratum GAP CIs from saved predictions.")
    p.add_argument("--results", type=str, default="/kaggle/working/artifacts/results")
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--baseline", type=str, default="baseline_raw",
                   help="reference cell, as <arm>_<condition>")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def bin_of(ratio: np.ndarray) -> pd.Categorical:
    return pd.cut(ratio, bins=list(BINS), right=False,
                  labels=list(LABELS), include_lowest=True)


def gap_ci(labels, preds, confs, iters, seed=42):
    point = global_average_precision(labels, preds, confs)
    rng = np.random.default_rng(seed)
    n = len(labels)
    boot = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, n, n)
        boot[i] = global_average_precision(labels[idx], preds[idx], confs[idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def main() -> None:
    args = parse_args()
    root = Path(args.results)

    cells: dict[str, pd.DataFrame] = {}
    for f in sorted(root.glob("preds_*.parquet")):
        key = f.stem.replace("preds_", "")
        df = pd.read_parquet(f)
        df["bin"] = bin_of(df.occlusion_ratio.values)
        cells[key] = df
    if not cells:
        raise FileNotFoundError(f"no preds_*.parquet under {root}")
    print(f"loaded {len(cells)} cells: {', '.join(cells)}\n")

    # ---- per-cell, per-stratum GAP with CI ---------------------------------
    rows = []
    for key, df in cells.items():
        for stratum in LABELS + ("ALL",):
            g = df if stratum == "ALL" else df[df["bin"] == stratum]
            if len(g) == 0:
                continue
            point, lo, hi = gap_ci(g.label.values, g.pred.values,
                                   g.conf.values, args.iters)
            rows.append({"cell": key, "stratum": stratum, "n": len(g),
                         "gap": round(point, 4), "ci_low": round(lo, 4),
                         "ci_high": round(hi, 4),
                         "half_width": round((hi - lo) / 2, 4),
                         "top1": round(float((g.pred.values == g.label.values).mean()), 4)})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    # ---- paired within-stratum deltas vs the reference cell ----------------
    ref = args.baseline
    if ref not in cells:
        raise KeyError(f"reference cell {ref!r} not found; have {list(cells)}")

    print(f"\npaired delta vs {ref}, within stratum "
          f"(same images, same bootstrap indices):\n")
    drows = []
    for key, df in cells.items():
        if key == ref:
            continue
        for stratum in LABELS + ("ALL",):
            m = slice(None) if stratum == "ALL" else (cells[ref]["bin"] == stratum).values
            a, b = cells[ref][m], df[m]
            if len(a) == 0:
                continue
            # Rows are in identical order across cells (evaluate.py used
            # shuffle=False on one frame), so positional pairing is valid.
            res = paired_bootstrap_delta(
                a.label.values, a.pred.values, a.conf.values,
                b.pred.values, b.conf.values, iters=args.iters,
            )
            drows.append({"cell": key, "stratum": stratum, "n": len(a),
                          "delta": round(res["delta"], 4),
                          "ci_low": round(res["ci_low"], 4),
                          "ci_high": round(res["ci_high"], 4),
                          "p": round(res["p_value"], 4),
                          "sig": res["significant"]})
    deltas = pd.DataFrame(drows)
    print(deltas.to_string(index=False))

    out = Path(args.out) if args.out else root
    table.to_csv(out / "stratified_gap_ci.csv", index=False)
    deltas.to_csv(out / "stratified_delta_ci.csv", index=False)
    print(f"\nwrote stratified_gap_ci.csv and stratified_delta_ci.csv to {out}")


if __name__ == "__main__":
    main()
