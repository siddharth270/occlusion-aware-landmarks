# Siddharth Mehta, CS5330 PRCV, Final Project
# Draws the main results figure, GAP against occlusion level, with matched
# conditions on the left panel and mismatched ones on the right.

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

STRATA = ["none", "low", "medium", "high"]
RANGES = {"none": "<0.02", "low": "0.02–0.10", "medium": "0.10–0.25", "high": "≥0.25"}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e3e2df"
COLORS = {"baseline": "#2a78d6", "masked": "#eb6834", "maskaug": "#1baf7a"}

MATCHED = [("baseline_raw", "baseline", "baseline / raw"),
           ("masked_masked", "masked", "masked / masked"),
           ("maskaug_masked", "maskaug", "maskaug / masked")]
MISMATCHED = [("baseline_masked", "baseline", "baseline / masked"),
              ("masked_raw", "masked", "masked / raw"),
              ("maskaug_raw", "maskaug", "maskaug / raw")]


# Draws one panel of the figure, a line per cell with its confidence band.
def panel(ax, table, cells, title, dashed: bool) -> None:
    x = range(len(STRATA))
    for cell, arm, label in cells:
        g = table[table.cell == cell].set_index("stratum").reindex(STRATA)
        ax.fill_between(x, g.ci_low, g.ci_high, color=COLORS[arm], alpha=0.13, linewidth=0)
        ax.plot(x, g.gap, color=COLORS[arm], linewidth=2,
                linestyle="--" if dashed else "-",
                marker="o", markersize=6, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=label)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{s}\n{RANGES[s]}" for s in STRATA], fontsize=8)
    ax.set_ylim(0.15, 0.92)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=3, color=GRID)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK_2)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=10)


# Builds the stratum figure, matched conditions on the left and
# mismatched on the right.
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, default="~/Downloads/results")
    ap.add_argument("--out", type=str, default="report/figures/gap_vs_stratum.png")
    args = ap.parse_args()

    src = Path(args.results).expanduser() / "stratified_gap_ci.csv"
    table = pd.read_csv(src)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True,
                                   facecolor=SURFACE)
    panel(ax1, table, MATCHED,
          "(a) Matched — each arm on the input it trained on", dashed=False)
    panel(ax2, table, MISMATCHED,
          "(b) Mismatched — each arm on the other input", dashed=True)

    ax1.set_ylabel("GAP (95% bootstrap CI)", fontsize=9, color=INK_2)
    for ax in (ax1, ax2):
        ax.set_xlabel("occlusion stratum", fontsize=9, color=INK_2)

    b = table[(table.cell == "baseline_masked") & (table.stratum == "high")].iloc[0]
    ax2.annotate("baseline: 0.6475 raw → 0.2304 masked\n(Δ −0.4171, p<0.001)",
                 xy=(2.94, b.gap), xytext=(1.35, 0.33),
                 fontsize=8, color=INK_2, ha="left", va="center", linespacing=1.4,
                 arrowprops=dict(arrowstyle="->", color=INK_2, linewidth=0.8,
                                 shrinkA=6, shrinkB=4))

    n = table[table.cell == "baseline_raw"].set_index("stratum").reindex(STRATA).n
    fig.text(0.5, -0.03,
             "test images per stratum:  " + "   ".join(
                 f"{s} {int(v):,}" for s, v in zip(STRATA, n)),
             ha="center", fontsize=8, color=INK_2)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
