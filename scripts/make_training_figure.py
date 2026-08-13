# Siddharth Mehta, CS5330 PRCV, Final Project
# Draws validation GAP and training loss per epoch for the three arms, and marks
# the epoch whose checkpoint was kept.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ARMS = ("baseline", "masked", "maskaug")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e3e2df"
COLORS = {"baseline": "#2a78d6", "masked": "#eb6834", "maskaug": "#1baf7a"}
LABEL = {"baseline": "baseline (raw)", "masked": "masked (p=1.0)", "maskaug": "maskaug (p=0.5)"}


# Loads per epoch history from the run directories, falling back to
# the copy kept in manifests.
def load_history(runs: Path | None, fallback: Path) -> pd.DataFrame:
    if runs is not None:
        frames = []
        for arm in ARMS:
            p = runs / arm / "history.json"
            if p.exists():
                d = pd.DataFrame(json.loads(p.read_text()))
                d["arm"] = arm
                frames.append(d)
        if len(frames) == len(ARMS):
            print(f"loaded history.json for {len(frames)} arms from {runs}")
            return pd.concat(frames, ignore_index=True)
        print(f"incomplete history.json under {runs}; using {fallback}")
    return pd.read_csv(fallback)


# Draws validation GAP and training loss per epoch for all three arms.
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=str, default=None)
    ap.add_argument("--history", type=str, default="manifests/training_history.csv")
    ap.add_argument("--out", type=str, default="report/figures/training_curves.png")
    args = ap.parse_args()

    df = load_history(Path(args.runs) if args.runs else None, Path(args.history))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6), facecolor=SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8, length=3, color=GRID)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, linewidth=0.7)
        ax.set_xlabel("epoch", fontsize=9, color=INK_2)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))

    OFFSET = {"baseline": (7, 4), "masked": (7, -12), "maskaug": (7, -13)}
    for arm in ARMS:
        g = df[df.arm == arm].sort_values("epoch")
        ax1.plot(g.epoch, g.val_gap, color=COLORS[arm], linewidth=2, label=LABEL[arm])

        best = g.loc[g.val_gap.idxmax()]
        ax1.plot(best.epoch, best.val_gap, "o", color=COLORS[arm],
                 markersize=7, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax1.annotate(f"{best.val_gap:.4f}",
                     xy=(best.epoch, best.val_gap), xytext=OFFSET[arm],
                     textcoords="offset points", fontsize=8, color=INK_2)

    ax1.set_ylabel("validation GAP", fontsize=9, color=INK_2)
    ax1.set_ylim(0.20, 0.83)
    ax1.set_xlim(right=df.epoch.max() + 2.6)
    ax1.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK_2)
    ax1.set_title("(a) Validation GAP — each arm on its own input condition",
                  fontsize=9.5, color=INK, loc="left", pad=10)

    finals = []
    for arm in ARMS:
        g = df[df.arm == arm].sort_values("epoch")
        ax2.plot(g.epoch, g.train_loss, color=COLORS[arm], linewidth=2)
        finals.append(g.train_loss.iloc[-1])
    ax2.annotate(f"final {min(finals):.2f}–{max(finals):.2f}",
                 xy=(df.epoch.max(), max(finals)), xytext=(8, -2),
                 textcoords="offset points", fontsize=8, color=INK_2)
    ax2.set_ylabel("training loss", fontsize=9, color=INK_2)
    ax2.set_xlim(right=df.epoch.max() + 4.2)
    ax2.set_title("(b) Training loss — near-identical across arms",
                  fontsize=9.5, color=INK, loc="left", pad=10)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, facecolor=SURFACE, bbox_inches="tight")

    best = df.loc[df.groupby("arm").val_gap.idxmax()]
    print(best[["arm", "epoch", "val_gap", "val_top1"]].to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
