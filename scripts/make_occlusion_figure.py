# Siddharth Mehta, CS5330 PRCV, Final Project
# Draws the figure showing how occlusion is spread across the dataset, with the
# four strata on one panel and the shape of the non zero tail on the other.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
GRID      = "#e3e2df"
RAMP      = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]

idx = pd.read_csv("manifests/occlusion_index.csv", dtype={"id": str})
r   = idx.occlusion_ratio.values
EDGES  = [0.0, 0.02, 0.10, 0.25, 1.01]
LABELS = ["none", "low", "medium", "high"]
counts = np.histogram(r, bins=EDGES)[0]
pct    = 100 * counts / counts.sum()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.5), facecolor=SURFACE)
for ax in (ax1, ax2):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=3, color=GRID)

bars = ax1.bar(LABELS, counts, color=RAMP, width=0.66, linewidth=0)
ax1.set_axisbelow(True)
ax1.yaxis.grid(True, color=GRID, linewidth=0.7)
for b, c, p in zip(bars, counts, pct):
    ax1.text(b.get_x() + b.get_width() / 2, c + counts.max() * 0.025,
             f"{c:,}\n{p:.1f}%", ha="center", va="bottom",
             fontsize=8, color=INK_2, linespacing=1.35)
ax1.set_ylim(0, counts.max() * 1.28)
ax1.yaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax1.set_ylabel("images", fontsize=9, color=INK_2)
ax1.set_xlabel("occlusion stratum", fontsize=9, color=INK_2)
ax1.set_title("(a) Evaluation strata, all 80,000 images",
              fontsize=9.5, color=INK, loc="left", pad=10)

nz = r[r > 0]
ax2.hist(nz, bins=60, color=RAMP[2], linewidth=0)
ax2.set_yscale("log")
ax2.set_axisbelow(True)
ax2.yaxis.grid(True, color=GRID, linewidth=0.7)
top = ax2.get_ylim()[1]
for edge, frac in zip(EDGES[1:-1], (0.55, 0.12, 0.55)):
    ax2.axvline(edge, color=INK_2, linestyle=(0, (3, 3)), linewidth=0.9, alpha=0.65)
    ax2.text(edge, top * frac, f" {edge:g}", fontsize=7, color=INK_2, va="top")
ax2.set_xlabel("occlusion ratio", fontsize=9, color=INK_2)
ax2.set_ylabel("images (log)", fontsize=9, color=INK_2)
ax2.set_title(f"(b) Non-zero occlusion only (n={len(nz):,}), stratum cuts dashed",
              fontsize=9.5, color=INK, loc="left", pad=10)

fig.tight_layout()
fig.savefig("report/figures/occlusion_distribution.png", dpi=220,
            facecolor=SURFACE, bbox_inches="tight")
print(f"n={len(r):,}  zero={np.mean(r == 0):.1%}  nonzero={len(nz):,}  max={r.max():.3f}")
print(dict(zip(LABELS, counts.tolist())))
