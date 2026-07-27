#!/usr/bin/env python
"""Figure: the optimizations in cuDESeq2 and their measured impact.

All numbers are pulled from benchmarks/results_per_step.json (60 x 20000 case,
single A100 vs R DESeq2 1.30.1, 12-core host) and the two ablations logged in
benchmarks/RESULTS.md (fixed-vs-chunked CUDA graph; apeGLM Newton->LM fix).

Panel A  per-step wall time, 60x20000, log-x, R 12-core vs the three GPU modes
Panel B  dispersion optimization ladder: eager -> +CUDA graph -> +Triton
Panel C  apeGLM shrinkage: Newton+scipy rescue -> Levenberg-Marquardt

Outputs paper/figures/optimizations.{pdf,png}.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(REPO, "benchmarks", "results_per_step.json")) as f:
    DATA = json.load(f)
CASE = DATA["tables"]["60x20000_cond"]

# ---- dataviz palette (light surface) --------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE = "#2a78d6"    # eager
AQUA = "#1baf7a"    # +CUDA graph
ORANGE = "#eb6834"  # +Triton
RGRAY = "#898781"   # R baseline
CRIT = "#d03b3b"    # "before" (slow) state
GOOD = "#0ca30c"

plt.rcParams.update({
    "font.family": ["DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": BASE,
    "axes.linewidth": 0.8,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": INK2,
})

fig = plt.figure(figsize=(12.0, 7.4))
gs = fig.add_gridspec(
    2, 2, height_ratios=[1.15, 1.0], width_ratios=[1.0, 1.0],
    hspace=0.42, wspace=0.22,
    left=0.11, right=0.965, top=0.86, bottom=0.09,
)
axA = fig.add_subplot(gs[0, :])
axB = fig.add_subplot(gs[1, 0])
axC = fig.add_subplot(gs[1, 1])

fig.suptitle(
    "Optimizations in cuDESeq2 and their measured impact",
    x=0.11, ha="left", y=0.965, fontsize=16, fontweight="bold", color=INK,
)
fig.text(
    0.11, 0.905,
    "60 samples × 20,000 genes  ·  single NVIDIA A100  vs  R DESeq2 1.30.1 (12-core host)  ·  bit-exact parity (92/92 tests)",
    ha="left", fontsize=10.5, color=INK2,
)


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0)


# ===========================================================================
# Panel A: per-step wall time, log-x grouped horizontal bars
# ===========================================================================
STEPS = ["normalization", "dispersion", "glm_fit", "significance", "lfc_shrink"]
LABELS = ["Normalization", "Dispersion", "GLM fit", "Significance", "LFC shrink"]
# row -> [step, R_1thr, R_12core, eager, graph, triton]
rowmap = {r[0]: r for r in CASE["rows"]}
series = [
    ("R 12-core", RGRAY, 2),   # index into row (R_12core)
    ("eager (batched GPU)", BLUE, 3),
    ("+ CUDA graph", AQUA, 4),
    ("+ Triton", ORANGE, 5),
]

n_ser = len(series)
group_h = 0.80
bar_h = group_h / n_ser
y0 = list(range(len(STEPS)))

for si, (name, color, idx) in enumerate(series):
    ys = [y + group_h / 2 - (si + 0.5) * bar_h for y in y0]
    vals = []
    for st in STEPS:
        v = rowmap[st][idx]
        vals.append(v if v is not None else float("nan"))
    axA.barh(ys, vals, height=bar_h * 0.86, color=color, label=name,
             zorder=3, edgecolor=SURFACE, linewidth=0.6)
    for y, v, st in zip(ys, vals, STEPS):
        if v != v:  # NaN (glm bundled into R dispersion)
            axA.text(1.15, y, "bundled", va="center", ha="left",
                     fontsize=7.2, color=MUTED, style="italic", zorder=4)
            continue
        txt = f"{v:,.0f}" if v >= 10 else f"{v:.1f}"
        axA.text(v * 1.14, y, txt, va="center", ha="left",
                 fontsize=7.6, color=INK2, zorder=4)

axA.set_xscale("log")
axA.set_xlim(0.7, 45000)
axA.set_ylim(-0.55, len(STEPS) - 0.30)
axA.set_yticks(y0)
axA.set_yticklabels(LABELS, fontsize=10.5, color=INK)
axA.invert_yaxis()
axA.set_xlabel("wall time per step (ms, log scale)", fontsize=9.5)
axA.xaxis.grid(True, which="major", color=GRID, linewidth=0.7, zorder=0)
axA.set_axisbelow(True)
clean(axA)
axA.legend(loc="lower right", frameon=False, fontsize=9.0, ncol=2,
           handlelength=1.1, columnspacing=1.3, labelspacing=0.35,
           bbox_to_anchor=(1.0, -0.02))
axA.set_title("A   Where the optimizations act  —  per-step wall time",
              loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=8)

# total callout (upper-right, inside the empty corner of the log plot)
tot = CASE["total"]
axA.text(
    0.985, 0.88,
    f"End-to-end total\nR 12-core {tot['R_multicore']/1000:.1f}s  →  "
    f"Triton {tot['triton']:.0f} ms\n({tot['R_multicore']/tot['triton']:.0f}× faster)",
    transform=axA.transAxes, ha="right", va="top", fontsize=9.4,
    color=GOOD, fontweight="bold", linespacing=1.4,
)


# ===========================================================================
# Panel B: dispersion optimization ladder
# ===========================================================================
disp = rowmap["dispersion"]
eager_d, graph_d, triton_d = disp[3], disp[4], disp[5]
ladder = [
    ("eager\n(batched GPU port)", eager_d, BLUE),
    ("+ CUDA-graph\nchunked replay", graph_d, AQUA),
    ("+ Triton\nfull-loop fusion", triton_d, ORANGE),
]
yb = [2, 1, 0]
for (lab, v, c), y in zip(ladder, yb):
    axB.barh(y, v, height=0.52, color=c, zorder=3)
    axB.text(v + 6, y, f"{v:.0f} ms", va="center", ha="left",
             fontsize=10, color=INK, fontweight="bold", zorder=4)

axB.set_yticks(yb)
axB.set_yticklabels([l for l, _, _ in ladder], fontsize=9.6, color=INK)
axB.set_xlim(0, eager_d * 1.32)
axB.set_ylim(-0.55, 2.55)
axB.set_xlabel("dispersion-fitting time (ms)", fontsize=9.5)
axB.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
axB.set_axisbelow(True)
clean(axB)
axB.set_title("B   Dispersion loop: two systems optimizations",
              loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=8)

# speedup arrows between rungs
def speed_arrow(ax, y_from, y_to, x, txt):
    ax.annotate(
        "", xy=(x, y_to + 0.24), xytext=(x, y_from - 0.24),
        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.3,
                        connectionstyle="arc3,rad=0.0"), zorder=5,
    )
    ax.text(x + 8, (y_from + y_to) / 2, txt, va="center", ha="left",
            fontsize=8.6, color=INK2, style="italic")

xa = eager_d * 0.60
speed_arrow(axB, 2, 1, xa, f"{eager_d/graph_d:.1f}×\nremoves kernel\nlaunch overhead")
speed_arrow(axB, 1, 0, xa, f"{graph_d/triton_d:.1f}×\nregisters +\nper-gene early exit")
axB.text(0.99, -0.30,
         f"{eager_d/triton_d:.1f}× over eager   ·   "
         f"chunked graph beats fixed-length capture (0.8×)",
         transform=axB.transAxes, ha="right", va="top", fontsize=7.8,
         color=MUTED)


# ===========================================================================
# Panel C: apeGLM shrinkage fix (Newton+scipy -> Levenberg-Marquardt)
# ===========================================================================
# ablation from RESULTS.md 2026-07-14: 5285 ms -> 91 ms (58x); 500 -> 39 iters
before_ms, after_ms = 5285.0, 91.0
yc = [1, 0]
axC.barh(1, before_ms, height=0.52, color=CRIT, zorder=3)
axC.barh(0, after_ms, height=0.52, color=GOOD, zorder=3)
axC.text(before_ms - 90, 1, f"{before_ms:,.0f} ms", va="center", ha="right",
         fontsize=10, color="white", fontweight="bold", zorder=4)
axC.text(after_ms + 70, 0, f"{after_ms:.0f} ms", va="center", ha="left",
         fontsize=10, color=INK, fontweight="bold", zorder=4)

axC.set_yticks(yc)
axC.set_yticklabels(
    ["Newton + per-gene\nscipy rescue", "Levenberg–Marquardt\n+ pinned boundary"],
    fontsize=9.6, color=INK,
)
axC.set_xlim(0, before_ms * 1.16)
axC.set_ylim(-0.55, 1.55)
axC.set_xlabel("LFC-shrinkage time (ms)", fontsize=9.5)
axC.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
axC.set_axisbelow(True)
clean(axC)
axC.set_title("C   apeGLM shrinkage: fixing a convergence failure",
              loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=8)

axC.annotate(
    "", xy=(before_ms * 0.52, 0.30), xytext=(before_ms * 0.52, 0.70),
    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.3), zorder=5,
)
axC.text(before_ms * 0.52 + 90, 0.5,
         f"{before_ms/after_ms:.0f}×\n500 → 39 iters\n0 scipy fallbacks",
         va="center", ha="left", fontsize=8.6, color=INK2, style="italic")
axC.text(0.99, -0.30,
         "indefinite Hessian on extreme-LFC genes → LM damping restores descent",
         transform=axC.transAxes, ha="right", va="top", fontsize=7.8, color=MUTED)

fig.text(0.11, 0.018,
         "Source: benchmarks/results_per_step.json (commit f2f13a8) + benchmarks/RESULTS.md ablations.",
         ha="left", fontsize=7.6, color=MUTED)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"optimizations.{ext}"), dpi=200,
                bbox_inches="tight")
print("wrote", os.path.join(OUT, "optimizations.pdf"),
      "and", os.path.join(OUT, "optimizations.png"))
