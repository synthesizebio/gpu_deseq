#!/usr/bin/env python
"""Figure 1: high-level DESeq2 workflow and cuDESeq2 optimizations.

The figure is intentionally conceptual: the upper half presents the canonical
analysis path, while the lower half maps each cuDESeq2 systems contribution to
the iterative stage it accelerates.

Outputs:
  paper/figures/gpt_schematic.png
  paper/figures/gpt_schematic.pdf
  paper/figures/triton_dispersion_schematic.png
  paper/figures/triton_dispersion_schematic.pdf
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

# Color-blind-friendly palette with enough contrast for print.
BG = "#F7F9FC"
WHITE = "#FFFFFF"
NAVY = "#12233F"
TEXT = "#24324A"
MUTED = "#65748B"
LINE = "#CBD5E1"
BLUE = "#2878D0"
BLUE_TINT = "#EAF3FC"
RED = "#D94B55"
RED_TINT = "#FCEDEF"
GREEN = "#159A79"
GREEN_TINT = "#E8F7F2"
ORANGE = "#E77732"
ORANGE_TINT = "#FFF1E8"
PURPLE = "#6651B8"
PURPLE_TINT = "#F0EDFA"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": TEXT,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

fig, ax = plt.subplots(figsize=(16, 9))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set_xlim(0, 160)
ax.set_ylim(0, 90)
ax.axis("off")


def rounded(x, y, w, h, face, edge, lw=1.5, radius=1.6, z=2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(x0, y0, x1, y1, color=NAVY, lw=1.8, dashed=False, z=3):
    patch = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=lw,
        linestyle=(0, (4, 3)) if dashed else "-",
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=z,
    )
    ax.add_patch(patch)


def stage(x, number, title, lines, hot=False):
    y, w, h = 50.5, 25, 18
    edge, tint = (RED, RED_TINT) if hot else (LINE, WHITE)
    rounded(x, y, w, h, tint, edge, lw=2.1 if hot else 1.5)
    ax.add_patch(Rectangle((x, y), 1.35, h, facecolor=edge, edgecolor="none", zorder=3))

    # Stage number.
    rounded(x + 3.0, y + 12.0, 4.0, 4.0, edge, edge, lw=0, radius=2.0, z=4)
    ax.text(
        x + 5.0,
        y + 14.0,
        str(number),
        ha="center",
        va="center",
        color=WHITE,
        fontsize=10.5,
        fontweight="bold",
        zorder=5,
    )
    ax.text(
        x + 8.4,
        y + 14.0,
        title,
        ha="left",
        va="center",
        fontsize=12.2,
        fontweight="bold",
        color=NAVY,
        zorder=5,
    )
    ax.text(
        x + 3.0,
        y + 9.5,
        lines,
        ha="left",
        va="top",
        fontsize=9.5,
        linespacing=1.28,
        color=TEXT,
        zorder=5,
    )
    label = "ITERATIVE · PER GENE" if hot else "VECTORIZED"
    ax.text(
        x + 3.0,
        y + 2.0,
        label,
        ha="left",
        va="center",
        fontsize=7.7,
        fontweight="bold",
        color=RED if hot else MUTED,
        zorder=5,
    )


def optimization_card(x, y, w, h, number, color, tint, title, body):
    rounded(x, y, w, h, WHITE, color, lw=1.8, radius=1.5, z=4)
    ax.add_patch(Rectangle((x, y), 1.2, h, facecolor=color, edgecolor="none", zorder=5))
    rounded(x + 3.0, y + h - 5.2, 4.0, 3.6, color, color, lw=0, radius=1.8, z=5)
    ax.text(
        x + 5.0,
        y + h - 3.4,
        number,
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        color=WHITE,
        zorder=6,
    )
    ax.text(
        x + 8.2,
        y + h - 3.4,
        title,
        ha="left",
        va="center",
        fontsize=10.8,
        fontweight="bold",
        color=NAVY,
        zorder=6,
    )
    ax.text(
        x + 3.0,
        y + h - 7.2,
        body,
        ha="left",
        va="top",
        fontsize=8.0,
        linespacing=1.28,
        color=TEXT,
        zorder=6,
    )
    return tint


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
ax.text(
    6,
    84.0,
    "DESeq2 workflow mapped to GPU execution",
    fontsize=20,
    fontweight="bold",
    color=NAVY,
    ha="left",
    va="center",
)
ax.text(
    6,
    79.5,
    "cuDESeq2 retains the statistical model and inference sequence while changing the execution strategy.",
    fontsize=11.4,
    color=MUTED,
    ha="left",
    va="center",
)

# Global execution rail.
rounded(6, 72.0, 148, 4.8, BLUE_TINT, BLUE, lw=1.5, radius=1.3, z=1)
ax.add_patch(Rectangle((6, 72.0), 1.5, 4.8, facecolor=BLUE, edgecolor="none", zorder=2))
ax.text(
    80,
    74.4,
    "Same model and inference  ·  fp64 arithmetic  ·  thousands of genes evaluated concurrently",
    fontsize=10.8,
    fontweight="bold",
    color=NAVY,
    ha="center",
    va="center",
)

# ---------------------------------------------------------------------------
# Canonical DESeq2 workflow
# ---------------------------------------------------------------------------
stage_x = [6, 37, 68, 99, 130]
stage(stage_x[0], 1, "Normalize", "Median-of-ratios\nsize factors", hot=False)
stage(stage_x[1], 2, "Estimate dispersion", "Gene-wise CR-MLE\n→ trend → MAP", hot=True)
stage(stage_x[2], 3, "Fit & test GLM", "Negative-binomial IRLS\nWald test or LRT", hot=True)
stage(stage_x[3], 4, "Control errors", "Cook's + independent filter\nBenjamini–Hochberg", hot=False)
stage(stage_x[4], 5, "Shrink LFC", "apeGLM Cauchy-prior\nMAP estimate", hot=True)

for i in range(4):
    arrow(stage_x[i] + 25.8, 59.5, stage_x[i + 1] - 0.8, 59.5, color=NAVY, lw=2.0)

ax.text(
    80,
    46.8,
    "OPTIMIZATION LAYER",
    fontsize=8.5,
    fontweight="bold",
    color=MUTED,
    ha="center",
    va="center",
)
ax.plot([6, 154], [45.2, 45.2], color=LINE, linewidth=1.2, zorder=1)

# ---------------------------------------------------------------------------
# cuDESeq2 contributions
# ---------------------------------------------------------------------------
# The batch card spans the two general iterative fitting stages.
optimization_card(
    6,
    27.5,
    45,
    14,
    "A",
    BLUE,
    BLUE_TINT,
    "Batched, reference-faithful port",
    "Replace serial gene loops with fp64\n"
    "tensor operations while retaining the same\n"
    "clamps, line searches, and fallback rules.",
)
optimization_card(
    55,
    27.5,
    30,
    14,
    "B",
    GREEN,
    GREEN_TINT,
    "CUDA Graph replay",
    "Replay short iteration chunks to\n"
    "remove kernel-launch overhead.\n"
    "Capture-safe no-pivot LU enables graphs.",
)
optimization_card(
    89,
    27.5,
    30,
    14,
    "C",
    ORANGE,
    ORANGE_TINT,
    "Fused dispersion kernel",
    "One program per gene runs the\n"
    "gradient-ascent loop in registers, with\n"
    "independent early convergence.",
)
optimization_card(
    123,
    27.5,
    31,
    14,
    "D",
    PURPLE,
    PURPLE_TINT,
    "Batched LFC shrinkage",
    "Reference-compatible L-BFGS with\n"
    "compiled loss, gradient, and Hessian\n"
    "evaluation on resident GPU tensors.",
)

# Mapping from workflow bottlenecks to the relevant optimizations.
arrow(49.5, 50.5, 28.5, 41.8, color=BLUE, lw=1.5, dashed=True)
arrow(80.5, 50.5, 36.0, 41.8, color=BLUE, lw=1.5, dashed=True)
arrow(49.5, 50.5, 70.0, 41.8, color=GREEN, lw=1.5, dashed=True)
arrow(49.5, 50.5, 104.0, 41.8, color=ORANGE, lw=1.5, dashed=True)
arrow(142.5, 50.5, 138.5, 41.8, color=PURPLE, lw=1.5, dashed=True)

# Outcome strip.
rounded(6, 17.8, 148, 5.6, NAVY, NAVY, lw=0, radius=1.4, z=2)
ax.text(
    80,
    20.6,
    "Outcome: preserved inference  ·  gene-parallel optimization  ·  lower dispatch overhead",
    fontsize=10.7,
    fontweight="bold",
    color=WHITE,
    ha="center",
    va="center",
)

# Compact legend / reading guide.
ax.add_patch(Rectangle((6, 10.7), 2.4, 1.4, facecolor=RED, edgecolor="none"))
ax.text(9.5, 11.4, "per-gene iterative bottleneck", fontsize=8.6, color=MUTED, va="center")
ax.add_patch(Rectangle((42, 10.7), 2.4, 1.4, facecolor=BLUE, edgecolor="none"))
ax.text(45.5, 11.4, "pipeline-wide GPU batching", fontsize=8.6, color=MUTED, va="center")
ax.plot([77, 80], [11.4, 11.4], color=GREEN, linewidth=1.8, linestyle=(0, (4, 3)))
ax.text(81.2, 11.4, "optimization-to-stage mapping", fontsize=8.6, color=MUTED, va="center")

ax.text(
    6,
    5.8,
    "DESeq2 stages follow Love et al.; apeGLM follows Zhu et al.  CR-MLE: Cox–Reid maximum-likelihood estimate; "
    "LFC: log2 fold change.",
    fontsize=8.0,
    color=MUTED,
    ha="left",
    va="center",
)

for extension in ("png", "pdf"):
    fig.savefig(
        OUT / f"gpt_schematic.{extension}",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08,
    )

print(f"wrote {OUT / 'gpt_schematic.png'} and {OUT / 'gpt_schematic.pdf'}")
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: detailed view of the fused dispersion kernel
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 8.4))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set_xlim(0, 160)
ax.set_ylim(0, 84)
ax.axis("off")

ax.text(
    6,
    78.5,
    "Fused Triton dispersion estimation",
    fontsize=20,
    fontweight="bold",
    color=NAVY,
    ha="left",
    va="center",
)
ax.text(
    6,
    74.2,
    "The same per-gene kernel evaluates the Cox–Reid MLE and the prior-regularized MAP estimate.",
    fontsize=11.2,
    color=MUTED,
    ha="left",
    va="center",
)


def flow_box(x, w, title, detail, accent=None):
    edge = accent or LINE
    face = GREEN_TINT if accent == GREEN else WHITE
    rounded(x, 59.5, w, 9.5, face, edge, lw=1.8 if accent else 1.3, radius=1.3)
    ax.text(x + w / 2, 65.3, title, fontsize=11.3, fontweight="bold",
            color=NAVY, ha="center", va="center")
    ax.text(x + w / 2, 61.9, detail, fontsize=8.4, color=MUTED,
            ha="center", va="center")


flow_box(6, 25, "Gene inputs", r"counts $y_g$  ·  fitted means $\mu_g$  ·  design $X$")
flow_box(38, 27, "Cox–Reid MLE", "gene-wise dispersion", GREEN)
flow_box(72, 21, "Trend fit", "shared across genes")
flow_box(100, 27, "MAP estimate", "Gaussian prior", GREEN)
flow_box(134, 20, "Final dispersion", r"$\alpha_g$")
for x0, x1 in ((31.5, 37.5), (65.5, 71.5), (93.5, 99.5), (127.5, 133.5)):
    arrow(x0, 64.25, x1, 64.25, color=NAVY, lw=1.8)
ax.text(51.5, 70.4, "FUSED", fontsize=7.7, fontweight="bold", color=GREEN,
        ha="center", va="center")
ax.text(82.5, 70.4, "GLOBAL FIT", fontsize=7.7, fontweight="bold", color=MUTED,
        ha="center", va="center")
ax.text(113.5, 70.4, "FUSED", fontsize=7.7, fontweight="bold", color=GREEN,
        ha="center", va="center")

ax.plot([6, 154], [55.8, 55.8], color=LINE, linewidth=1.2, zorder=1)

# GPU grid: the gene dimension becomes the launch grid.
rounded(6, 11, 48, 39.8, WHITE, GREEN, lw=1.8, radius=1.5, z=2)
ax.text(10, 46.7, "GPU launch grid", fontsize=13, fontweight="bold", color=NAVY,
        ha="left", va="center")
ax.text(10, 43.0, "one independent program per gene", fontsize=9.2, color=MUTED,
        ha="left", va="center")

tile_x = [10, 20.5, 31, 41.5]
tile_y = [33, 23]
tile_labels = [r"$g_0$", r"$g_1$", r"$g_2$", r"$g_3$",
               r"$g_4$", r"$g_5$", r"$\cdots$", r"$g_{G-1}$"]
for idx, (yy, xx) in enumerate((y, x) for y in tile_y for x in tile_x):
    rounded(xx, yy, 8.5, 7.0, GREEN_TINT, GREEN, lw=1.1, radius=0.9, z=3)
    ax.text(xx + 4.25, yy + 3.5, tile_labels[idx], fontsize=10.5,
            fontweight="bold", color=GREEN, ha="center", va="center")
ax.text(30, 15.9, "genes execute concurrently", fontsize=9.0, color=GREEN,
        fontweight="bold", ha="center", va="center")

arrow(56.0, 30.9, 65.0, 30.9, color=GREEN, lw=2.0)
ax.text(60.5, 34.1, "inspect one", fontsize=8.0, color=MUTED,
        ha="center", va="center")

# One program: data are loaded once and the complete optimization loop remains
# within the kernel rather than returning to the Python dispatcher.
rounded(66, 8.5, 88, 44.8, WHITE, GREEN, lw=1.8, radius=1.5, z=2)
ax.text(70, 49.0, r"Inside program $g$", fontsize=13, fontweight="bold",
        color=NAVY, ha="left", va="center")
ax.text(150, 49.0, "register-resident state", fontsize=8.5, color=GREEN,
        fontweight="bold", ha="right", va="center")

rounded(71, 33.2, 23, 10, BLUE_TINT, BLUE, lw=1.3, radius=1.0, z=3)
ax.text(82.5, 39.5, "Load once", fontsize=10.5, fontweight="bold", color=NAVY,
        ha="center", va="center")
ax.text(82.5, 35.9, r"$y_g,\ \mu_g,\ X$", fontsize=10.0, color=BLUE,
        ha="center", va="center")

rounded(100, 31.2, 28, 14, GREEN_TINT, GREEN, lw=1.3, radius=1.0, z=3)
ax.text(114, 40.9, "Evaluate objective", fontsize=10.5, fontweight="bold",
        color=NAVY, ha="center", va="center")
ax.text(114, 36.6, "NB likelihood + Cox–Reid", fontsize=8.4, color=TEXT,
        ha="center", va="center")
ax.text(114, 33.4, r"$\ell(a)$ and $\partial\ell/\partial a$", fontsize=9.0,
        color=GREEN, ha="center", va="center")

rounded(133.5, 33.2, 15.5, 10, ORANGE_TINT, ORANGE, lw=1.3, radius=1.0, z=3)
ax.text(141.25, 39.5, "Armijo step", fontsize=9.6, fontweight="bold", color=NAVY,
        ha="center", va="center")
ax.text(141.25, 35.9, r"update $a=\log\alpha$", fontsize=8.4, color=ORANGE,
        ha="center", va="center")

arrow(94.8, 38.2, 99.5, 38.2, color=NAVY, lw=1.6)
arrow(128.5, 38.2, 133.0, 38.2, color=NAVY, lw=1.6)

loop = FancyArrowPatch(
    (141.0, 32.5),
    (114.0, 30.5),
    connectionstyle="arc3,rad=-0.35",
    arrowstyle="-|>",
    mutation_scale=14,
    linewidth=1.5,
    color=ORANGE,
    zorder=4,
)
ax.add_patch(loop)
ax.text(128, 24.7, "repeat until this gene converges", fontsize=8.3,
        color=MUTED, ha="center", va="center")

rounded(98, 12.8, 35, 7.0, NAVY, NAVY, lw=0, radius=1.0, z=3)
ax.text(115.5, 16.3, r"write converged $\alpha_g$", fontsize=9.5,
        fontweight="bold", color=WHITE, ha="center", va="center")
arrow(141.2, 32.7, 133.0, 20.1, color=NAVY, lw=1.5)

ax.text(
    6,
    4.3,
    "CR-MLE: Cox–Reid maximum-likelihood estimate. The trend fit is shared across genes; both gene-wise fits use the fused kernel.",
    fontsize=8.0,
    color=MUTED,
    ha="left",
    va="center",
)

for extension in ("png", "pdf"):
    fig.savefig(
        OUT / f"triton_dispersion_schematic.{extension}",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08,
    )

print(
    f"wrote {OUT / 'triton_dispersion_schematic.png'} and "
    f"{OUT / 'triton_dispersion_schematic.pdf'}"
)
plt.close(fig)
