"""
MalXplain dashboard charts.

Shared plotting helpers. Two categorical hues (blue, orange) validated for
colour vision deficiency separation, and a single hue sequential ramp for the
cross subtype recall matrix, which encodes magnitude rather than identity.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Categorical: validated pair, CVD separation dE 19.8 (protan), 27.0 normal vision.
BLUE = "#2F6FAD"
ORANGE = "#C8622B"
INK = "#16202B"
MUTED = "#6B7280"
GRID = "#E4E2DD"
SURFACE = "#FBFAF8"

# Sequential: one hue, light to dark, for magnitude.
SEQ = LinearSegmentedColormap.from_list("mx_seq", ["#FBFAF8", "#D8E3EE", "#8FB2CE", "#3F7BB3", "#143A5C"])


def _frame(ax, xlabel="", ylabel="", title=""):
    """Recessive axes and grid, so the data carries the ink."""
    ax.set_facecolor(SURFACE)
    ax.figure.patch.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    if title:
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=10)
    return ax


def reproduction_chart(df: pd.DataFrame):
    """Our accuracy against the paper's, per subtype. Two series, so legend plus labels."""
    d = df.dropna(subset=["Paper Accuracy"]).sort_values("Accuracy")
    y = np.arange(len(d))
    h = 0.36

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.barh(y + h / 2, d["Accuracy"] * 100, height=h, color=BLUE, label="MalXplain")
    ax.barh(y - h / 2, d["Paper Accuracy"] * 100, height=h, color=ORANGE, label="Base paper")

    ax.set_yticks(y)
    ax.set_yticklabels(d["Subtype"], fontsize=8.5, color=INK)
    ax.set_xlim(96, 100.4)
    _frame(ax, xlabel="Accuracy (%)", title="Reproduction fidelity: our results against the published figures")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK)
    plt.tight_layout()
    return fig


def tuning_chart(df: pd.DataFrame):
    """Change in unseen subtype recall after tuning. One series, so no legend."""
    d = df.sort_values("Δ unseen recall")
    vals = d["Δ unseen recall"] * 100
    colors = [BLUE if v >= 0 else ORANGE for v in vals]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(d["Subtype"], vals, color=colors, height=0.6)
    ax.axvline(0, color=MUTED, lw=1)
    _frame(ax, xlabel="Change in unseen subtype recall (percentage points)",
           title="Effect of hyperparameter tuning, which the base paper did not perform")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)

    for i, v in enumerate(vals):
        off = 0.03 if v >= 0 else -0.03
        ax.text(v + off, i, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=7.5, color=MUTED)
    plt.tight_layout()
    return fig


def classifier_chart(df: pd.DataFrame):
    """Mean unseen subtype recall by classifier family. One series."""
    d = df.sort_values("Mean_unseen_recall")
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    bars = ax.barh(d.index, d["Mean_unseen_recall"] * 100, color=BLUE, height=0.55)

    # Highlight Random Forest, the base paper's only choice in this setting.
    for bar, name in zip(bars, d.index):
        if name == "Random Forest":
            bar.set_color(ORANGE)

    ax.set_xlim(97, 99.2)
    _frame(ax, xlabel="Mean recall on unseen subtypes (%)",
           title="Classifier comparison under the zero shot protocol")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for i, v in enumerate(d["Mean_unseen_recall"] * 100):
        ax.text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=8, color=MUTED)
    plt.tight_layout()
    return fig


def blindspot_heatmap(matrix: pd.DataFrame):
    """
    Detector by target miss rate matrix. Magnitude, so a single hue sequential ramp.

    Miss rate (100 minus recall) is plotted rather than recall itself. Recall sits
    between 95 and 100 for most cells, so encoding it leaves the whole grid a uniform
    dark block with the failures pale and easy to miss, which is the opposite of what
    this chart is for. Miss rate keeps the convention that darker means a larger value,
    and puts the ink on the blind spots. The diagonal is the seen subtype, so it is masked.
    """
    m = matrix.to_numpy(dtype=float).copy()
    np.fill_diagonal(m, np.nan)
    miss = (1.0 - m) * 100.0

    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    im = ax.imshow(miss, cmap=SEQ, vmin=0, vmax=36, aspect="auto")

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8, color=INK)
    ax.set_yticklabels(matrix.index, fontsize=8, color=INK)

    ax.set_xlabel("Tested against this unseen subtype", fontsize=9, color=MUTED)
    ax.set_ylabel("Detector trained on", fontsize=9, color=MUTED)
    ax.set_title("Miss rate by detector and unseen subtype. Darker means more samples slipped through.",
                 fontsize=10.5, color=INK, loc="left", pad=12)

    # 2px surface gap between cells.
    ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linestyle="-", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0, colors=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)

    # Label only the cells that matter: where more than 5 percent was missed.
    # Ink colour follows cell darkness so the label always has contrast.
    for i in range(miss.shape[0]):
        for j in range(miss.shape[1]):
            v = miss[i, j]
            if not np.isnan(v) and v >= 5.0:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                        color="#FFFFFF" if v >= 18 else INK, fontweight="bold")

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Miss rate (%)", fontsize=9, color=MUTED)
    cb.ax.tick_params(labelsize=8, colors=MUTED, length=0)
    cb.outline.set_visible(False)
    fig.patch.set_facecolor(SURFACE)
    plt.tight_layout()
    return fig


def hardest_chart(series: pd.Series):
    """Mean recall per subtype when it is unseen. One series."""
    d = series.sort_values()
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    colors = [ORANGE if v < 0.95 else BLUE for v in d.values]
    ax.barh(d.index, d.values * 100, color=colors, height=0.58)
    ax.set_xlim(80, 100.5)
    _frame(ax, xlabel="Mean recall across the 14 detectors that never saw it (%)",
           title="How hard each subtype is to detect when it is unseen")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for i, v in enumerate(d.values * 100):
        ax.text(v + 0.2, i, f"{v:.1f}", va="center", fontsize=8, color=MUTED)
    plt.tight_layout()
    return fig


def shap_bar(evidence: pd.DataFrame):
    """Per feature SHAP contribution for a single prediction."""
    d = evidence.iloc[::-1]
    colors = [ORANGE if v > 0 else BLUE for v in d["shap_value"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.barh(d["feature"], d["shap_value"], color=colors, height=0.55)
    ax.axvline(0, color=MUTED, lw=1)
    _frame(ax, xlabel="SHAP contribution (left benign, right malware)",
           title="Per feature contribution to this decision")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7.5)
    plt.tight_layout()
    return fig
