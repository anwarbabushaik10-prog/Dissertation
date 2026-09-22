"""
Shared matplotlib styling so every figure in outputs/figures/ reads as one
consistent, report-ready system: clean light background, muted gridlines,
a fixed colorblind-safe categorical order (Okabe-Ito), and a single-hue
sequential ramp for magnitude (confusion matrices, heatmap overlays).
"""
import matplotlib.pyplot as plt

# Okabe-Ito: colorblind-safe, the standard choice for categorical scientific
# figures. Assigned in this fixed order everywhere — never re-cycled per plot.
CATEGORICAL = [
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
    "#56B4E9", "#F0E442", "#000000", "#999999",
]

SEQUENTIAL_CMAP = "Blues"       # magnitude (confusion matrix counts)
DIVERGING_CMAP = "RdBu_r"       # heatmap overlays (Grad-CAM signed or centred data)

INK = "#1a1a1a"
MUTED_INK = "#595959"
GRID = "#e3e3e3"
SURFACE = "#ffffff"


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": MUTED_INK,
        "ytick.color": MUTED_INK,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "figure.dpi": 130,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })
