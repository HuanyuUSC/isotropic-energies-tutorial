"""One visual language for every figure in the talk and the notebooks."""
import matplotlib as mpl

# validated categorical slots (light mode); use at most three together
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
ACCENT = "#b5005b"          # the deck's accent (titles, emphasis)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
LAND, GRID = "#d9d8d4", "#e6e5e1"
SEQ = "Blues"               # sequential: one hue, light -> dark


def use():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 14,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "mathtext.fontset": "cm",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def mirror_half(ax, lo=0.0, hi=3.0, label=True, fs=12):
    """Dim the half sigma_2 > sigma_1 of a sigma-plane plot: by isotropy it mirrors the other half,
    and sorted singular values never land there. Works on linear and log axes."""
    from matplotlib.patches import Polygon
    ax.add_patch(Polygon([[lo, lo], [lo, hi], [hi, hi]], closed=True, fc="white", alpha=0.62,
                         ec="none", zorder=3))
    ax.add_patch(Polygon([[lo, lo], [lo, hi], [hi, hi]], closed=True, fc="none", hatch="///",
                         ec=GRID, lw=0, zorder=3))
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.0, zorder=4)
    if label:
        ax.text(0.04, 0.96, "mirror image\n(σ₂ > σ₁)", transform=ax.transAxes, fontsize=fs,
                color=MUTED, va="top", zorder=5)
