"""Why flips are bad: a fan of triangles around one vertex, each carrying a letter "F" (its texture).
Drag the center vertex across an edge: the triangles on that side flip (det F < 0). Their "F" is mirrored
and they cover their neighbours: the map folds the sheet over itself (not one-to-one).
    python figures/fig_flip.py      -> assets/fig/d1_flip_fold.png"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

from distortion import style

style.use()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "fig", "d1_flip_fold.png")
GOOD, BAD, EDGE = "#e7e5df", "#f4c7d9", "#55534e"

ring = np.array([[np.cos(a), np.sin(a)] for a in np.radians(np.arange(6) * 60 + 30)])
tris = [(0, 1 + k, 1 + (k + 1) % 6) for k in range(6)]
glyph = TextPath((0, 0), "F", size=1, prop=FontProperties(family="DejaVu Sans", weight="bold"))
gv = glyph.vertices
gv = (gv - gv.mean(0)) * 0.30


def draw(ax, X, title, sub, color_title):
    for t in tris:
        P = X[list(t)]
        e1, e2 = P[1] - P[0], P[2] - P[0]
        det = e1[0] * e2[1] - e1[1] * e2[0]
        ax.add_patch(Polygon(P, closed=True, fc=GOOD if det > 0 else BAD, ec=EDGE, lw=1.6,
                             alpha=1.0 if det > 0 else 0.85, zorder=3 if det < 0 else 2))
        # the triangle's texture: an "F" placed at the rest centroid, carried by the triangle's affine map
        R = V[list(t)]
        A = np.linalg.solve(np.c_[R[1] - R[0], R[2] - R[0]], np.eye(2))      # rest -> barycentric frame
        M = np.c_[e1, e2] @ A                                                 # the triangle's F
        c0 = R.mean(0)
        pts = (gv + c0 - R[0]) @ M.T + P[0]
        from matplotlib.path import Path
        from matplotlib.patches import PathPatch
        path = Path(pts, glyph.codes)
        ax.add_patch(PathPatch(path, fc="#b5005b" if det < 0 else "#2a78d6", ec="none", zorder=4 if det < 0 else 3))
    ax.plot(*X[0], "o", color="#1a1a1a", ms=7, zorder=6)
    ax.set_title(title, fontsize=19, color=color_title, pad=6)
    ax.text(0.5, -0.06, sub, transform=ax.transAxes, ha="center", va="top", fontsize=15, color="#555555")
    ax.set_xlim(-1.15, 1.95); ax.set_ylim(-1.2, 1.2); ax.set_aspect("equal"); ax.axis("off")


V = np.vstack([[0, 0], ring])
X = V.copy()
X[0] = [1.75, 0.08]                                    # the center vertex dragged out across the right edges
fig, axs = plt.subplots(1, 2, figsize=(10.5, 5.2))
draw(axs[0], V, "every triangle: det F > 0", "each \"F\" reads correctly", "#1a1a1a")
draw(axs[1], X, "drag one vertex too far", "flipped triangles (pink): mirrored \"F\",\nand they cover their neighbours",
     "#b5005b")
axs[1].annotate("", xy=X[0], xytext=V[0], arrowprops=dict(arrowstyle="->", color="#1a1a1a", lw=1.8, ls="--"))
fig.tight_layout()
fig.savefig(OUT, dpi=200)
print("wrote", OUT)
