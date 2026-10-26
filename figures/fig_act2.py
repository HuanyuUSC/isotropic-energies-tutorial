"""Act 2 renders from the saved runs: flipped triangles in red.

  act2_linear.png   |F - I|^2, arm raised 100 deg: the arm shrivels and folds (22 flips)
  act2_arap.png     ARAP, same drag: the arm swings (0 flips)
  act2_twist.png    head twisted 180 deg: ARAP (31 flips) vs. Symmetric Dirichlet (0)

    python figures/fig_act2.py <out-dir>
"""
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from distortion import style
from distortion.planar import dets, gingerbread

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)
style.use()
V, F, G = gingerbread()
A = pickle.load(open(os.path.join(HERE, "..", "scratch", "act2.pkl"), "rb"))
T = pickle.load(open(os.path.join(HERE, "..", "scratch", "twist.pkl"), "rb"))
RED = "#d6402f"


def draw(ax, X, lim=None, title=None, pins=("feet",)):
    bad = dets(X, F) <= 0
    ax.add_collection(PolyCollection(X[F[~bad]], fc=style.LAND, ec=style.MUTED, lw=0.2))
    ax.add_collection(PolyCollection(X[F[bad]], fc=RED, ec=RED, lw=0.6, zorder=3))
    for g in pins:
        ax.plot(*X[G[g]].T, "s", color=style.INK, ms=2.5)
    if lim:
        ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1])
    else:
        ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=18, color=style.INK2)
    return int(bad.sum())


for key, name in [("raise_linear", "act2_linear"), ("raise_arap", "act2_arap")]:
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    k = draw(ax, A[key], lim=((-2.4, 2.1), (-2.3, 2.9)))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name + ".png"))
    plt.close(fig)
    print(name, "flips", k)

fig, axs = plt.subplots(1, 2, figsize=(10, 5.0))
lim = ((-1.5, 1.5), (0.3, 2.9))
for ax, (e, t) in zip(axs, [("arap", "ARAP"), ("symdir", "Symmetric Dirichlet")]):
    k = draw(ax, T[e][12], lim=lim, pins=())
    ax.set_title(f"{t}: {k} flipped", fontsize=18, color=RED if k else style.AQUA)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "act2_twist.png"))
print("ok")
