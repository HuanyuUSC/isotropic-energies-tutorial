"""Energy personalities: the same drag (raise the arm 100 deg and pull it 40% longer) under four energies.
Triangles colored by area change (log2 of the area ratio; blue shrunk, orange grown).

    python figures/fig_personalities.py <out-dir>
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from distortion import style
from distortion.eigensystem import ARAP, Conformal2D, NeoHookean, SymmetricDirichlet
from distortion.planar import DragProblem, dets, gingerbread

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
os.makedirs(OUT, exist_ok=True)
style.use()
V, F, G = gingerbread()
rot = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
shoulder = np.array([0.45, 0.75])


def drag(energy, deg=100, stretch=1.4, steps=12):
    P = DragProblem(V, F, energy, G["feet"], G["right_hand"], k=1e3)
    X = V.copy()
    for t in np.linspace(0, 1, steps + 1)[1:]:
        P.target = (V[G["right_hand"]] - shoulder) @ rot(np.radians(deg) * t).T * (1 + (stretch - 1) * t) + shoulder
        X = P.solve(X, "pnewton", iters=60)
    return X


def diverging(t):
    lo, mid, hi = np.array([0.165, 0.471, 0.839]), np.array([0.94, 0.937, 0.925]), np.array([0.922, 0.408, 0.204])
    t = np.clip(t, -1, 1)[:, None]
    return np.where(t < 0, mid + (lo - mid) * -t, mid + (hi - mid) * t)


runs = [("ARAP", ARAP()), ("Symmetric Dirichlet", SymmetricDirichlet()),
        ("Conformal", Conformal2D()), ("Neo-Hookean", NeoHookean(1.0, 1.0))]
fig, axs = plt.subplots(1, 4, figsize=(16, 5.2))
A0 = dets(V, F)
for ax, (name, e) in zip(axs, runs):
    X = drag(e)
    r = np.log2(np.abs(dets(X, F)) / A0)
    flips = int((dets(X, F) <= 0).sum())
    ax.add_collection(PolyCollection(X[F], fc=diverging(r / 1.5), ec="none"))
    ax.plot(*X[G["feet"]].T, "s", color=style.INK, ms=2.5)
    ax.set_xlim(-2.4, 2.6)
    ax.set_ylim(-2.3, 2.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(name, fontsize=19, color=style.INK2)
    print(f"{name:20s} area x{np.exp2(r).min():.2f}..x{np.exp2(r).max():.2f}  flips {flips}")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "personalities.png"))
