"""Energy fingerprints (style of writeup/materialFiltering/fig/fingerprints): each energy as a landscape over the
(sigma1, sigma2) square, linear axes 0..2 so the degenerate edges sigma = 0 show, every energy scaled to the same
curvature at rest (ARAP's), shared color scale and contour levels. Conformal shows a valley along the whole
diagonal (size is free); the other three have one well at (1, 1).
    python figures/fig_fingerprints.py <out-dir>      -> d1_fingerprints.png"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from distortion import style
from distortion.eigensystem import ARAP, Conformal2D, NeoHookean, SymmetricDirichlet

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
os.makedirs(OUT, exist_ok=True)
style.use()

L = 2.0
g = np.linspace(0.004, L, 500)                      # linear, down to (almost) sigma = 0
S1, S2 = np.meshgrid(g, g)
S = np.stack([S1.ravel(), S2.ravel()], 1)
energies = [("ARAP", ARAP()), ("Symmetric Dirichlet", SymmetricDirichlet()),
            ("Conformal", Conformal2D()), ("Neo-Hookean", NeoHookean(1.0, 1.0))]


def scaled(e):
    """psi - psi(rest), scaled so the curvature along one stretch at rest is 2 (ARAP's)."""
    h = 1e-4
    p = lambda a, b: e.psi(np.array([[a, b]], float))[0]
    k = (p(1 + h, 1) - 2 * p(1, 1) + p(1 - h, 1)) / h ** 2
    return ((e.psi(S) - p(1, 1)) * 2 / k).reshape(S1.shape)


levels = np.array([0.02, 0.06, 0.12, 0.2, 0.35, 0.55, 0.8, 1.2, 1.8, 2.6, 3.6])
fig, axs = plt.subplots(2, 2, figsize=(7.0, 7.0))
for ax, (name, e) in zip(axs.ravel(), energies):
    Z = scaled(e)
    C = np.log1p(np.clip(Z, 0, None))
    ax.pcolormesh(S1, S2, C, cmap="YlOrRd_r", vmin=0, vmax=np.log1p(4.0), shading="auto", rasterized=True)
    ax.contour(S1, S2, Z, levels=levels, colors="#222222", linewidths=0.7, alpha=0.85)
    ax.plot([1], [1], "o", ms=10, mfc="white", mec="#111111", mew=1.6, zorder=5)
    ax.set_title(name, fontsize=17, color=style.INK2, pad=6)
    t = [0, 1, 2]
    ax.set_xticks(t); ax.set_yticks(t)
    ax.set_xticklabels(["0", "1", "2"], fontsize=12); ax.set_yticklabels(["0", "1", "2"], fontsize=12)
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("#222222"); sp.set_linewidth(1)
for ax in axs[1]:
    ax.set_xlabel(r"$\sigma_1$", fontsize=16, labelpad=0)
for ax in axs[:, 0]:
    ax.set_ylabel(r"$\sigma_2$", fontsize=16, rotation=0, labelpad=10)
fig.tight_layout(h_pad=1.2, w_pad=1.0)
fig.savefig(os.path.join(OUT, "d1_fingerprints.png"))
print("ok")
