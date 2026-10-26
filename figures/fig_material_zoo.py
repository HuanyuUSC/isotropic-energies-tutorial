"""The variety of materials used in physical simulation (after slide 19 of the SCA 2026 deck): eight common
isotropic 3D energies, all with the same Young's modulus E = 1 and Poisson's ratio nu = 0.25 (Lame mu = lambda = 0.4),
each as a fingerprint: psi(sigma1, sigma2, 1) over [0, 2]^2, shared colors and contour levels.
    python figures/fig_material_zoo.py <out-dir>      -> d5_material_zoo.png"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from distortion import style

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
style.use()
E, NU = 1.0, 0.25
MU, LAM = E / (2 * (1 + NU)), E * NU / ((1 + NU) * (1 - 2 * NU))      # 0.4, 0.4

L = 2.0
g = np.linspace(0.004, L, 400)
S1, S2 = np.meshgrid(g, g)
S = np.stack([S1, S2, np.ones_like(S1)], -1)                           # the sigma3 = 1 slice
J = S.prod(-1)
lg = lambda x: np.log(np.where(x > 0, x, np.nan))
sm = lambda f: f(S).sum(-1)

ZOO = [
    ("Linear Corotational", r"$\mu\sum(\sigma_i-1)^2+\frac{\lambda}{2}(\sum\sigma_i-3)^2$",
     lambda: MU * sm(lambda s: (s - 1) ** 2) + LAM / 2 * (sm(lambda s: s) - 3) ** 2),
    ("St. Venant-Kirchhoff", r"$\frac{\mu}{4}\sum(\sigma_i^2-1)^2+\frac{\lambda}{8}(\sum\sigma_i^2-3)^2$",
     lambda: MU / 4 * sm(lambda s: (s ** 2 - 1) ** 2) + LAM / 8 * (sm(lambda s: s ** 2) - 3) ** 2),
    ("Vol.-corrected Lin. Corot.", r"$\mu\sum(\sigma_i-1)^2+\frac{\lambda}{2}\log^2 J$",
     lambda: MU * sm(lambda s: (s - 1) ** 2) + LAM / 2 * lg(J) ** 2),
    ("Vol.-corrected StVK", r"$\frac{\mu}{4}\sum(\sigma_i^2-1)^2+\frac{\lambda}{2}\log^2 J$",
     lambda: MU / 4 * sm(lambda s: (s ** 2 - 1) ** 2) + LAM / 2 * lg(J) ** 2),
    ("Neo-Hookean", r"$\frac{\mu}{2}(\sum\sigma_i^2-3)-\mu\log J+\frac{\lambda}{2}\log^2 J$",
     lambda: MU / 2 * (sm(lambda s: s ** 2) - 3) - MU * lg(J) + LAM / 2 * lg(J) ** 2),
    ("Neo-Hookean (Ogden)", r"$\frac{\mu}{2}(\sum\sigma_i^2-3)-\mu\log J+\frac{\lambda}{2}(J-1)^2$",
     lambda: MU / 2 * (sm(lambda s: s ** 2) - 3) - MU * lg(J) + LAM / 2 * (J - 1) ** 2),
    ("Stable Neo-Hookean", r"$\frac{\mu}{2}(\sum\sigma_i^2-3)-\mu(J-1)+\frac{\lambda+\mu}{2}(J-1)^2$",
     lambda: MU / 2 * (sm(lambda s: s ** 2) - 3) - MU * (J - 1) + (LAM + MU) / 2 * (J - 1) ** 2),
    ("Hencky", r"$\mu\sum\log^2\sigma_i+\frac{\lambda}{2}\log^2 J$",
     lambda: MU * sm(lambda s: lg(s) ** 2) + LAM / 2 * lg(J) ** 2),
]

levels = np.array([0.01, 0.03, 0.06, 0.1, 0.16, 0.25, 0.4, 0.6, 0.9, 1.3, 1.9])
fig, axs = plt.subplots(2, 4, figsize=(15, 8.2))
for ax, (name, formula, f) in zip(axs.ravel(), ZOO):
    Z = f()
    Z = Z - np.nanmin(np.where(np.hypot(S1 - 1, S2 - 1) < 0.02, Z, np.nan))   # psi(rest) = 0
    C = np.log1p(np.clip(np.nan_to_num(Z, nan=1e9), 0, None) * 2.5)
    ax.pcolormesh(S1, S2, C, cmap="YlOrRd_r", vmin=0, vmax=np.log1p(2.5 * 2.2), shading="auto", rasterized=True)
    ax.contour(S1, S2, np.nan_to_num(Z, nan=1e9), levels=levels, colors="#222222", linewidths=0.6, alpha=0.85)
    ax.plot([1], [1], "o", ms=8, mfc="white", mec="#111111", mew=1.4, zorder=5)
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal")
    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2]); ax.tick_params(labelsize=10, length=2)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("#222222"); sp.set_linewidth(0.9)
    ax.set_title(name, fontsize=15, color=style.INK2, pad=5)
    ax.text(0.5, -0.13, formula, transform=ax.transAxes, ha="center", va="top", fontsize=12.5, color=style.INK)
fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.08, wspace=0.12, hspace=0.42)
fig.savefig(os.path.join(OUT, "d5_material_zoo.png"))
print("ok")
