"""Auxiliary 1D panels: an isotropic energy along one principal stretch, psi_hat(sigma) per axis.

  d1_curve_arap_vs_symdir.png  ARAP stays finite at sigma = 0; Symmetric Dirichlet blows up
  d1_curve_hencky.png          log^2 sigma turns concave past sigma = e
  d1_curves_six.png            the six nu = 0 energies, scaled to the same curvature at sigma = 1

    python figures/fig_energy_curves.py <out-dir>
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from distortion import style
from distortion.eigensystem import (ARAP, Conformal2D, Hencky, NeoHookean, StVK, SymmetricARAP, SymmetricDirichlet,
                                    WithBarrier)

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
os.makedirs(OUT, exist_ok=True)
style.use()
s = np.linspace(0.02, 3.2, 1200)[:, None]
curve = lambda e: e.psi(s)


def axes(ax, ymax):
    ax.set_xlim(0, 3.2)
    ax.set_ylim(0, ymax)
    ax.set_xlabel(r"stretch $\sigma$", fontsize=15)
    ax.axvline(1, color=style.GRID, lw=1, zorder=0)
    ax.text(1.03, ymax * 0.03, "rest", color=style.MUTED, fontsize=12, va="bottom")


# ---- ARAP vs Symmetric Dirichlet (both scaled to curvature 2 at rest)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(s, curve(ARAP()), color=style.ORANGE, lw=2.6)
ax.plot(s, curve(SymmetricDirichlet()) / 4 - 0.5, color=style.AQUA, lw=2.6)
ax.plot([0], [1], "o", color=style.ORANGE, ms=9, clip_on=False, zorder=5)
ax.annotate("ARAP: only 1 at σ = 0", (0.02, 1.0), (0.35, 2.2), color=style.ORANGE, fontsize=13,
            arrowprops=dict(arrowstyle="-", color=style.ORANGE))
ax.text(0.42, 3.4, "Symmetric Dirichlet: → ∞ at σ = 0", color=style.AQUA, fontsize=13)
axes(ax, 4)
ax.set_ylabel(r"$\hat\psi(\sigma)$", fontsize=15)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "d1_curve_arap_vs_symdir.png"))
plt.close(fig)

# ---- Hencky: concave beyond e
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ss = np.linspace(0.02, 12, 2000)
ax.axvspan(np.e, 12, color=style.ORANGE, alpha=0.12, lw=0)
ax.plot(ss, np.log(ss) ** 2, color=style.INK, lw=2.6)
ax.plot([np.e], [1], "o", color=style.ACCENT, ms=9, zorder=5)
ax.annotate("inflection at σ = e", (np.e, 1), (3.6, 0.5), color=style.ACCENT, fontsize=13,
            arrowprops=dict(arrowstyle="-", color=style.ACCENT))
ax.text(7.2, 1.2, "concave:\nstretching gets\ncheaper", color=style.ORANGE, fontsize=13, ha="center")
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.set_xlabel(r"stretch $\sigma$", fontsize=15)
ax.set_ylabel(r"$\log^2\sigma$", fontsize=15)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "d1_curve_hencky.png"))
plt.close(fig)

# ---- the six nu = 0 energies, scaled so psi''(1) = 2 and psi(1) = 0
E = [("Sym. Dirichlet", SymmetricDirichlet(), style.AQUA, "-"),
     ("Sym. ARAP", SymmetricARAP(), style.BLUE, "--"),
     ("Hencky", Hencky(), style.INK, "-"),
     ("Neo-Hookean", NeoHookean(1.0, 0.0), style.ORANGE, "-"),
     ("ARAP + barrier", WithBarrier(ARAP()), style.ACCENT, "-"),
     ("StVK + barrier", WithBarrier(StVK()), style.MUTED, "-")]
fig, ax = plt.subplots(figsize=(9.0, 4.6))
one = np.ones((1, 1))
for name, e, c, ls in E:
    k = 2 / e.hess(one)[0, 0, 0]
    y = k * (curve(e) - e.psi(one))
    ax.plot(s, y, ls, color=c, lw=2.4, label=name)
axes(ax, 3)
ax.set_ylabel(r"$\hat\psi(\sigma)$, same curvature at rest", fontsize=13)
ax.set_xlabel("← shrink               σ               stretch →", fontsize=14)
leg = ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "d1_curves_six.png"))
plt.close(fig)

# thumbnail for slides whose panel titles serve as the legend: no legend, thick lines, few labels
fig, ax = plt.subplots(figsize=(5.0, 3.0))
for name, e, c, ls in E:
    k = 2 / e.hess(one)[0, 0, 0]
    ax.plot(s, k * (curve(e) - e.psi(one)), ls, color=c, lw=3.4)
ax.set_xlim(0, 3.2); ax.set_ylim(0, 3)
ax.axvline(1, color=style.GRID, lw=1, zorder=0)
ax.set_xticks([0, 1, 2, 3]); ax.set_yticks([])
ax.tick_params(labelsize=15)
ax.set_xlabel("σ", fontsize=18, labelpad=0)
ax.set_ylabel("ψ̂", fontsize=18, rotation=0, labelpad=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "d1_curves_six_thumb.png"))
plt.close(fig)
print("ok")

# ---- signed stretch: ARAP crosses sigma = 0 into the flipped side at finite cost (d1_curve_signed_arap.png);
# Symmetric Dirichlet walls it off (d1_curve_signed_barrier.png, both curves)
from matplotlib.patches import FancyArrowPatch


def signed_curves(with_sd, name):
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.linspace(-1.2, 3.0, 2000)
    ax.axvspan(-1.2, 0, color="#f6c9c4", alpha=0.6, lw=0, zorder=0)
    ax.text(-0.86, 1.3, "flipped\n(det F < 0)", ha="center", va="center", fontsize=14, color="#c0392b")
    ax.plot(x, (x - 1) ** 2, color=style.ORANGE, lw=2.8, zorder=3)
    if with_sd:
        xs = x[np.abs(x) > 0.05]
        sd = (xs ** 2 + xs ** -2) / 4 - 0.5
        ax.plot(xs[xs > 0], sd[xs > 0], color=style.AQUA, lw=2.8, zorder=3)
        ax.plot(xs[xs < 0], sd[xs < 0], color=style.AQUA, lw=1.6, ls="--", alpha=0.6, zorder=2)   # unreachable behind the wall
        ax.text(0.28, 3.5, "Symmetric Dirichlet: → ∞ at σ = 0, a wall", color=style.AQUA, fontsize=13, va="top")
    ax.axvline(0, color="#c0392b", lw=1.2, ls="--", zorder=1)
    # a step that slides over sigma = 0 on ARAP
    ax.add_patch(FancyArrowPatch((0.47, 0.45), (-0.36, 2.05), connectionstyle="arc3,rad=-0.08", arrowstyle="-|>",
                                 mutation_scale=22, lw=2.4, color=style.ORANGE, zorder=4))
    ax.text(0.5, 1.45, "ARAP: finite at σ = 0,\na step slides into a flip", color=style.ORANGE, fontsize=13)
    ax.plot([0], [1], "o", color=style.ORANGE, ms=8, zorder=5)
    ax.axvline(1, color=style.GRID, lw=1, zorder=0)
    ax.text(1.03, 0.08, "rest", color=style.MUTED, fontsize=12, va="bottom")
    ax.set_xlim(-1.2, 3.0); ax.set_ylim(0, 3.7)
    ax.set_xlabel(r"signed stretch $\sigma$", fontsize=15); ax.set_ylabel(r"$\hat\psi(\sigma)$", fontsize=15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


signed_curves(False, "d1_curve_signed_arap.png")
signed_curves(True, "d1_curve_signed_barrier.png")


# ---- the four energies of the personalities slide, along one stretch (the other fixed at 1), each scaled to
# curvature 2 at rest like ARAP: which ones have a wall at sigma = 0
fig, ax = plt.subplots(figsize=(7.2, 4.4))
one = np.c_[s, np.ones_like(s)]
four = [("ARAP", ARAP(), style.ORANGE), ("Symmetric Dirichlet", SymmetricDirichlet(), style.AQUA),
        ("Conformal", Conformal2D(), style.BLUE), ("Neo-Hookean", NeoHookean(1.0, 1.0), "#7b4fa0")]
h = 1e-4
for name, e, c in four:
    p = lambda t: e.psi(np.c_[t, np.ones_like(t)])
    k = (p(np.array([1 + h])) - 2 * p(np.array([1.0])) + p(np.array([1 - h])))[0] / h ** 2   # curvature at rest
    y = (e.psi(one) - p(np.array([1.0]))[0]) * 2 / k
    ax.plot(s[:, 0], y, color=c, lw=2.6, label=name)
ax.plot([0], [1], "o", color=style.ORANGE, ms=8, clip_on=False, zorder=5)
axes(ax, 4)
ax.set_xlabel(r"one stretch $\sigma$ (the other fixed at 1)", fontsize=15)
ax.set_ylabel(r"$\hat\psi$", fontsize=15)
ax.legend(frameon=False, fontsize=13, loc="upper left", bbox_to_anchor=(0.37, 1.0))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "d1_curve_four.png"))
plt.close(fig)
