"""Act 3 solver figures from the saved gingerbread runs (scratch/solvers2d.pkl, scratch/newton_fail.pkl).

  solve_gd_snaps.png      gradient descent after 10 / 100 / 1000 iterations vs. projected Newton after 10
  solve_newton_stall.png  head twist 90 deg, Symmetric Dirichlet: true-Hessian Newton stalls, projected Newton converges
  solve_convergence.png   arm raise, Symmetric Dirichlet: E - E* per iteration for GD, Newton, projected Newton

    python figures/fig_solvers.py <out-dir>
"""
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from distortion import style
from distortion.planar import gingerbread

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)
style.use()
S = pickle.load(open(os.path.join(HERE, "..", "scratch", "solvers2d.pkl"), "rb"))
NF = pickle.load(open(os.path.join(HERE, "..", "scratch", "newton_fail.pkl"), "rb"))
V, F, G = gingerbread()
RED = "#c0392b"


def mesh(ax, X, title, color=style.INK2):
    ax.add_collection(PolyCollection(X[F], fc=style.LAND, ec=style.MUTED, lw=0.2))
    ax.plot(*X[G["feet"]].T, "s", color=style.INK, ms=2.5)
    ax.set_xlim(-2.6, 2.4)
    ax.set_ylim(-2.3, 2.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=17, color=color)


# ---- GD snapshots vs projected Newton
gd, pn = S[("symdir", "gd")], S[("symdir", "pnewton")]
fig, axs = plt.subplots(1, 4, figsize=(15, 4.6))
for ax, (X, t, c) in zip(axs, [(gd["snaps"][10], "GD: 10 iterations", style.INK2),
                               (gd["snaps"][100], "GD: 100", style.INK2),
                               (gd["snaps"][1000], "GD: 1000", style.INK2),
                               (pn["snaps"][10], "projected Newton: 10", style.AQUA)]):
    mesh(ax, X, t, c)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "solve_gd_snaps.png"))
plt.close(fig)

# ---- convergence: arm raise (GD vs Newton) and head twist (Newton stalls)
fstar = min(h["f"] for k in [("symdir", "newton"), ("symdir", "pnewton")] for h in S[k]["hist"])
nt, pt = NF[("twist90", "symdir", "newton")], NF[("twist90", "symdir", "pnewton")]
fn, fp = [np.array([h["f"] for h in r["hist"]]) for r in (nt, pt)]
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axs[0]
for key, c, lab in [(("symdir", "gd"), style.MUTED, "gradient descent"), (("symdir", "newton"), style.ORANGE, "Newton"),
                    (("symdir", "pnewton"), style.AQUA, "projected Newton")]:
    f = np.array([h["f"] for h in S[key]["hist"]]) - fstar
    ax.loglog(np.arange(1, len(f) + 1), np.maximum(f, 1e-9), color=c, lw=2.6, label=lab)
ax.set_title("arm raise", fontsize=15, color=style.INK2)
ax.set_ylim(1e-9, 1e5)
ax.set_xlabel("iteration", fontsize=14)
ax.set_ylabel(r"$E - E^*$", fontsize=14)
ax.legend(frameon=False, fontsize=13, loc="center right")
ax = axs[1]
ft = fp[-1]
ax.semilogy(np.arange(len(fn)), np.maximum(fn - ft, 1e-9), "o-", color=RED, lw=2.4, ms=5, label="Newton (true Hessian)")
ax.semilogy(np.arange(len(fp)), np.maximum(fp - ft, 1e-9), "o-", color=style.AQUA, lw=2.4, ms=5, label="projected Newton")
ax.set_title("head twist 90°", fontsize=15, color=style.INK2)
ax.set_ylim(1e-9, 1e5)
ax.set_xlabel("iteration", fontsize=14)
ax.legend(frameon=False, fontsize=13, loc="center right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "solve_convergence.png"))
plt.close(fig)

# ---- Newton stalls on the head twist (linear zoom on the plateaus)
fig, ax = plt.subplots(figsize=(7.6, 4.6))
ax.plot(np.arange(len(fn)), fn, "o-", color=RED, lw=2.6, ms=6, label="Newton (true Hessian)")
ax.plot(np.arange(len(fp)), fp, "o-", color=style.AQUA, lw=2.6, ms=6, label="projected Newton")
ax.axhline(fn[-1], color=RED, lw=1, ls=":")
ax.text(len(fp) - 1, fn[-1] + 0.12, f"Newton stalls: {fn[-1]:.2f}", color=RED, fontsize=14, ha="right")
ax.text(len(fp) - 1, fp[-1] - 0.35, f"projected Newton: {fp[-1]:.2f}", color=style.AQUA, fontsize=14, ha="right")
ax.set_ylim(25.0, 29.0)
ax.text(10.5, 28.0, f"(both start at E = {fn[0]:.0f})", fontsize=12, color=style.MUTED, va="top")
ax.set_xlabel("iteration", fontsize=14)
ax.set_ylabel("energy E", fontsize=14)
ax.legend(frameon=False, fontsize=13, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "solve_newton_stall.png"))
plt.close(fig)
print("ok", fstar, fn[-1], fp[-1])
