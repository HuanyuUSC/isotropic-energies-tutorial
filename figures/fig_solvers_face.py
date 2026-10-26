"""Solver figures on the parameterization problem: Max Planck's face, Tutte start, ARAP, free boundary.
Data: scratch/flatten_face_arap.pkl (scratch/flatten_face_arap.py).

  aux_face_problem.png          the 3D face mesh -> its flattened 2D triangles ("an optimization problem")
  solve_newton_stall_face.png   true-Hessian Newton vs projected Newton: uphill full steps, then stuck
  solve_convergence_face.png    E - E* per iteration (log-log) for GD, Newton, projected Newton

    python figures/fig_solvers_face.py <out-dir>
"""
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from distortion import style
from distortion.shade import face_colors
from distortion.planar import dets

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)
style.use()
D = pickle.load(open(os.path.join(HERE, "..", "scratch", "flatten_face_arap.pkl"), "rb"))
V3, F = D["V3"], D["F"]
RED = "#c0392b"
SHADE = face_colors(V3, F, cam=(-0.5, 0.6, -2.6), target=(0, 0.05, -0.3), ambient=0.45, diffuse=0.6)
fstar = D["pnewton"]["hist"][-1]["f"]
energies = {m: np.array([h["f"] for h in D[m]["hist"][:-1]] + [D[m]["hist"][-1]["f"]]) for m in ("gd", "newton", "pnewton")}
# newton's last hist entry repeats the stuck iterate: drop it
fn = np.array([h["f"] for h in D["newton"]["hist"][:-1]])
full = np.array([h["f_full"] for h in D["newton"]["hist"][:-1]])     # raw step x + d, no line search
stuck = len(fn) - 1                                                  # iterate where d is an ascent direction
fp = energies["pnewton"]
flips_n = int((dets(D["newton"]["X"], F) <= 0).sum())
flips_p = int((dets(D["pnewton"]["X"], F) <= 0).sum())


def upright(X):
    """Rotation is free in the plane: align the UV map to the face's frontal projection."""
    T = np.c_[-V3[:, 0], V3[:, 1]]
    T = T - T.mean(0)
    Xc = X - X.mean(0)
    U, _, Vt = np.linalg.svd(Xc.T @ T)
    return Xc @ (U @ np.diag([1, np.sign(np.linalg.det(U @ Vt))]) @ Vt)


# ---------------------------------------------------------------- the problem: 3D face -> 2D triangles
def face_problem():
    import pyvista as pv
    from matplotlib.patches import FancyArrowPatch
    faces = np.hstack([np.full((len(F), 1), 3), F]).ravel()
    m = pv.PolyData(V3, faces).compute_normals(split_vertices=False)
    p = pv.Plotter(off_screen=True, window_size=(1100, 1250))
    p.add_mesh(m, color=style.LAND, show_edges=True, edge_color=style.MUTED, line_width=0.6,
               smooth_shading=True, ambient=0.35, diffuse=0.7, specular=0.05)
    p.set_background("white")
    p.camera_position = [(-1.6, 0.25, -2.9), (0, 0.05, -0.3), (0, 1, 0)]
    img = p.screenshot(return_img=True)
    p.close()
    ys, xs = np.nonzero((img < 250).any(2))
    img = img[max(0, ys.min() - 10):ys.max() + 10, max(0, xs.min() - 10):xs.max() + 10]

    X = upright(D["pnewton"]["X"])
    fig = plt.figure(figsize=(14, 6.4))
    axL = fig.add_axes([0.0, 0.02, 0.40, 0.84])
    axR = fig.add_axes([0.54, 0.06, 0.44, 0.76])
    axL.imshow(img); axL.axis("off")
    axR.add_collection(PolyCollection(X[F], fc=SHADE, ec=SHADE * 0.85, lw=0.12))
    axR.autoscale_view(); axR.set_aspect("equal"); axR.axis("off")
    fig.text(0.20, 0.92, "3D triangles (fixed)", ha="center", fontsize=24, color=style.INK2)
    fig.text(0.76, 0.92, "2D triangles: the unknowns $x$", ha="center", fontsize=24, color=style.INK2)
    fig.patches.append(FancyArrowPatch((0.405, 0.47), (0.525, 0.47), transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=34, lw=4, color=style.ACCENT))
    fig.text(0.465, 0.55, "flatten", ha="center", fontsize=22, color=style.ACCENT)
    fig.text(0.465, 0.33, r"$\min_x \sum_t A_t\,\psi(F_t)$", ha="center", fontsize=22, color=style.INK)
    fig.savefig(os.path.join(OUT, "aux_face_problem.png"))
    plt.close(fig)


# ---------------------------------------------------------------- Newton (true Hessian) vs projected Newton
def newton_stall():
    fig, ax = plt.subplots(figsize=(11, 6.4))
    k = np.arange(len(fn))
    ax.semilogy(np.arange(len(fp)), fp, "o-", color=style.AQUA, lw=3.2, ms=8, label="projected Newton", zorder=4)
    ax.semilogy(k, fn, "o-", color=RED, lw=3.2, ms=8, label="Newton (true Hessian)", zorder=5)
    # the raw Newton step (a = 1) from each iterate: uphill where it lands above the current energy
    up = full > fn
    for i in np.nonzero(up)[0]:
        ax.plot([i, i + 1], [fn[i], full[i]], ls=":", color=RED, lw=2.2, zorder=3)
        ax.plot(i + 1, full[i], "^", mfc="white", mec=RED, mew=2.4, ms=14, zorder=6)
    ax.text(1.25, full[0], f"full Newton step goes uphill:\n{fn[0]:.2f} → {full[0]:.2f}", color=RED,
            fontsize=18, va="center")
    ax.annotate("stuck: Newton direction\npoints uphill", xy=(stuck, fn[stuck]), xytext=(stuck + 3, 0.06),
                color=RED, fontsize=18, arrowprops=dict(arrowstyle="-|>", color=RED, lw=2), va="center")
    ax.axhline(fn[-1], color=RED, lw=1.2, ls=":", zorder=1)
    ax.text(len(fp) - 1, fn[-1] * 1.06, f"Newton stuck at {fn[-1]:.4f}  ({flips_n} flipped triangles)",
            color=RED, fontsize=17, ha="right", va="bottom")
    ax.text(len(fp) - 1, fp[-1] * 0.94, f"projected Newton: {fp[-1]:.4f}  ({flips_p} flipped)",
            color=style.AQUA, fontsize=17, ha="right", va="top")
    ax.set_ylim(0.0095, 3.0)
    ax.set_xlim(-0.5, len(fp) - 0.5)
    ax.set_xlabel("iteration", fontsize=19)
    ax.set_ylabel("ARAP energy / area", fontsize=19)
    ax.tick_params(labelsize=16)
    ax.legend(frameon=False, fontsize=18, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "solve_newton_stall_face.png"))
    plt.close(fig)


# ---------------------------------------------------------------- E - E* for all three
def convergence():
    fig, ax = plt.subplots(figsize=(11, 6.4))
    fg = energies["gd"]
    floor = 1e-10
    ax.loglog(np.arange(1, len(fg) + 1), np.maximum(fg - fstar, floor), color=style.MUTED, lw=3.2, label="gradient descent")
    ax.loglog(np.arange(1, len(fn) + 1), np.maximum(fn - fstar, floor), "o-", color=RED, lw=3.2, ms=7, label="Newton (true Hessian)")
    ax.plot(len(fn), fn[-1] - fstar, "X", color=RED, ms=18)
    ax.loglog(np.arange(1, len(fp) + 1), np.maximum(fp - fstar, floor), "o-", color=style.AQUA, lw=3.2, ms=7, label="projected Newton")
    ax.text(len(fn) * 1.3, (fn[-1] - fstar) * 0.55, "stuck", color=RED, fontsize=18, va="top")
    ax.text(len(fg), (fg[-1] - fstar) * 0.35, f"{len(fg) - 1} iterations,\n{D['gd']['t'][-1]:.0f} s: not there yet",
            color=style.INK2, fontsize=17, ha="right", va="top")
    ax.text(len(fp) * 1.15, 3e-9, f"{len(fp) - 1} iterations, {D['pnewton']['t'][-1]:.1f} s", color=style.AQUA,
            fontsize=17, va="center")
    ax.set_ylim(floor, 1)
    ax.set_xlabel("iteration", fontsize=19)
    ax.set_ylabel(r"$E - E^*$  (per area)", fontsize=19)
    ax.tick_params(labelsize=16)
    ax.legend(frameon=False, fontsize=18, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "solve_convergence_face.png"))
    plt.close(fig)


face_problem()
newton_stall()
convergence()
print("E* =", fstar, " newton stuck at", fn[-1], "iterate", stuck, " flips newton/pnewton", flips_n, flips_p)
