"""Act-2 foundations on the parameterization mesh (Max Planck's face, flattened by ARAP + projected Newton):

  aux_ds_dm_face.png    where F comes from: one 3D face triangle, laid flat in its own plane (D_m), and its
                        2D image in the UV map (D_s); F = D_s D_m^{-1}
  aux_face_cloud.png    the flattened face -> every triangle a point on the sigma-plane

Data: scratch/flatten_face_arap.pkl (scratch/flatten_face_arap.py).
    python figures/fig_face_ds_dm.py <out-dir>
"""
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import FancyArrowPatch

from distortion import style
from distortion.shade import face_colors
from distortion.core import rest_frames

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
HERE = os.path.dirname(os.path.abspath(__file__))
style.use()
D = pickle.load(open(os.path.join(HERE, "..", "scratch", "flatten_face_arap.pkl"), "rb"))
V3, F = D["V3"], D["F"]
RED = "#c0392b"
SHADE = face_colors(V3, F, cam=(-0.5, 0.6, -2.6), target=(0, 0.05, -0.3), ambient=0.45, diffuse=0.6)


def upright(X):
    T = np.c_[-V3[:, 0], V3[:, 1]]
    T = T - T.mean(0)
    Xc = X - X.mean(0)
    U, _, Vt = np.linalg.svd(Xc.T @ T)
    return Xc @ (U @ np.diag([1, np.sign(np.linalg.det(U @ Vt))]) @ Vt)


X = upright(D["pnewton"]["X"])
Dm = rest_frames(V3, F)                                          # (t, 2, 2): each 3D triangle in its own plane
Ds = np.stack([X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]], 2)
Fm = Ds @ np.linalg.inv(Dm)
sig = np.linalg.svd(Fm, compute_uv=False)
area = 0.5 * np.abs(np.linalg.det(Dm))


def arrow(ax, a, b, color=None, lw=2.2, ms=18):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color or style.INK2,
                                 shrinkA=0, shrinkB=0))


def render_face(t, size=(900, 1000)):
    """The 3D face with triangle t (and a few neighbours, so it shows) in red; returns image + its pixel position."""
    import pyvista as pv
    faces = np.hstack([np.full((len(F), 1), 3), F]).ravel()
    m = pv.PolyData(V3, faces).compute_normals(split_vertices=False)
    p = pv.Plotter(off_screen=True, window_size=size)
    p.add_mesh(m, color=style.LAND, show_edges=True, edge_color=style.MUTED, line_width=0.4,
               smooth_shading=True, ambient=0.35, diffuse=0.7, specular=0.05)
    tri = pv.PolyData(V3, np.r_[3, F[t]])
    p.add_mesh(tri, color=RED, show_edges=True, edge_color=RED, line_width=3)
    p.set_background("white")
    p.camera_position = [(-1.6, 0.25, -2.9), (0, 0.05, -0.3), (0, 1, 0)]
    img = p.screenshot(return_img=True)
    r = p.renderer
    r.SetWorldPoint(*V3[F[t]].mean(0), 1.0)
    r.WorldToDisplay()
    px, py, _ = r.GetDisplayPoint()
    p.close()
    H = img.shape[0]
    py = H - py
    ys, xs = np.nonzero((img < 250).any(2))
    y0, x0 = max(0, ys.min() - 10), max(0, xs.min() - 10)
    img = img[y0:ys.max() + 10, x0:xs.max() + 10]
    return img, (px - x0, py - y0)


def ds_dm_face():
    # a triangle on the cheek, clearly visible from the camera
    # a typical triangle (stretches near the median, well shaped) that faces the camera
    c = V3[F].mean(1)
    n = np.cross(V3[F[:, 1]] - V3[F[:, 0]], V3[F[:, 2]] - V3[F[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    cam = np.array([-1.6, 0.25, -2.9])
    facing = np.abs(np.einsum("ij,ij->i", n, (cam - c) / np.linalg.norm(cam - c, axis=1, keepdims=True)))
    e = np.linalg.norm(V3[F[:, [1, 2, 0]]] - V3[F], axis=2)
    quality = 4 * np.sqrt(3) * area / (e ** 2).sum(1)                # 1 = equilateral
    med = np.median(sig, 0)
    score = np.linalg.norm(sig - med, axis=1) * 4 + (1 - facing) + (1 - quality)
    score[np.abs(c[:, 1] - c[:, 1].mean()) > 0.15] = np.inf          # mid-height (cheek level)
    t = int(np.argmin(score))
    print(f"triangle {t}: sigma {sig[t].round(3)}, facing {facing[t]:.2f}, quality {quality[t]:.2f}")
    img, (px, py) = render_face(t)
    fig = plt.figure(figsize=(15, 5.4))
    A = [fig.add_axes([0.0, 0.04, 0.2, 0.84]), fig.add_axes([0.21, 0.0, 0.58, 0.92]),
         fig.add_axes([0.8, 0.04, 0.2, 0.84])]
    A[0].imshow(img); A[0].plot([px], [py], "o", ms=26, mfc="none", mec=RED, mew=2.5); A[0].axis("off")
    A[0].set_title("rest: 3D surface", fontsize=18, color=style.INK2)
    ax = A[2]
    ax.add_collection(PolyCollection(X[F], fc=SHADE, ec=SHADE * 0.85, lw=0.12))
    ax.add_collection(PolyCollection(X[F[t:t + 1]], fc=RED, ec=RED, lw=1.5, zorder=3))
    cc = X[F[t]].mean(0)
    ax.plot([cc[0]], [cc[1]], "o", ms=26, mfc="none", mec=RED, mew=2.5, zorder=4)
    ax.autoscale_view(); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("deformed: 2D UV map", fontsize=18, color=style.INK2)
    # middle: rest triangle laid flat in its own plane, reference triangle, 2D triangle
    ax = A[1]
    rest = np.array([[0, 0], Dm[t][:, 0], Dm[t][:, 1]])
    dfm = X[F[t]] - X[F[t]][0]
    rest, dfm = rest - rest.mean(0), dfm - dfm.mean(0)
    sc = 1.0 / np.abs(np.r_[rest, dfm]).max()
    tris = [(rest * sc * 1.2 + [-2.6, 0.75], r"$X_0$", r"$X_1$", r"$X_2$", "3D triangle, laid flat\nin its own plane: $D_m$"),
            (np.array([[0, 0], [1, 0], [0, 1]]) * 1.0 + [-0.5, -2.1], "", "", "", "reference triangle"),
            (dfm * sc * 1.2 + [2.6, 0.75], r"$x_0$", r"$x_1$", r"$x_2$", "its 2D image\n(the unknowns): $D_s$")]
    for P, l0, l1, l2, cap in tris:
        ax.add_patch(plt.Polygon(P, fc="#f6c9c4", ec=RED, lw=1.5, zorder=1))
        arrow(ax, tuple(P[0]), tuple(P[1]), color=style.BLUE, lw=3, ms=20)
        arrow(ax, tuple(P[0]), tuple(P[2]), color=style.ORANGE, lw=3, ms=20)
        cP = P.mean(0)
        for q, l in zip(P, (l0, l1, l2)):
            if l:
                d = (q - cP) / np.linalg.norm(q - cP)
                ax.text(*(q + 0.28 * d), l, fontsize=17, ha="center", va="center", color=style.INK)
        ax.text(cP[0], P[:, 1].min() - 0.35, cap, fontsize=15, ha="center", va="top", color=style.INK2)
    arrow(ax, (-0.7, -1.45), (-2.2, -0.75), lw=2.2, ms=20)
    ax.text(-1.55, -1.55, r"$D_m$", fontsize=24, color=style.INK2, ha="right", va="top")
    arrow(ax, (0.7, -1.45), (2.2, -0.75), lw=2.2, ms=20)
    ax.text(1.55, -1.55, r"$D_s$", fontsize=24, color=style.INK2, ha="left", va="top")
    ax.add_patch(FancyArrowPatch((-1.9, 1.55), (1.9, 1.55), connectionstyle="arc3,rad=-0.25", arrowstyle="-|>",
                                 mutation_scale=24, lw=3, color=style.ACCENT))
    ax.text(0, 2.15, r"$F = D_s\,D_m^{-1}$", fontsize=26, color=style.ACCENT, ha="center")
    ax.set_xlim(-3.9, 3.9); ax.set_ylim(-2.7, 2.55); ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(os.path.join(OUT, "aux_ds_dm_face.png"))
    plt.close(fig)


def face_cloud():
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw=dict(width_ratios=[1, 1.05]))
    ax = axs[0]
    ax.add_collection(PolyCollection(X[F], fc=SHADE, ec=SHADE * 0.85, lw=0.12))
    ax.autoscale_view(); ax.set_aspect("equal"); ax.axis("off")
    ax = axs[1]
    lim = 2.0
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.set_xlabel(r"$\sigma_1$", fontsize=16); ax.set_ylabel(r"$\sigma_2$", fontsize=16)
    s = np.linspace(1 / lim, lim, 300)
    ax.plot(s, s, ":", color=style.MUTED, lw=1); ax.plot(s, 1 / s, ":", color=style.MUTED, lw=1)
    ax.plot([1], [1], "o", color=style.ACCENT, ms=9, zorder=6)
    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
    style.mirror_half(ax, 0, lim, fs=11)
    ax.scatter(sig[:, 0], sig[:, 1], s=6, color=style.BLUE, alpha=0.45, lw=0, zorder=4)
    fig.text(0.455, 0.5, "→", fontsize=40, ha="center", va="center", color=style.INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "aux_face_cloud.png"))
    plt.close(fig)
    print(f"face: {len(F)} triangles, sigma1 {sig[:, 0].min():.2f}..{sig[:, 0].max():.2f}, "
          f"sigma2 {sig[:, 1].min():.2f}..{sig[:, 1].max():.2f}")


ds_dm_face()
face_cloud()
print("ok")
