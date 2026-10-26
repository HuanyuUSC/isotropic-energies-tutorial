"""Slide videos (mp4, H.264) and their poster frames (png, used by the PDF export).

  vid_modes        the four Hessian modes as oscillating perturbations of a stretched disk
  vid_svd          rotate -> stretch -> rotate, continuously
  vid_twist        head twisted 0 -> 720 deg: ARAP (flips in red) vs. Symmetric Dirichlet
  vid_drag         arm raised step by step: |F - I|^2 vs. ARAP
  vid_solvers      gradient descent vs. projected Newton, iteration by iteration
  vid_flatten_arap the face flattened with ARAP by projected Newton (Tutte start)
  vid_solvers_face GD vs projected Newton on the face (ARAP), equal wall clock
  vid_skull        the skull's Earth map during optimization (tetrahedron start -> SLIM -> Newton)

    python figures/fig_videos.py <out-dir> [name ...]
"""
import io
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from PIL import Image

from distortion import style

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
ONLY = set(sys.argv[2:])
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)
style.use()
matplotlib.rcParams["savefig.bbox"] = None          # videos need a fixed frame size
RED = "#d6402f"
VIDEOS = []


def video(f):
    VIDEOS.append(f)
    return f


class Writer:
    """Collect RGB frames (numpy or matplotlib figures) into an mp4 + a poster png."""

    def __init__(self, name, fps=30, poster=None):
        self.name, self.fps, self.poster, self.w, self.n = name, fps, poster, None, 0

    def add(self, img):
        if isinstance(img, plt.Figure):
            buf = io.BytesIO()
            img.savefig(buf, format="png", dpi=100, facecolor="white", bbox_inches=None)   # fixed frame size
            img = np.asarray(Image.open(buf).convert("RGB"))
        if self.w is None:
            H, W = img.shape[:2]
            self.H, self.W = H - H % 2, W - W % 2
            self.w = imageio_ffmpeg.write_frames(os.path.join(OUT, self.name + ".mp4"), (self.W, self.H),
                                                 fps=self.fps, quality=8, macro_block_size=1)
            self.w.send(None)
        img = np.ascontiguousarray(img[:self.H, :self.W])
        self.w.send(img)
        if self.poster is not None and self.n == self.poster:
            Image.fromarray(img).save(os.path.join(OUT, self.name + "_poster.png"))
        self.n += 1

    def close(self):
        self.w.close()
        if self.poster is None or self.poster >= self.n:
            pass
        print(self.name, self.n, "frames")


def arrow(ax, a, b, color, lw=2.2, ms=18):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color, shrinkA=0, shrinkB=0))


def checker(ax, M, n=8, res=120, zorder=1):
    """The unit disk with an n x n material checkerboard, mapped by the linear map M."""
    from matplotlib.colors import ListedColormap
    g = np.linspace(-1, 1, res + 1)
    X, Y = np.meshgrid(g, g)
    P = np.einsum("ij,jkl->ikl", M, np.stack([X, Y]))
    xc, yc = 0.5 * (g[1:] + g[:-1]), 0.5 * (g[1:] + g[:-1])
    XC, YC = np.meshgrid(xc, yc)
    C = ((np.floor((XC + 1) / 2 * n) + np.floor((YC + 1) / 2 * n)) % 2).astype(float)
    C = np.ma.masked_where(XC ** 2 + YC ** 2 > 1, C)
    ax.pcolormesh(P[0], P[1], C, cmap=ListedColormap(["#ecebe7", "#bdbcb6"]), shading="flat", zorder=zorder,
                  rasterized=True, antialiased=False)


def disk(ax, M, color=style.INK2, fill=style.LAND, lw=1.6, axes=None, labels=None, texture=False, heads=False):
    """Image of the unit disk under M, with material lines (a polar grid) so rotations show."""
    t = np.linspace(0, 2 * np.pi, 200)
    C = M @ np.stack([np.cos(t), np.sin(t)])
    if texture:
        checker(ax, M)
    else:
        ax.fill(C[0], C[1], color=fill, zorder=1)
    segs = []
    for r in [0.33, 0.66]:
        P = M @ (r * np.stack([np.cos(t), np.sin(t)]))
        segs.append(P.T)
    for a in np.linspace(0, np.pi, 6, endpoint=False):
        d = np.array([np.cos(a), np.sin(a)])
        segs.append((M @ np.stack([-d, d]).T).T)
    if not texture:
        ax.add_collection(LineCollection(segs, colors=style.MUTED, lw=0.8, zorder=2))
    ax.plot(C[0], C[1], color=color, lw=lw, zorder=3)
    # two marked material axes (v1 blue, v2 orange)
    A = np.eye(2) if axes is None else axes                 # columns: the two material directions to mark
    for k, (v, c) in enumerate([(A[:, 0], style.BLUE), (A[:, 1], style.ORANGE)]):
        a, b = M @ v, -(M @ v)
        if heads:                                            # arrowhead on the + end shows handedness
            ax.plot([b[0], 0], [b[1], 0], color=c, lw=3, zorder=4, solid_capstyle="round")
            arrow(ax, (0, 0), tuple(a), c, lw=3, ms=20)
        else:
            ax.plot([b[0], a[0]], [b[1], a[1]], color=c, lw=3, zorder=4, solid_capstyle="round")
        if labels:
            n = np.linalg.norm(a)
            ax.text(*(a * (1 + 0.3 / max(n, 0.3))), labels[k], color=c, fontsize=20, ha="center", va="center", zorder=5)


# ------------------------------------------------------------------ the four Hessian modes
@video
def vid_modes():
    """Three shapes: the reference disk (axes v_i = columns of V), the deformed shape F (dashed, axes
    sigma_i u_i = columns of U), and F after each Hessian mode, flowed exactly (solid, material grid):
    scale sigma_i -> sigma_i e^t; twist F = R Sigma R (U, V turn opposite ways); flip F = R Sigma R^T."""
    rot = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    s0 = np.array([1.45, 0.75])
    # same amplitude in F-space for every mode: |dF/dt| = sigma_i for scaling, (s1 +- s2)/sqrt2 per radian
    kt, kf = 1.0 / ((s0[0] + s0[1]) / np.sqrt(2)), 1.0 / ((s0[0] - s0[1]) / np.sqrt(2))
    flows = [(r"$D_1$: scale $\sigma_1$", lambda e: (np.eye(2), np.diag(s0 * [np.exp(e / s0[0]), 1]), np.eye(2))),
             (r"$D_2$: scale $\sigma_2$", lambda e: (np.eye(2), np.diag(s0 * [1, np.exp(e / s0[1])]), np.eye(2))),
             (r"$T_{12}$: twist", lambda e: (rot(e * kt), np.diag(s0), rot(-e * kt))),
             (r"$L_{12}$: flip", lambda e: (rot(e * kf), np.diag(s0), rot(e * kf)))]
    th = np.linspace(0, 2 * np.pi, 200)
    circ = np.stack([np.cos(th), np.sin(th)])
    fig = plt.figure(figsize=(16, 4.6))
    # fixed layout, computed once: reference panel, then four mode panels
    ax_ref = fig.add_axes([0.005, 0.14, 0.15, 0.72])
    axs = [fig.add_axes([0.20 + k * 0.193, 0.20, 0.188, 0.66]) for k in range(4)]
    insets = {t: fig.add_axes([0.20 + k * 0.193 + 0.06, 0.02, 0.065, 0.2]) for k, (t, _) in enumerate(flows)}
    fig.text(0.172, 0.50, "F", fontsize=24, ha="center", color=style.INK2)
    from matplotlib.patches import FancyArrowPatch
    fig.patches.append(FancyArrowPatch((0.152, 0.45), (0.195, 0.45), transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=24, lw=2.4, color=style.INK2))
    fig.text(0.5, 0.955, "left: reference, axes vᵢ (columns of V)   ·   dashed: deformed by F, axes σᵢuᵢ (columns of U)"
             "   ·   solid: F after the mode (checkerboard: the material)", ha="center", fontsize=14, color=style.INK2)
    # reference: static
    checker(ax_ref, np.eye(2))
    ax_ref.plot(circ[0], circ[1], color=style.INK2, lw=1.6)
    for k, (c, lab) in enumerate([(style.BLUE, r"$v_1$"), (style.ORANGE, r"$v_2$")]):
        e = np.eye(2)[:, k]
        arrow(ax_ref, (0, 0), tuple(e), color=c, lw=2.6)
        ax_ref.text(*(e * 1.3), lab, color=c, fontsize=20, ha="center", va="center")
    ax_ref.set_xlim(-1.5, 1.5); ax_ref.set_ylim(-1.5, 1.5); ax_ref.set_aspect("equal"); ax_ref.axis("off")
    ax_ref.set_title("reference", fontsize=20, color=style.INK2)
    W = Writer("vid_modes", fps=30, poster=22)
    for k in range(120):
        e = 0.36 * np.sin(2 * np.pi * k / 60)
        for ax, (t, f) in zip(axs, flows):
            U, S, V = f(e)
            ax.clear()
            # the deformed shape F (dashed) and its axes sigma_i u_i
            P = np.diag(s0) @ circ
            ax.plot(P[0], P[1], color=style.MUTED, lw=1.6, ls="--", zorder=6)
            for kk, (c, lab) in enumerate([(style.BLUE, r"$\sigma_1 u_1$"), (style.ORANGE, r"$\sigma_2 u_2$")]):
                a = s0[kk] * np.eye(2)[:, kk]
                ax.plot([0, a[0] * 1.55 if kk == 0 else 0], [0, 0 if kk == 0 else a[1] * 1.9], color=c, lw=1.4,
                        ls=(0, (3, 2)), zorder=0, alpha=0.7)          # rest axes, dashed, extended past the shape
                pos = (s0[0] * 1.55 + 0.05, 0.12) if kk == 0 else (0.0, s0[1] * 1.9 + 0.2)
                ax.text(*pos, lab, color=c, fontsize=17, ha="left" if kk == 0 else "center", va="center", zorder=7)
            # F after the mode (solid, material grid); colored lines: its current axes
            disk(ax, U @ S @ V.T, axes=V, texture=True)
            ax.set_xlim(-2.4, 3.0); ax.set_ylim(-1.9, 1.9); ax.set_aspect("equal"); ax.axis("off")
            ax.set_title(t, fontsize=20, color=style.INK2)
            ins = insets[t]
            ins.clear()
            checker(ins, np.eye(2), res=60); ins.plot(circ[0], circ[1], color=style.INK2, lw=1)
            for kk, c in enumerate([style.BLUE, style.ORANGE]):
                v = V[:, kk]
                ins.plot([-v[0], v[0]], [-v[1], v[1]], color=c, lw=2.2)
            ins.text(0, -1.75, "reference: vᵢ(t)", ha="center", fontsize=11, color=style.INK2)
            ins.set_xlim(-1.3, 1.3); ins.set_ylim(-2.1, 1.3); ins.set_aspect("equal"); ins.axis("off")
        W.add(fig)
    plt.close(fig)
    W.close()


# ------------------------------------------------------------------ SVD as a continuous motion
@video
def vid_svd():
    rot = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    a_v, s, a_u = np.radians(-25), np.array([1.8, 0.6]), np.radians(35)
    W = Writer("vid_svd", fps=30, poster=179)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    stages = [("rotate: Vᵀ", 45), ("stretch: Σ", 45), ("rotate: U", 45), ("F = U Σ Vᵀ", 45)]
    k = 0
    for si, (lab, n) in enumerate(stages):
        for j in range(n):
            t = min(1.0, j / (n * 0.75))
            t = 0.5 - 0.5 * np.cos(np.pi * t)
            if si == 0:
                M = rot(-a_v * t)                    # V^T = rot(-a_v)
            elif si == 1:
                M = np.diag(1 + t * (s - 1)) @ rot(-a_v)
            elif si == 2:
                M = rot(a_u * t) @ np.diag(s) @ rot(-a_v)
            else:
                M = rot(a_u) @ np.diag(s) @ rot(-a_v)
            ax.clear()
            # what the two colored lines are: v_i -> (V^T) e_i -> (Sigma) sigma_i e_i -> (U) sigma_i u_i
            done = t > 0.5                       # relabel halfway through each stage
            labs = [((r"$e_1$", r"$e_2$") if done else (r"$v_1$", r"$v_2$")),
                    ((r"$\sigma_1 e_1$", r"$\sigma_2 e_2$") if done else (r"$e_1$", r"$e_2$")),
                    ((r"$\sigma_1 u_1$", r"$\sigma_2 u_2$") if done else (r"$\sigma_1 e_1$", r"$\sigma_2 e_2$")),
                    (r"$\sigma_1 u_1$", r"$\sigma_2 u_2$")][si]
            disk(ax, M, axes=rot(a_v), labels=labs, heads=True)   # mark v1, v2: V^T sends them to the coordinate axes
            ax.set_xlim(-2.5, 2.5); ax.set_ylim(-1.8, 1.8); ax.set_aspect("equal"); ax.axis("off")
            ax.set_title(lab, fontsize=22, color=style.ACCENT if si < 3 else style.INK)
            W.add(fig)
            k += 1
    plt.close(fig)
    W.close()


# ------------------------------------------------------------------ gingerbread helpers
def gb():
    from distortion.planar import gingerbread
    return gingerbread()


def draw_gb(ax, X, F, pins, title=None, lim=((-2.5, 2.4), (-2.3, 2.9)), color=None):
    from distortion.planar import dets
    bad = dets(X, F) <= 0
    ax.add_collection(PolyCollection(X[F[~bad]], fc=style.LAND, ec=style.MUTED, lw=0.2))
    if bad.any():
        ax.add_collection(PolyCollection(X[F[bad]], fc=RED, ec=RED, lw=0.6, zorder=3))
    for p in pins:
        ax.plot(*X[p].T, "s", color=style.INK, ms=2.5)
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=17, color=color or style.INK2)
    return int(bad.sum())


@video
def vid_twist():
    """Head twisted 0 -> 180 deg: ARAP (flips), Symmetric Dirichlet (no flips, but the boundary crosses
    itself), Symmetric Dirichlet + boundary barrier [Smith & Schaefer 2015] (globally injective).
    Data: scratch/twist180.pkl (scratch/twist_inj.py). Boundary drawn thick so self-overlap is visible."""
    import igl
    from distortion.boundary_barrier import boundary_crossings
    V, F, G = gb()
    T = pickle.load(open(os.path.join(HERE, "..", "scratch", "twist180.pkl"), "rb"))
    loop = igl.boundary_loop(np.ascontiguousarray(F, dtype=np.int64))
    loop = np.r_[loop, loop[0]]
    lim = ((-1.6, 1.6), (0.25, 3.0))
    panels = [("arap", "ARAP"), ("symdir", "Symmetric Dirichlet"), ("inj", "Sym. Dirichlet + boundary barrier")]
    W = Writer("vid_twist", fps=12, poster=48)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.8))
    n = len(T["angles"])
    for i in list(range(n)) + [n - 1] * 24:
        for ax, (key, name) in zip(axs, panels):
            ax.clear()
            X = T[key][i]
            k = draw_gb(ax, X, F, [], lim=lim)
            ax.plot(X[loop, 0], X[loop, 1], color=style.INK, lw=2.4, zorder=5)
            c = boundary_crossings(X, F)
            ok = k == 0 and c == 0
            ax.set_title(f"{name}\n{k} flipped · {c} boundary crossing{'' if c == 1 else 's'}", fontsize=15, color=style.AQUA if ok else RED)
        fig.suptitle(f"head turned {T['angles'][i]:.0f}°", fontsize=18, color=style.INK2)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.02, wspace=0.04)
        W.add(fig)
    plt.close(fig)
    W.close()


@video
def vid_twist_arap():
    """The twist, ARAP only (the left panel of vid_twist), for the slide that poses the problem."""
    import igl
    from distortion.boundary_barrier import boundary_crossings
    V, F, G = gb()
    T = pickle.load(open(os.path.join(HERE, "..", "scratch", "twist180.pkl"), "rb"))
    loop = igl.boundary_loop(np.ascontiguousarray(F, dtype=np.int64))
    loop = np.r_[loop, loop[0]]
    lim = ((-1.6, 1.6), (0.25, 3.0))
    W = Writer("vid_twist_arap", fps=12, poster=48)
    fig, ax = plt.subplots(figsize=(6.0, 5.8))
    n = len(T["angles"])
    for i in list(range(n)) + [n - 1] * 24:
        ax.clear()
        X = T["arap"][i]
        k = draw_gb(ax, X, F, [], lim=lim)
        ax.plot(X[loop, 0], X[loop, 1], color=style.INK, lw=2.4, zorder=5)
        c = boundary_crossings(X, F)
        ax.set_title(f"ARAP, head turned {T['angles'][i]:.0f}°\n{k} flipped · {c} boundary crossing{'' if c == 1 else 's'}",
                     fontsize=16, color=RED if (k or c) else style.INK2)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.02)
        W.add(fig)
    plt.close(fig)
    W.close()


@video
def vid_flatten():
    """Max Planck's face flattened: Tutte (circle boundary) -> Symmetric Dirichlet, one Newton iterate
    per beat. Left: the 3D face with a checker drawn through the current UV map. Right: the UV plane,
    triangles shaded by sigma1/sigma2. Data: scratch/flatten_face.pkl (scratch/flatten_face.py)."""
    import pyvista as pv
    from matplotlib.colors import LogNorm
    D = pickle.load(open(os.path.join(HERE, "..", "scratch", "flatten_face.pkl"), "rb"))
    V3, F, frames, en = D["V3"], D["F"], D["frames"], D["energy"]
    e1, e2 = V3[F[:, 1]] - V3[F[:, 0]], V3[F[:, 2]] - V3[F[:, 0]]
    area3 = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1).sum()
    ii, jj = np.meshgrid(np.arange(1024), np.arange(1024))
    chk = np.where((((ii // 64) + (jj // 64)) % 2).astype(bool)[..., None], [42, 120, 214], [236, 235, 231]).astype(np.uint8)
    tex = pv.Texture(chk); tex.repeat = True; tex.interpolate = True
    faces = np.hstack([np.full((len(F), 1), 3), F]).ravel()

    def face3d(X):
        uv = X - X.mean(0)
        uv *= np.sqrt(area3 / (0.5 * np.abs(np.cross(uv[F[:, 1]] - uv[F[:, 0]], uv[F[:, 2]] - uv[F[:, 0]])).sum()))
        m = pv.PolyData(V3, faces)
        m.active_texture_coordinates = (uv * 2.0).astype(np.float32)        # ~8 checks across the face
        m = m.compute_normals(split_vertices=False)
        p = pv.Plotter(off_screen=True, window_size=(800, 900))
        p.add_mesh(m, texture=tex, smooth_shading=True, ambient=0.3, diffuse=0.8, specular=0.1)
        p.set_background("white")
        p.camera_position = [(0, 0.05, -3.3), (0, 0.05, -0.3), (0, 1, 0)]
        img = p.screenshot(return_img=True)
        p.close()
        return img

    from distortion.atlas import local_energy
    from distortion.eigensystem import SymmetricDirichlet
    E = local_energy(V3, F, SymmetricDirichlet())
    sched = [k for k in range(len(frames)) for _ in range(5 if k < 6 else 3)] + [len(frames) - 1] * 30
    W = Writer("vid_flatten", fps=12, poster=len(sched) - 1)
    fig = plt.figure(figsize=(13, 6.6))
    axL = fig.add_axes([0.0, 0.0, 0.46, 0.86])
    axR = fig.add_axes([0.5, 0.04, 0.42, 0.8])
    cax = fig.add_axes([0.93, 0.2, 0.012, 0.5])
    norm = LogNorm(1, 4)
    T = np.c_[-V3[:, 0], V3[:, 1]]                  # upright, as seen from the front (camera on -z)
    T = T - T.mean(0)

    def upright(X):                                  # rotation is free in the plane: align to T
        Xc = X - X.mean(0)
        U, _, Vt = np.linalg.svd(Xc.T @ T)
        R = U @ np.diag([1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        return Xc @ R

    for n_, k in enumerate(sched):
        X = upright(frames[k])
        if n_ == 0 or k != sched[n_ - 1]:
            img = face3d(X)
            s = np.linalg.svd(E.deformation(X), compute_uv=False)
            r = s[:, 0] / s[:, 1]
            axL.clear(); axL.imshow(img); axL.axis("off")
            axR.clear()
            pc = PolyCollection(X[F], array=np.clip(r, 1, 4), cmap="magma_r", norm=norm, ec="face", lw=0.1)
            axR.add_collection(pc)
            axR.autoscale_view(); axR.set_aspect("equal"); axR.axis("off")
            if n_ == 0:
                cb = fig.colorbar(pc, cax=cax)
                cb.ax.minorticks_off(); cb.set_ticks([1, 2, 4]); cb.set_ticklabels(["1", "2", "≥4"]); cb.set_label("σ₁/σ₂", fontsize=14)
            name = "Tutte (circle boundary)" if k == 0 else f"Symmetric Dirichlet, Newton iteration {k}"
            fig.texts.clear()
            fig.text(0.5, 0.93, f"{name}   ·   energy / area {en[k]:.2f}  (isometric = 4)", ha="center",
                     fontsize=17, color=style.INK2 if k else RED)
        W.add(fig)
    plt.close(fig)
    W.close()


@video
def vid_jelly3d():
    """The jelly drop with tetrahedra: implicit Euler, Neo-Hookean, floor barrier (scratch/solid_jelly.py).
    Surface triangles colored by the volume change J of their tet, as in the 2D jelly."""
    import pyvista as pv
    from distortion.solid import TetEnergy
    from distortion.eigensystem import NeoHookean
    D = np.load(os.path.join(HERE, "..", "scratch", "solid_jelly.npz"))
    V, T, fr, floor = D["V"], D["T"], D["frames"], float(D["floor"])
    E = TetEnergy(V, T, NeoHookean())
    tri = np.concatenate([T[:, [0, 2, 1]], T[:, [0, 1, 3]], T[:, [1, 2, 3]], T[:, [0, 3, 2]]])
    owner = np.tile(np.arange(len(T)), 4)
    _, idx, cnt = np.unique(np.sort(tri, 1), axis=0, return_index=True, return_counts=True)
    surf, own = tri[idx[cnt == 1]], owner[idx[cnt == 1]]
    faces = np.hstack([np.full((len(surf), 1), 3), surf]).ravel()
    lo, mid, hi = np.array([0.165, 0.471, 0.839]), np.array([0.94, 0.937, 0.925]), np.array([0.922, 0.408, 0.204])
    W = Writer("vid_jelly3d", fps=60, poster=int(np.argmin([f[:, 1].min() for f in fr])))
    p = pv.Plotter(off_screen=True, window_size=(1280, 820))
    p.set_background("white")
    p.add_mesh(pv.Plane(center=(0.3, floor, 0), direction=(0, 1, 0), i_size=9, j_size=5), color="#e2e1dc", lighting=False)
    mesh = pv.PolyData(fr[0].copy(), faces)
    mesh.cell_data["rgb"] = np.zeros((len(surf), 3), np.uint8)
    p.add_mesh(mesh, scalars="rgb", rgb=True, ambient=0.3, diffuse=0.8)
    p.camera_position = [(4.5, 2.5, 12.5), (0.2, -0.5, 0), (0, 1, 0)]
    for X in fr:
        t = np.clip(np.log2(E.dets(X)[own]) / 0.6, -1, 1)[:, None]
        col = np.where(t < 0, mid + (lo - mid) * -t, mid + (hi - mid) * t)
        mesh.points = X
        mesh.cell_data["rgb"][:] = (255 * col).astype(np.uint8)
        mesh.Modified()
        p.render()
        W.add(np.asarray(p.screenshot(return_img=True)))
    p.close()
    W.close()


@video
def vid_buckle():
    """Squash the cookie (feet pinned, head pressed down): triangles colored by their twist eigenvalue,
    a ghost of the softest mode of the whole Hessian, and that mode's eigenvalue, which touches 0 at
    the buckle. Data: scratch/buckle_symdir.pkl (scratch/buckle.py)."""
    import igl
    from matplotlib.colors import TwoSlopeNorm
    D = pickle.load(open(os.path.join(HERE, "..", "scratch", "buckle_symdir.pkl"), "rb"))
    V, F, G = D["V"], D["F"], D["G"]
    loop = igl.boundary_loop(np.ascontiguousarray(F, dtype=np.int64))
    loop = np.r_[loop, loop[0]]
    from distortion.boundary_barrier import boundary_crossings
    # no contact model: stop at the last frame before the body overlaps itself
    n = next(k for k, X in enumerate(D["frames"]) if boundary_crossings(X, F) > 0)
    sq, low, sway = np.array(D["squash"])[:n], np.array(D["lowest"])[:n], np.array(D["sway"])[:n]
    low = low / low[0]
    kb = int(np.argmin(low))                              # the buckle: lowest eigenvalue closest to 0
    cmap = matplotlib.colormaps["RdBu"]
    norm = TwoSlopeNorm(0, -3, 3)
    W = Writer("vid_buckle", fps=12, poster=len(sq) - 1)
    fig = plt.figure(figsize=(14, 6.2))
    ax = fig.add_axes([0.0, 0.02, 0.46, 0.86])
    a1 = fig.add_axes([0.56, 0.55, 0.41, 0.33])
    a2 = fig.add_axes([0.56, 0.12, 0.41, 0.33])
    for i in list(range(len(sq))) + [len(sq) - 1] * 24:
        X = D["frames"][i]
        ax.clear()
        lam = np.where(np.abs(D["twist"][i]) < 1e-6, 0, D["twist"][i])
        ax.add_collection(PolyCollection(X[F], array=lam, cmap=cmap, norm=norm, ec="face", lw=0.2))
        if i <= kb:                                        # the softest mode, before it takes over
            M = D["modes"][i]
            Y = X + 0.3 * M / np.abs(M).max()
            ax.plot(Y[loop, 0], Y[loop, 1], color=style.INK2, lw=1.4, ls="--", alpha=0.7)
        ax.plot(X[loop, 0], X[loop, 1], color=style.INK, lw=1.4)
        ax.plot(*X[G["feet"]].T, "s", color=style.INK, ms=3)
        h = X[G["head"]].mean(0)
        arrow(ax, (h[0], h[1] + 0.75), (h[0], h[1] + 0.12), color=style.INK, lw=3, ms=22)
        ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.2, 2.9); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title("red: twist eigenvalue < 0 (compressed)\ndashed: the softest mode", fontsize=15, color=style.INK2)
        for a, y, lab, c in [(a1, low, "lowest eigenvalue\nof the whole Hessian", style.ACCENT), (a2, sway, "sideways sway", style.BLUE)]:
            a.clear()
            a.plot(sq, y, color=style.GRID, lw=2)
            a.plot(sq[:i + 1], y[:i + 1], color=c, lw=3)
            a.plot([sq[i]], [y[i]], "o", color=c, ms=9)
            a.axhline(0, color=style.MUTED, lw=1)
            a.axvline(sq[kb], color="#c0392b", lw=1.2, ls=":")
            a.set_ylabel(lab, fontsize=13, color=style.INK2)
            a.set_xlim(0, sq[-1] * 1.02); a.set_yticks([0])
            a.spines[["top", "right"]].set_visible(False)
        a1.set_ylim(-0.05, 1.15); a1.set_xticklabels([])
        a1.text(sq[kb] - 0.015, 0.06, "buckles", color="#c0392b", fontsize=14, ha="right")
        a2.set_ylim(-0.05, sway.max() * 1.1)
        a2.set_xlabel("head pressed down", fontsize=14, color=style.INK2)
        W.add(fig)
    plt.close(fig)
    W.close()


@video
def vid_drag():
    from distortion.eigensystem import ARAP
    from distortion.planar import DragProblem, LinearElasticLike
    V, F, G = gb()
    rot = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    sh = np.array([0.45, 0.75])
    frames = {}
    for key, e in [("linear", LinearElasticLike()), ("arap", ARAP())]:
        P = DragProblem(V, F, e, G["feet"], G["right_hand"], k=1e3)
        X, out = V.copy(), [V.copy()]
        for a in np.linspace(0, np.radians(100), 41)[1:]:
            P.target = (V[G["right_hand"]] - sh) @ rot(a).T + sh
            X = P.solve(X, "pnewton", iters=50)
            out.append(X.copy())
        frames[key] = out
    for key, title in [("linear", "‖F − I‖²"), ("arap", "ARAP")]:
        W = Writer(f"vid_drag_{key}", fps=20, poster=len(frames[key]) - 1)
        fig, ax = plt.subplots(figsize=(5.6, 5.6))
        for X in frames[key] + [frames[key][-1]] * 20:
            ax.clear()
            k = draw_gb(ax, X, F, [G["feet"]])
            ax.set_title(f"{title}" + (f": {k} flipped" if k else ""), fontsize=18, color=RED if k else style.INK2)
            W.add(fig)
        plt.close(fig)
        W.close()


@video
def vid_solvers():
    """Wall-clock race: both solvers on one real-time clock; each panel shows its latest iterate."""
    import time
    from distortion.eigensystem import SymmetricDirichlet
    from distortion.planar import DragProblem
    V, F, G = gb()
    rot = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    sh = np.array([0.45, 0.75])
    runs = {}
    for m, n in [("gd", 1000), ("pnewton", 30)]:
        P = DragProblem(V, F, SymmetricDirichlet(), G["feet"], G["right_hand"], k=1e3)
        P.target = (V[G["right_hand"]] - sh) @ rot(np.radians(80)).T + sh
        X, out, ts = V.copy(), [V.copy()], [0.0]
        t0 = time.perf_counter()
        for it in range(n):
            Xn, info = P.step(X, m)
            ts.append(time.perf_counter() - t0)
            out.append(Xn.copy())
            if m == "pnewton" and abs(info["f"] - P.value(Xn)) < 1e-10 * max(1, abs(info["f"])):
                break
            X = Xn
        runs[m] = (out, np.array(ts))
        print(m, len(out) - 1, "iterations", f"{ts[-1]:.2f} s")
    T_end = runs["gd"][1][-1]
    fps = 30
    W = Writer("vid_solvers", fps=fps, poster=int(0.5 * fps))
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 5.8))
    frames = int(np.ceil(T_end * fps)) + 1
    for k in list(range(frames)) + [frames - 1] * int(1.5 * fps):
        tau = k / fps
        for ax, (m, name, c) in zip(axs, [("gd", "gradient descent", style.INK2), ("pnewton", "projected Newton", style.AQUA)]):
            out, ts = runs[m]
            i = int(np.searchsorted(ts, tau, side="right") - 1)
            ax.clear()
            draw_gb(ax, out[i], F, [G["feet"]])
            done = m == "pnewton" and i == len(out) - 1
            ax.set_title(f"{name}\niteration {i}" + ("  (converged)" if done else ""), fontsize=17, color=c)
        fig.suptitle(f"wall clock  t = {min(tau, T_end):.2f} s", fontsize=20, color=style.INK)
        W.add(fig)
    plt.close(fig)
    W.close()


@video
def vid_skull():
    """The skull's Earth map during optimization, seen from three sides at once: the face (Africa),
    and four more axis views; Greenland tinted orange, as in Act 0."""
    import pyvista as pv
    from PIL import ImageDraw
    from distortion import maps
    from distortion.render import earth_texture, corner_uv_mesh
    from distortion.sphere_param import parameterize
    d = np.load(os.path.join(HERE, "..", "..", "assets", "fig", "cache", "skull.npz"))
    V, F = d["V"], d["F"]
    iters = [d["naive"].copy()]
    parameterize(V, F, d["naive"], iters=200, verbose=False, callback=lambda it, S, f, m: iters.append(S.copy()))
    print("skull iterations", len(iters))
    src = open(os.path.join(HERE, "fig_d4_earth.py"), encoding="utf8").read()
    ns = {"np": np, "AFRICA": np.radians([20.0, 5.0])}
    exec(src[src.index("SCENES = {"):src.index("def maps_for")], ns)
    exec(src[src.index("def sphere_point"):src.index("def tissot_texture")], ns)
    scene = ns["SCENES"]["skull"]
    R = ns["orient"](V, iters[-1], scene["face"], scene["up"])
    # texture: faded Earth with Greenland tinted orange
    tex = earth_texture().astype(float)
    h, w = tex.shape[:2]
    lab = Image.new("L", (w, h), 0)
    dr = ImageDraw.Draw(lab)
    for outer, r in maps.countries(lambda q: q["NAME"] == "Greenland"):
        if outer:
            dr.polygon([((lo + np.pi) / (2 * np.pi) * w, (np.pi / 2 - la) / np.pi * h) for lo, la in r], fill=1)
    g = np.asarray(lab) > 0
    tex = tex * 0.55 + 235 * 0.45
    orange = np.array([int(style.ORANGE[i:i + 2], 16) for i in (1, 3, 5)], float)
    tex[g] = 0.25 * tex[g] + 0.75 * orange
    tex = tex.astype(np.uint8)
    # six cameras along the skull's own axes: front/back, left/right, top/bottom
    c = V.mean(0)
    dist = np.linalg.norm(np.array(scene["cam"][0]) - np.array(scene["cam"][1]))
    fwd = np.array(scene["cam"][0], float) - np.array(scene["cam"][1], float)
    up = np.array(scene["cam"][2], float)
    fwd -= (fwd @ up) * up
    fwd /= np.linalg.norm(fwd)
    side = np.cross(up, fwd)
    view = lambda dvec, vup: [tuple(c + dist * dvec), tuple(c), tuple(vup)]
    cams = [("front", view(fwd, up)), ("left", view(side, up)), ("top", view(up, -fwd)),
            ("back", view(-fwd, up)), ("right", view(-side, up)), ("bottom", view(-up, fwd))]
    tx = pv.Texture(np.ascontiguousarray(tex)); tx.repeat = True; tx.interpolate = True
    W = Writer("vid_skull", fps=10, poster=len(iters) - 1)
    for k, S in enumerate(iters + [iters[-1]] * 15):
        Vc, Fc, uv = corner_uv_mesh(V, S @ R.T, F)
        m = pv.PolyData(Vc, np.hstack([np.full((len(Fc), 1), 3), Fc]).ravel())
        m.active_texture_coordinates = uv.astype(np.float32)
        m = m.compute_normals(split_vertices=False)
        tiles = []
        for title, cam in cams:
            p = pv.Plotter(off_screen=True, window_size=(420, 400), lighting="none")
            p.add_light(pv.Light(position=(0.6, 0.8, 1.0), light_type="camera light", intensity=1.0))
            p.add_light(pv.Light(position=(-1.0, 0.1, 0.6), light_type="camera light", intensity=0.35))
            p.add_mesh(m, texture=tx, smooth_shading=True, ambient=0.12, diffuse=0.95, specular=0.25)
            p.set_background("white")
            p.camera_position = cam
            p.add_text(title, font_size=13, color="black", position="upper_edge")
            tiles.append(np.asarray(p.screenshot(return_img=True)))
            p.close()
        img = np.concatenate([np.concatenate(tiles[:3], 1), np.concatenate(tiles[3:], 1)], 0)
        frame = Image.fromarray(img)
        ImageDraw.Draw(frame).text((12, img.shape[0] - 36), f"iteration {min(k, len(iters) - 1)}", fill=(30, 30, 30),
                                   font=__import__("PIL.ImageFont", fromlist=["truetype"]).truetype("C:/Windows/Fonts/segoeui.ttf", 26))
        W.add(np.asarray(frame))
    W.close()


def face_texture(path=None, res=1024):
    """Texture over the gingerbread's rest bounding box: gingerbread brown, with a face on the head.
    path: an image to paste as the face (cropped to a circle); None draws a placeholder emoji."""
    from PIL import ImageDraw
    V, F, G = gb()
    lo, hi = V.min(0) - 0.05, V.max(0) + 0.05
    W_, H_ = res, int(res * (hi[1] - lo[1]) / (hi[0] - lo[0]))
    img = Image.new("RGB", (W_, H_), (196, 134, 72))
    dr = ImageDraw.Draw(img)
    to_px = lambda x, y: ((x - lo[0]) / (hi[0] - lo[0]) * W_, (hi[1] - y) / (hi[1] - lo[1]) * H_)
    # icing buttons and squiggles on the body
    for y in [0.75, 0.35, -0.05]:
        cx, cy = to_px(0, y); r = 0.06 / (hi[0] - lo[0]) * W_
        dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(235, 104, 52))
    for side in [-1, 1]:
        pts = [to_px(side * (1.0 + 0.08 * k), 0.62 + 0.06 * (-1) ** k) for k in range(5)]
        dr.line(pts, fill=(250, 248, 240), width=int(0.025 / (hi[0] - lo[0]) * W_))
    c, R = np.array([0.0, 1.62]), 0.55
    x0, y0 = to_px(c[0] - R, c[1] + R); x1, y1 = to_px(c[0] + R, c[1] - R)
    box = [int(x0), int(y0), int(x1), int(y1)]
    size = (box[2] - box[0], box[3] - box[1])
    if path:
        face = Image.open(path).convert("RGB")
        w, h = face.size; m = min(w, h)
        face = face.crop(((w - m) // 2, (h - m) // 2, (w + m) // 2, (h + m) // 2)).resize(size)
    else:                                               # placeholder: a round cartoon baby face
        face = Image.new("RGB", size, (196, 134, 72))
        fd = ImageDraw.Draw(face); u = size[0] / 100
        fd.ellipse([2 * u, 2 * u, 98 * u, 98 * u], fill=(255, 224, 189))
        for ex in [33, 67]:
            fd.ellipse([(ex - 7) * u, 36 * u, (ex + 7) * u, 52 * u], fill=(40, 30, 30))
            fd.ellipse([(ex - 3) * u, 39 * u, (ex + 1) * u, 44 * u], fill=(255, 255, 255))
            fd.ellipse([(ex - 13) * u, 56 * u, (ex + 5) * u, 66 * u], fill=(250, 160, 160))
        fd.arc([36 * u, 54 * u, 64 * u, 78 * u], 20, 160, fill=(150, 50, 50), width=int(3 * u))
        fd.ellipse([44 * u, 8 * u, 56 * u, 20 * u], fill=(120, 80, 50))          # a tuft of hair
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).ellipse([0, 0, size[0] - 1, size[1] - 1], fill=255)
    img.paste(face, box[:2], mask)
    return np.asarray(img), lo, hi


def frog_texture(res=1400):
    """An original cartoon frog laid out on the gingerbread rest shape (front view, T-pose)."""
    from PIL import ImageDraw, ImageFilter
    V, F, G = gb()
    lo, hi = V.min(0) - 0.05, V.max(0) + 0.05
    W_, H_ = res, int(res * (hi[1] - lo[1]) / (hi[0] - lo[0]))
    sc = W_ / (hi[0] - lo[0])
    P = lambda x, y: ((x - lo[0]) * sc, (hi[1] - y) * sc)
    E = lambda x, y, rx, ry: [P(x - rx, y + ry), P(x + rx, y - ry)]
    green, dark, belly, cheek = (250, 212, 70), (214, 160, 40), (255, 246, 214), (250, 150, 140)   # yellow frog
    img = Image.new("RGB", (W_, H_), green)
    d = ImageDraw.Draw(img)
    # spots on the back of the arms and legs
    for x, y, r in [(-1.05, 0.68, 0.06), (1.1, 0.6, 0.05), (-0.5, -1.0, 0.07), (0.55, -1.3, 0.06), (-0.62, -1.5, 0.05)]:
        d.ellipse(E(x, y, r, r), fill=dark)
    # creamy belly
    d.ellipse(E(0, 0.32, 0.46, 0.62), fill=belly)
    # webbed hands and feet: darker green with toe pads
    for sx in [-1, 1]:
        for k in range(3):
            d.ellipse(E(sx * 1.55, 0.55 + 0.11 * (k - 1), 0.06, 0.05), fill=dark)
        for k in range(3):
            d.ellipse(E(sx * (0.45 + 0.12 * (k - 1)), -1.93, 0.055, 0.045), fill=dark)
    # head: green with a pale chin, two big eyes bulging at the top
    for ex in [-0.24, 0.24]:
        d.ellipse(E(ex, 1.92, 0.19, 0.19), fill=green)
        d.ellipse(E(ex, 1.92, 0.15, 0.15), fill=(255, 255, 255))
        d.ellipse(E(ex + 0.02, 1.9, 0.08, 0.09), fill=(30, 30, 30))
        d.ellipse(E(ex + 0.05, 1.95, 0.03, 0.03), fill=(255, 255, 255))
    for cx in [-0.33, 0.33]:
        d.ellipse(E(cx, 1.58, 0.09, 0.05), fill=cheek)
    d.arc(E(0, 1.66, 0.26, 0.16), 20, 160, fill=(120, 40, 40), width=int(0.025 * sc))
    for nx in [-0.05, 0.05]:
        d.ellipse(E(nx, 1.72, 0.012, 0.012), fill=dark)
    return np.asarray(img.filter(ImageFilter.SMOOTH)), lo, hi


@video
def vid_twist_face():
    """The head twist with a face on it: same three runs as vid_twist, textured in the rest shape."""
    import igl
    import pyvista as pv
    V, F, G = gb()
    T = pickle.load(open(os.path.join(HERE, "..", "scratch", "twist180.pkl"), "rb"))
    tex, lo, hi = frog_texture() if os.environ.get("FACE") == "frog" else face_texture(os.environ.get("FACE"))
    uv = ((V - lo) / (hi - lo)).astype(np.float32)
    loop = igl.boundary_loop(np.ascontiguousarray(F, dtype=np.int64))
    loop = np.r_[loop, loop[0]]
    faces = np.hstack([np.full((len(F), 1), 3), F]).ravel()
    t = pv.Texture(np.ascontiguousarray(tex)); t.interpolate = True
    panels = [("arap", "ARAP"), ("symdir", "Symmetric Dirichlet"), ("inj", "Sym. Dirichlet + boundary barrier")]
    W = Writer("vid_twist_face", fps=12, poster=48)
    n = len(T["angles"])
    for i in list(range(n)) + [n - 1] * 24:
        tiles = []
        for key, name in panels:
            X = T[key][i]
            m = pv.PolyData(np.c_[X, np.zeros(len(X))], faces)
            m.active_texture_coordinates = uv
            p = pv.Plotter(off_screen=True, window_size=(520, 560))
            p.add_mesh(m, texture=t, lighting=False)
            p.add_lines(np.c_[np.repeat(X[loop], 2, axis=0)[1:-1], np.zeros(2 * len(loop) - 2)], color="black", width=4)
            p.set_background("white")
            p.camera_position = [(0, 1.55, 5), (0, 1.55, 0), (0, 1, 0)]
            p.enable_parallel_projection(); p.camera.parallel_scale = 1.5
            p.add_text(name, font_size=12, color="black", position="upper_edge")
            tiles.append(np.asarray(p.screenshot(return_img=True)))
            p.close()
        frame = Image.fromarray(np.concatenate(tiles, 1))
        from PIL import ImageDraw, ImageFont
        ImageDraw.Draw(frame).text((12, frame.height - 40), f"head turned {T['angles'][i]:.0f}°", fill=(40, 40, 40),
                                   font=ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 26))
        W.add(np.asarray(frame))
    W.close()


# ---------------------------------------------------------------- the face, ARAP (solver act)
def _face_arap():
    """Data of scratch/flatten_face_arap.py, plus helpers: upright UV, per-triangle ARAP density."""
    from distortion.atlas import local_energy
    from distortion.eigensystem import ARAP, signed_singular_values
    D = pickle.load(open(os.path.join(HERE, "..", "scratch", "flatten_face_arap.pkl"), "rb"))
    V3, F = D["V3"], D["F"]
    E = local_energy(V3, F, ARAP())
    T = np.c_[-V3[:, 0], V3[:, 1]]
    T = T - T.mean(0)

    def upright(X):
        Xc = X - X.mean(0)
        U, _, Vt = np.linalg.svd(Xc.T @ T)
        return Xc @ (U @ np.diag([1, np.sign(np.linalg.det(U @ Vt))]) @ Vt)

    def density(X):
        s = signed_singular_values(E.deformation(X))
        return np.sum((s - 1) ** 2, axis=1), s[:, 1] <= 0

    def snap(run, i):                                # latest saved iterate <= i
        ks = np.array(sorted(D[run]["snaps"]))
        return D[run]["snaps"][int(ks[np.searchsorted(ks, i, side="right") - 1])]
    return D, V3, F, upright, density, snap


def _uv_panel(ax, X, F, psi, flip, norm):
    ax.clear()
    ax.add_collection(PolyCollection(X[F], array=np.clip(psi, norm.vmin, norm.vmax), cmap="magma_r", norm=norm,
                                     ec="face", lw=0.1))
    if flip.any():
        ax.add_collection(PolyCollection(X[F[flip]], fc=RED, ec=RED, lw=1.5))
    ax.autoscale_view(); ax.set_aspect("equal"); ax.axis("off")


@video
def vid_flatten_arap():
    """Max Planck's face flattened with ARAP by projected Newton (Tutte start, free boundary), one iterate
    per beat. Left: the 3D face with a checker drawn through the current UV map. Right: the UV plane,
    triangles shaded by the ARAP density sum (sigma_i - 1)^2. Data: scratch/flatten_face_arap.pkl."""
    import pyvista as pv
    from matplotlib.colors import LogNorm
    D, V3, F, upright, density, snap = _face_arap()
    frames = [D["pnewton"]["snaps"][k] for k in sorted(D["pnewton"]["snaps"])]
    en = [h["f"] for h in D["pnewton"]["hist"]]
    e1, e2 = V3[F[:, 1]] - V3[F[:, 0]], V3[F[:, 2]] - V3[F[:, 0]]
    area3 = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1).sum()
    ii, jj = np.meshgrid(np.arange(1024), np.arange(1024))
    chk = np.where((((ii // 64) + (jj // 64)) % 2).astype(bool)[..., None], [42, 120, 214], [236, 235, 231]).astype(np.uint8)
    tex = pv.Texture(chk); tex.repeat = True; tex.interpolate = True
    faces = np.hstack([np.full((len(F), 1), 3), F]).ravel()

    def face3d(X):
        # the checker is drawn at fixed UV scale: the Tutte map's shrunken interior shows as big checks
        m = pv.PolyData(V3, faces)
        m.active_texture_coordinates = ((X - X.mean(0)) * 2.0).astype(np.float32)
        m = m.compute_normals(split_vertices=False)
        p = pv.Plotter(off_screen=True, window_size=(800, 900))
        p.add_mesh(m, texture=tex, smooth_shading=True, ambient=0.3, diffuse=0.8, specular=0.1)
        p.set_background("white")
        p.camera_position = [(0, 0.05, -3.3), (0, 0.05, -0.3), (0, 1, 0)]
        img = p.screenshot(return_img=True)
        p.close()
        return img

    sched = [k for k in range(len(frames)) for _ in range(6 if k < 4 else 2)] + [len(frames) - 1] * 30
    W = Writer("vid_flatten_arap", fps=12, poster=len(sched) - 1)
    fig = plt.figure(figsize=(13, 6.6))
    axL = fig.add_axes([0.0, 0.0, 0.46, 0.86])
    axR = fig.add_axes([0.48, 0.04, 0.40, 0.8])
    cax = fig.add_axes([0.89, 0.2, 0.012, 0.5])
    norm = LogNorm(1e-3, 1)
    for n_, k in enumerate(sched):
        X = upright(frames[k])
        if n_ == 0 or k != sched[n_ - 1]:
            axL.clear(); axL.imshow(face3d(X)); axL.axis("off")
            psi, flip = density(X)
            _uv_panel(axR, X, F, psi, flip, norm)
            if n_ == 0:
                cb = fig.colorbar(axR.collections[0], cax=cax)
                cb.ax.minorticks_off(); cb.set_ticks([1e-3, 1e-2, 1e-1, 1])
                cb.set_ticklabels(["≤0.001", "0.01", "0.1", "≥1"]); cb.ax.tick_params(labelsize=13)
                cb.set_label("Σ(σᵢ − 1)²", fontsize=16)
            name = "Tutte (circle boundary)" if k == 0 else f"ARAP, projected Newton iteration {k}"
            fig.texts.clear()
            fig.text(0.5, 0.93, f"{name}   ·   energy / area {en[k]:.4f}  (isometric = 0)", ha="center",
                     fontsize=18, color=style.INK2 if k else RED)
        W.add(fig)
    plt.close(fig)
    W.close()


@video
def vid_solvers_face():
    """Wall-clock race on the face (ARAP, Tutte start): gradient descent vs projected Newton on one
    real-time clock (the times recorded in scratch/flatten_face_arap.pkl); each panel shows its latest iterate."""
    from matplotlib.colors import LogNorm
    D, V3, F, upright, density, snap = _face_arap()
    T_end, fps = 8.0, 30
    norm = LogNorm(1e-3, 1)
    W = Writer("vid_solvers_face", fps=fps, poster=int(round(D["pnewton"]["t"][-1] * fps)) + 3)
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.6))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.80, wspace=0.06)
    frames = int(np.ceil(T_end * fps)) + 1
    last = {}
    for k in list(range(frames)) + [frames - 1] * int(1.5 * fps):
        tau = k / fps
        for ax, (m, name, c) in zip(axs, [("gd", "gradient descent", style.INK2), ("pnewton", "projected Newton", style.AQUA)]):
            ts = D[m]["t"]
            i = int(np.searchsorted(ts, tau, side="right") - 1)
            if last.get(m) != i:
                X = upright(snap(m, i))
                psi, flip = density(X)
                _uv_panel(ax, X, F, psi, flip, norm)
                done = m == "pnewton" and i == len(ts) - 1
                f = D[m]["hist"][i]["f"]
                ax.set_title(f"{name}\niteration {i}" + ("  (converged)" if done else "") + f"\nenergy / area {f:.4f}",
                             fontsize=18, color=c)
                last[m] = i
        fig.texts.clear()
        fig.text(0.5, 0.94, f"wall clock  t = {min(tau, T_end):.2f} s", ha="center", fontsize=22, color=style.INK)
        W.add(fig)
    plt.close(fig)
    W.close()


for f in VIDEOS:
    if not ONLY or f.__name__ in ONLY:
        f()
