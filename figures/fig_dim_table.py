"""The master visual of the talk: {2D, 3D} x {parameterization, deformation, dynamics}, one picture
per cell, every cell an isotropic energy of F minimized by projected Newton.

  parameterization  2D: Max Planck's face flattened (scratch/flatten_face.pkl)
                    3D: the skull onto the sphere (quasi-3D: tets are triangles coned to the center)
  deformation       2D: gingerbread arm raise (scratch/act2.pkl)
                    3D: a giraffe lowers its head (scratch/demo3d/giraffe.py; demo3d_giraffe_sphere_cell.png)
  dynamics          2D: sixteen letters fall into a box (scratch/demo2d/scenes.py crowd; demo2d_crowd_cell.png)
                    3D: an armadillo falls on a cantilever (demo3d_cantilever_cell.png)
Every cell has its own shape. (The old 3D cells, the extruded gingerbread, are solid_cell / jelly3d_cell.)

    python figures/fig_dim_table.py <out-dir>
writes dim_table.png, dim_table_2d.png / _3d.png (one column highlighted), dim_table_code.png.
"""
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.collections import PolyCollection

from distortion import style

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "..", "assets", "fig")
os.makedirs(OUT, exist_ok=True)
style.use()


def crop(a, pad=10):
    ys, xs = np.nonzero((a[..., :3] < 245).any(-1))
    return a[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]


def solid_cell():
    D = pickle.load(open(os.path.join(HERE, "..", "scratch", "solid_drag.pkl"), "rb"))
    V, T = D["V"], D["T"]
    tri = np.concatenate([T[:, [0, 2, 1]], T[:, [0, 1, 3]], T[:, [1, 2, 3]], T[:, [0, 3, 2]]])
    key = np.sort(tri, 1)
    _, idx, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
    surf = tri[idx[cnt == 1]]                                   # boundary faces
    faces = np.hstack([np.full((len(surf), 1), 3), surf]).ravel()
    p = pv.Plotter(off_screen=True, window_size=(1100, 1100))
    p.add_mesh(pv.PolyData(V, faces), color="#d9d6cf", opacity=0.25, smooth_shading=False)
    m = pv.PolyData(D["frames"][-1], faces)
    p.add_mesh(m, color="#c98a4b", smooth_shading=False, show_edges=True, edge_color="#7a5230", line_width=0.4,
               ambient=0.3, diffuse=0.8)
    p.set_background("white")
    p.camera_position = "xy"
    p.camera.azimuth = 35
    p.camera.elevation = 18
    p.reset_camera()
    p.camera.zoom(1.15)
    img = crop(np.asarray(p.screenshot(return_img=True)))
    p.close()
    plt.imsave(os.path.join(OUT, "solid_drag.png"), img)
    return img


def flatten_cell():
    D = pickle.load(open(os.path.join(HERE, "..", "scratch", "flatten_face.pkl"), "rb"))
    V3, F, X = D["V3"], D["F"], D["frames"][-1]
    T = np.c_[-V3[:, 0], V3[:, 1]]; T -= T.mean(0)
    Xc = X - X.mean(0)
    U, _, Vt = np.linalg.svd(Xc.T @ T)
    X = Xc @ (U @ np.diag([1, np.sign(np.linalg.det(U @ Vt))]) @ Vt)
    ii, jj = np.meshgrid(np.arange(1024), np.arange(1024))
    chk = np.where((((ii // 128) + (jj // 128)) % 2).astype(bool)[..., None], [42, 120, 214], [236, 235, 231]).astype(np.uint8)
    tex = pv.Texture(chk); tex.repeat = True
    faces = np.hstack([np.full((len(F), 1), 3), F]).ravel()
    m = pv.PolyData(V3, faces)
    m.active_texture_coordinates = (X * 1.2).astype(np.float32)          # 8 checks per texture tile
    m = m.compute_normals(split_vertices=False)
    p = pv.Plotter(off_screen=True, window_size=(900, 1000))
    p.add_mesh(m, texture=tex, smooth_shading=True, ambient=0.3, diffuse=0.8, specular=0.1)
    p.set_background("white")
    p.camera_position = [(0, 0.05, -3.3), (0, 0.05, -0.3), (0, 1, 0)]
    face = crop(np.asarray(p.screenshot(return_img=True)))
    p.close()
    fig = plt.figure(figsize=(9, 4.6))
    a1 = fig.add_axes([0, 0, 0.45, 1]); a2 = fig.add_axes([0.55, 0.05, 0.45, 0.9])
    a1.imshow(face); a1.axis("off")
    # the flat face with the same checker: color each triangle by the checker cell of its centroid
    c = X[F].mean(1) * 1.2
    on = ((np.floor(c[:, 0] * 8) + np.floor(c[:, 1] * 8)) % 2).astype(bool)
    a2.add_collection(PolyCollection(X[F], fc=np.where(on[:, None], [42 / 255, 120 / 255, 214 / 255], [.925, .92, .905]),
                                     ec="face", lw=0.1))
    a2.autoscale_view(); a2.set_aspect("equal"); a2.axis("off")
    fig.text(0.5, 0.5, "→", fontsize=40, ha="center", va="center", color=style.INK2)
    fig.savefig(os.path.join(OUT, "flatten_pair.png"))
    plt.close(fig)
    return crop(plt.imread(os.path.join(OUT, "flatten_pair.png")))


def gingerbread_cell():
    from distortion.planar import gingerbread
    V, F, G = gingerbread()
    X = pickle.load(open(os.path.join(HERE, "..", "scratch", "act2.pkl"), "rb"))["raise_arap"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_collection(PolyCollection(V[F], fc="#e6e4df", ec="none"))
    ax.add_collection(PolyCollection(X[F], fc="#c98a4b", ec="#7a5230", lw=0.15))
    ax.plot(*X[G["feet"]].T, "s", color=style.INK, ms=2.5)
    ax.autoscale_view(); ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(os.path.join(OUT, "gb_raise.png"))
    plt.close(fig)
    return crop(plt.imread(os.path.join(OUT, "gb_raise.png")))


def sphere_cell():
    """The skull and its sphere, as the face and its plane: same layout as flatten_cell."""
    from distortion.render import earth_texture, render_textured
    S = pv.Icosphere(radius=1.0, nsub=5)
    P = np.asarray(S.points)
    Fs = np.asarray(S.faces).reshape(-1, 4)[:, 1:]
    globe = crop(np.asarray(render_textured(P, P, Fs, earth_texture(), size=(900, 900), shape=True,
                                            camera=[(2.2, 3.4, 1.2), (0, 0, 0), (0, 0, 1)])))
    skull = crop(plt.imread(os.path.join(FIG, "d4_skull_symdir.png")))
    fig = plt.figure(figsize=(9, 4.6))
    a1 = fig.add_axes([0, 0, 0.45, 1]); a2 = fig.add_axes([0.55, 0.08, 0.42, 0.84])
    a1.imshow(skull); a1.axis("off")
    a2.imshow(globe); a2.axis("off")
    fig.text(0.5, 0.5, "→", fontsize=40, ha="center", va="center", color=style.INK2)
    fig.savefig(os.path.join(OUT, "sphere_pair.png"))
    plt.close(fig)
    return crop(plt.imread(os.path.join(OUT, "sphere_pair.png")))



BROWN, EDGE, GHOST = "#c98a4b", "#7a5230", "#e6e4df"


def impact(frames):
    return int(np.argmin([f[:, 1].min() for f in frames]))


def jelly2d_cell():
    from distortion.planar import gingerbread
    _, F, _ = gingerbread(max_area=0.006)
    fr = np.load(os.path.join(HERE, "..", "scratch", "jelly_frames.npy"))
    k = impact(fr)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_collection(PolyCollection(fr[0][F], fc=GHOST, ec="none"))
    ax.add_collection(PolyCollection(fr[k][F], fc=BROWN, ec=EDGE, lw=0.15))
    ax.axhspan(-3.6, -3.0, color=style.GRID)
    ax.set_xlim(-3.0, 3.4); ax.set_ylim(-3.6, 3.4); ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(os.path.join(OUT, "jelly2d_cell.png"))
    plt.close(fig)
    return crop(plt.imread(os.path.join(OUT, "jelly2d_cell.png")))


def boundary_faces(T):
    tri = np.concatenate([T[:, [0, 2, 1]], T[:, [0, 1, 3]], T[:, [1, 2, 3]], T[:, [0, 3, 2]]])
    _, idx, cnt = np.unique(np.sort(tri, 1), axis=0, return_index=True, return_counts=True)
    surf = tri[idx[cnt == 1]]
    return np.hstack([np.full((len(surf), 1), 3), surf]).ravel()


def jelly3d_cell():
    D = np.load(os.path.join(HERE, "..", "scratch", "solid_jelly.npz"))
    fr, faces, floor = D["frames"], boundary_faces(D["T"]), float(D["floor"])
    k = impact(fr)
    p = pv.Plotter(off_screen=True, window_size=(1100, 1100))
    p.add_mesh(pv.PolyData(fr[0], faces), color="#d9d6cf", opacity=0.3)
    p.add_mesh(pv.PolyData(fr[k], faces), color=BROWN, show_edges=True, edge_color=EDGE, line_width=0.4,
               ambient=0.3, diffuse=0.8)
    p.add_mesh(pv.Plane(center=(0.2, floor, 0), direction=(0, 1, 0), i_size=6, j_size=3), color="#e2e1dc", lighting=False)
    p.set_background("white")
    p.camera_position = "xy"
    p.camera.azimuth = 35
    p.camera.elevation = 18
    p.reset_camera()
    p.camera.zoom(1.25)
    img = crop(np.asarray(p.screenshot(return_img=True)))
    p.close()
    plt.imsave(os.path.join(OUT, "jelly3d_cell.png"), img)
    return img


ROWS = ["parameter-\nization", "deformation", "dynamics"]
COLS = ["2D: triangles\n" + r"$F \in \mathbb{R}^{2\times2}$", "3D: tetrahedra\n" + r"$F \in \mathbb{R}^{3\times3}$"]
NOTE = {(0, 1): "quasi-3D: radial stretch fixed"}
CODE = {(0, 0): "demos/flatten.py", (0, 1): "demos/sphere.py",          # the runnable script for each cell
        (1, 0): "demos/drag2d.py", (1, 1): "demos/drag3d.py",
        (2, 0): "demos/letters.py", (2, 1): "demos/cantilever.py"}
MORE = ["volume maps [Abulnaga\net al. 2023], seamless\nmaps, …", "shells, cloth,\n4D, …",
        "friction, fracture,\nmuscles, …"]


def table(cells, name, highlight=None, code=False):
    """Rows x columns of cell images. highlight: box one column, dim the other. code: name the module
    under each cell and add an open column (more to explore)."""
    nr = len(ROWS)
    cols = COLS + (["more to explore"] if code else [])
    fig = plt.figure(figsize=(16 if code else 14, 11.5))
    x0, y0, w, h, wm = (0.13, 0.015, 0.33, 0.29, 0.21) if code else (0.17, 0.015, 0.415, 0.29, 0)
    xs = [x0, x0 + w, x0 + 2 * w]
    for j, ct in enumerate(cols):
        fig.text(xs[j] + (w if j < 2 else wm) / 2, 0.94, ct, ha="center", va="center", fontsize=28,
                 color=style.MUTED if j == 2 else (style.ACCENT if highlight == j else style.INK2))
    if not code:
        fig.text(x0 + w, 0.94, "+1 dim", ha="center", va="center", fontsize=22, color=style.ACCENT,
                 bbox=dict(boxstyle="rarrow,pad=0.35", fc="white", ec=style.ACCENT, lw=1.8))
    for i, rt in enumerate(ROWS):
        fig.text(0.065 if code else 0.08, y0 + (nr - 0.5 - i) * h, rt, ha="center", va="center", fontsize=26,
                 color=style.INK2)
    for (i, j), img in cells.items():
        lab = CODE[(i, j)] if code else NOTE.get((i, j))
        bot = 0.035 if lab else 0.01
        ax = fig.add_axes([xs[j] + 0.012, y0 + (nr - 1 - i) * h + bot, w - 0.024, h - bot - 0.012])
        ax.imshow(img); ax.axis("off")
        if highlight is not None and j != highlight:
            ax.images[0].set_alpha(0.25)
        if lab:
            fig.text(xs[j] + w / 2, y0 + (nr - 1 - i) * h + 0.018, lab, ha="center", va="center",
                     fontsize=16 if code else 17, family="monospace" if code else None,
                     color=style.ACCENT if code else style.MUTED,
                     alpha=0.35 if (highlight is not None and j != highlight) else 1)
    if code:
        # one open box for the whole column (not one per row)
        fig.add_artist(plt.Rectangle((xs[2] + 0.012, y0 + 0.012), wm - 0.024, nr * h - 0.024,
                                     transform=fig.transFigure, fill=False, ec=style.MUTED, lw=2, ls=(0, (6, 4))))
        fig.text(xs[2] + wm / 2, y0 + nr * h * 0.62, "?", ha="center", va="center", fontsize=70, color=style.GRID)
        fig.text(xs[2] + wm / 2, y0 + nr * h * 0.38, "\n\n".join(MORE), ha="center", va="center", fontsize=16,
                 color=style.MUTED, linespacing=1.3)
    if highlight is not None:
        fig.add_artist(plt.Rectangle((xs[highlight] + 0.005, y0 + 0.004), w - 0.01, nr * h - 0.008,
                                     transform=fig.transFigure, fill=False, ec=style.ACCENT, lw=3))
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


def code_table(cells, name):
    """Transposed table for the closing slide: rows = 2D / 3D, columns = applications, each cell with its
    module name; one open box on the right (more to explore)."""
    apps = ["parameterization", "deformation", "dynamics"]
    dims = ["2D", "3D"]
    fig = plt.figure(figsize=(18, 8.2))
    x0, y0, w, h, wm = 0.07, 0.02, 0.24, 0.43, 0.19
    for j, t in enumerate(apps):
        fig.text(x0 + (j + 0.5) * w, 0.94, t, ha="center", va="center", fontsize=28, color=style.INK2)
    fig.text(x0 + 3 * w + wm / 2, 0.94, "more to explore", ha="center", va="center", fontsize=28, color=style.MUTED)
    for i, t in enumerate(dims):
        fig.text(0.035, y0 + (1.5 - i) * h, t, ha="center", va="center", fontsize=30, color=style.INK2)
    for (a, d), img in cells.items():
        ax = fig.add_axes([x0 + a * w + 0.01, y0 + (1 - d) * h + 0.05, w - 0.02, h - 0.065])
        ax.imshow(img); ax.axis("off")
        fig.text(x0 + (a + 0.5) * w, y0 + (1 - d) * h + 0.025, CODE[(a, d)], ha="center", va="center",
                 fontsize=17, family="monospace", color=style.ACCENT)
    xb = x0 + 3 * w + 0.012
    fig.add_artist(plt.Rectangle((xb, y0 + 0.012), wm - 0.024, 2 * h - 0.024, transform=fig.transFigure,
                                 fill=False, ec=style.MUTED, lw=2, ls=(0, (6, 4))))
    fig.text(xb + wm / 2 - 0.012, y0 + 1.45 * h, "?", ha="center", va="center", fontsize=70, color=style.GRID)
    fig.text(xb + wm / 2 - 0.012, y0 + 0.8 * h, "\n\n".join(MORE), ha="center", va="center", fontsize=18,
             color=style.MUTED, linespacing=1.3)
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


def still(name):
    return crop(plt.imread(os.path.join(FIG, name)))


def relief_pair(a, b, name):
    """Textured mesh -> its domain with the same texture, shaded by the mesh's relief (fig_param_relief.py)."""
    A, B = crop(plt.imread(os.path.join(FIG, a))), crop(plt.imread(os.path.join(FIG, b)))
    fig = plt.figure(figsize=(9, 4.6))
    a1 = fig.add_axes([0, 0, 0.45, 1]); a2 = fig.add_axes([0.55, 0.0, 0.45, 1])
    a1.imshow(A); a1.axis("off")
    a2.imshow(B); a2.axis("off")
    fig.text(0.5, 0.5, "→", fontsize=40, ha="center", va="center", color=style.INK2)
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    return crop(plt.imread(os.path.join(OUT, name)))


cells = {(0, 0): relief_pair("relief_tex_face3d_a30.png", "relief_tex_facedom_a30.png", "flatten_pair_a30.png"),
         (0, 1): relief_pair("relief_tex_skull3d_a30.png", "relief_tex_skulldom_a30.png", "sphere_pair_a30.png"),
         (1, 0): gingerbread_cell(), (1, 1): still("demo3d_giraffe_sphere_cell.png"),
         (2, 0): still("demo2d_crowd_cell.png"), (2, 1): still("demo3d_cantilever_cell.png")}
table(cells, "dim_table.png")
table(cells, "dim_table_2d.png", highlight=0)
table(cells, "dim_table_3d.png", highlight=1)
code_table(cells, "dim_table_code.png")
print("ok")
