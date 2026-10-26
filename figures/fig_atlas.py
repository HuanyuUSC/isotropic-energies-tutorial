"""Texture atlas of the bunny: charts (seams in black) with a checker drawn through the atlas, and
the flattened charts packed in the plane.

    python figures/fig_atlas.py <out-dir> [n_charts]
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from distortion import style
from distortion.atlas import flatten, grow_charts, is_disk, pack
from distortion.planar import dets

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 10
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)
style.use()
d = np.load(os.path.join(HERE, "..", "..", "assets", "fig", "cache", "bunny.npz"))
V, F = d["V"], d["F"]
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#b5005b", "#8a6bd1", "#d9a21b", "#3aa6b9", "#c45a7a",
           "#6b8e23", "#7f7f7f", "#e07b9b", "#4a6fa5"]

def chart_of(fs):
    vs, Fl = np.unique(F[fs], return_inverse=True)
    return vs, Fl.reshape(-1, 3)


def build(fs, depth=0):
    """Flatten a chart; split it in two (and recurse) if it is not a disk or distorts too much."""
    vs, Fl = chart_of(fs)
    ok = is_disk(Fl)
    if ok:
        X, E = flatten(V[vs], Fl)
        s = np.linalg.svd(E.deformation(X), compute_uv=False)
        w = E.A / E.A.sum()
        q = w @ (s[:, 0] / s[:, 1])
        ok = q < 1.35 or depth >= 4
    if not ok and depth < 4 and len(fs) > 200:
        sub = grow_charts(V[vs], Fl, 2)
        return build(fs[sub == 0], depth + 1) + build(fs[sub == 1], depth + 1)
    if not is_disk(Fl):
        raise SystemExit("could not split into disks")
    print(f"chart: {len(fs)} faces, flips {(dets(X, Fl) <= 0).sum()}, mean s1/s2 {q:.3f}")
    return [(fs, vs, Fl, X)]


lab = grow_charts(V, F, K)
charts = []
for c in range(K):
    charts += build(np.nonzero(lab == c)[0])
print(len(charts), "charts")
UV = pack([c[3] for c in charts])
scale = max(np.concatenate(UV).max(0))

# ---- 2D atlas
fig, ax = plt.subplots(figsize=(7, 7))
for k, ((fs, vs, Fl, _), X) in enumerate(zip(charts, UV)):
    ax.add_collection(PolyCollection(X[Fl], fc=PALETTE[k % len(PALETTE)], ec="white", lw=0.15, alpha=0.9))
ax.set_aspect("equal")
ax.autoscale_view()
ax.axis("off")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "atlas_2d.png"))
plt.close(fig)

# ---- 3D: checker through the atlas, each chart tinted, seams in black
import pyvista as pv
n = 64
ii, jj = np.meshgrid(np.arange(1024), np.arange(1024))
chk = ((ii // (1024 // 16) + jj // (1024 // 16)) % 2).astype(bool)
p = pv.Plotter(off_screen=True, window_size=(1100, 1100))
for k, ((fs, vs, Fl, _), X) in enumerate(zip(charts, UV)):
    col = np.array([int(PALETTE[k % len(PALETTE)][i:i + 2], 16) for i in (1, 3, 5)])
    tex = np.where(chk[..., None], col, (0.35 * col + 0.65 * 245)).astype(np.uint8)
    m = pv.PolyData(V[vs], np.hstack([np.full((len(Fl), 1), 3), Fl]).ravel())
    m.active_texture_coordinates = (X / scale * 3.0).astype(np.float32)   # 3 checker tiles across the atlas
    t = pv.Texture(tex)
    t.repeat = True
    p.add_mesh(m, texture=t, smooth_shading=True, ambient=0.35, diffuse=0.75, specular=0.1)
    edges = m.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
    p.add_mesh(edges, color="black", line_width=3)
p.set_background("white")
p.camera_position = [(0.4, 0.8, 5.2), (0, 0.25, 0), (0, 1, 0)]
img = p.screenshot(return_img=True)
p.close()
a = np.asarray(img)
ys, xs = np.nonzero((a < 250).any(-1))
plt.imsave(os.path.join(OUT, "atlas_3d.png"), a[ys.min() - 8:ys.max() + 8, xs.min() - 8:xs.max() + 8])
print("ok")
