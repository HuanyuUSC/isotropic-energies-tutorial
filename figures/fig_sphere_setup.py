"""Spherical parameterization, the problem: the bunny (a closed genus-0 mesh) and its one-to-one map onto the unit
sphere (Symmetric Dirichlet, cached in assets/fig/cache/bunny.npz). Both colored by position on the bunny, so the
ears, head and body can be found on the sphere.
    python figures/fig_sphere_setup.py      -> assets/fig/aux_sphere_setup_{bunny,sphere}.png"""
import os

import numpy as np
import pyvista as pv

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "..", "assets", "fig")
d = np.load(os.path.join(FIG, "cache", "bunny.npz"))
V, F, S = d["V"], d["F"], d["symdir"]

# color by position on the bunny: a soft pastel cube of the normalized coordinates
P = (V - V.min(0)) / np.ptp(V, 0)
base = 0.45 + 0.5 * P[:, [0, 1, 2]]
# bake the bunny's own shading (its 3D vertex normals under a fixed light) into the colors, so its features
# (eyes, snout, ears) are visible wherever the map puts them, including on the sphere
from distortion.shade import vertex_normals
N = vertex_normals(V, F)
L = np.array([-0.85, 0.4, 0.45]); L /= np.linalg.norm(L)           # from the snout side
shade = 0.5 + 0.6 * np.clip(N @ L, 0, 1)
rgb = (np.clip(base * shade[:, None], 0, 1) * 255).astype(np.uint8)
faces = np.hstack([np.full((len(F), 1), 3), F]).ravel()


def shoot(X, cam, name):
    m = pv.PolyData(X, faces)
    m.point_data["rgb"] = rgb
    m = m.compute_normals(split_vertices=False, point_normals=True, cell_normals=False)
    p = pv.Plotter(off_screen=True, window_size=(1000, 1000))
    p.add_mesh(m, scalars="rgb", rgb=True, show_edges=True, edge_color=(0.62, 0.62, 0.62), line_width=0.1,
               smooth_shading=True, ambient=1.0, diffuse=0.0, specular=0.0)        # shading is baked in
    p.set_background("white")
    p.camera_position = cam
    img = p.screenshot(return_img=True)
    p.close()
    ys, xs = np.nonzero((img < 250).any(2))
    img = img[max(0, ys.min() - 8):ys.max() + 8, max(0, xs.min() - 8):xs.max() + 8]
    import matplotlib.pyplot as plt
    plt.imsave(os.path.join(FIG, name), img)


shoot(V, [(0.4, 0.8, 5.2), (0, 0.25, 0), (0, 1, 0)], "aux_sphere_setup_bunny.png")
# look at the sphere straight at the image of the bunny's face (the snout), with the ears' image toward "up"
head = V[:, 1] > np.quantile(V[:, 1], 0.6)
snout = np.flatnonzero(head)[np.argmin(V[head, 0])]                  # the bunny faces -x in this mesh
near = np.linalg.norm(V - V[snout], axis=1) < 0.06 * np.ptp(V, 0).max()
look = S[near].mean(0); look /= np.linalg.norm(look)
ears = S[V[:, 1] > np.quantile(V[:, 1], 0.97)].mean(0)
up = ears - (ears @ look) * look; up /= np.linalg.norm(up)
shoot(S, [tuple(4.6 * look), (0, 0, 0), tuple(up)], "aux_sphere_setup_sphere.png")
print("ok")
