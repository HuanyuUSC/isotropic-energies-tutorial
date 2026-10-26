"""Act 4 figures: the Earth painted on a bunny / armadillo through spherical maps.

For each mesh: the naive map (tetrahedron layout), the conformal map (cMCF), and the
distortion-minimizing map (Symmetric Dirichlet; SLIM -> projected Newton).  The globe is
rotated per map so Africa faces the camera, with north pointing "up" the body: the same
continent, shown where it is easiest to compare.  Plus Tissot circles and sigma-clouds.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from distortion import style
from distortion.meshes import genus0_mesh
from distortion.polytope_init import one_hole_map, polytope_map
from distortion.render import earth_texture, render_textured, subdivide
from distortion.sphere_param import conformal_flow_init, parameterize

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
CACHE = os.path.join(OUT, "cache")
os.makedirs(CACHE, exist_ok=True)
style.use()

SCENES = {   # camera, a body point facing the camera, and a point "above" it on the body
    "bunny": dict(cam=[(0.4, 0.8, 5.2), (0, 0.25, 0), (0, 1, 0)], face=(0.09, 0.08, 0.54), up=(0.09, 0.45, 0.45)),
    "armadillo": dict(cam=[(0.04, -0.19, -6.0), (0.04, -0.19, 0), (0, 1, 0)], face=(0.0, 0.1, -0.6), up=(0.0, 0.5, -0.6)),
    # meshes from the thesis website (F:\Dropbox\3Dmodels\parameterization), cached by figures/param_meshes.py
    "max": dict(look=dict(blend=0.45, shape=True), cam=[(-2.4, 0.5, -5.6), (0, -0.1, 0), (0, 1, 0)], face=(-0.02, -0.11, -1.09), up=(-0.02, 0.7, -0.85)),
    "skull": dict(look=dict(blend=0.45, shape=True), cam=[(2.2, 0.7, 5.6), (0, 0.1, 0), (0, 1, 0)], face=(0.09, 0.4, 0.78), up=(0.05, 1.1, 0.35)),
}
AFRICA = np.radians([20.0, 5.0])   # lon, lat


def maps_for(name):
    path = os.path.join(CACHE, f"{name}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["V"], d["F"], {k: d[k] for k in ("naive", "conformal", "symdir", "one_hole") if k in d}
    V, F = genus0_mesh(name)
    naive, _ = polytope_map(V, F, "tetrahedron")
    conformal = conformal_flow_init(V, F)
    symdir, _ = parameterize(V, F, naive, verbose=False)
    M = dict(naive=naive, conformal=conformal, symdir=symdir)
    if name == "bunny":   # the armadillo's limbs underflow double precision in this map
        M["one_hole"] = one_hole_map(V, F)
    np.savez(path, V=V, F=F, **M)
    return V, F, M


def sphere_point(V, S, p):
    """Spherical image of the mesh vertex nearest to p."""
    return S[np.argmin(np.linalg.norm(V - np.asarray(p), axis=1))]


def orient(V, S, face, up):
    """Rotation of the globe so that Africa lands at `face` with north toward `up`."""
    a = sphere_point(V, S, face)
    b = sphere_point(V, S, up)
    e2 = b - (b @ a) * a
    E = np.c_[a, e2 / np.linalg.norm(e2), np.cross(a, e2 / np.linalg.norm(e2))]
    lon, lat = AFRICA
    t1 = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    t2 = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])   # local north
    T = np.c_[t1, t2, np.cross(t1, t2)]
    return T @ E.T


def tissot_texture(tex, radius_deg=4.0):
    """The Earth texture with true small circles (Tissot's indicatrix) painted on it."""
    h, w, _ = tex.shape
    lon = (np.arange(w) + 0.5) / w * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi
    LON, LAT = np.meshgrid(lon, lat)
    P = np.stack([np.cos(LAT) * np.cos(LON), np.cos(LAT) * np.sin(LON), np.sin(LAT)], -1)
    mask = np.zeros((h, w), bool)
    for la in np.radians([-60, -30, 0, 30, 60]):
        for lo in np.radians(np.arange(-165, 181, 30)):
            c = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
            mask |= P @ c > np.cos(np.radians(radius_deg))
    out = tex.astype(float)
    acc = np.array([int(style.ACCENT[i:i + 2], 16) for i in (1, 3, 5)], float)
    out[mask] = 0.35 * out[mask] + 0.65 * acc
    return out.astype(np.uint8)


def crop(img, pad=12):
    a = np.asarray(img)
    ys, xs = np.nonzero((a < 250).any(-1))
    return a[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]


def sigmas(V, S, F):
    """Per-triangle singular values of the map mesh triangle -> chord triangle on the sphere."""
    def frame(P):
        e1, e2 = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
        x = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
        n = np.cross(e1, e2)
        y = np.cross(n / np.linalg.norm(n, axis=1, keepdims=True), x)
        return np.stack([np.stack([np.einsum("ij,ij->i", e, x), np.einsum("ij,ij->i", e, y)], 1) for e in (e1, e2)], 2)
    Dm, Ds = frame(V[F]), frame(S[F])
    return np.linalg.svd(Ds @ np.linalg.inv(Dm), compute_uv=False)


tex = earth_texture()
tex_t = tissot_texture(tex)
ONLY = sys.argv[2:]
for name, scene in SCENES.items():
    if ONLY and name not in ONLY:
        continue
    V, F, M = maps_for(name)
    for key, S in M.items():
        R = orient(V, S, scene["face"], scene["up"])
        Vs, Ss, Fs = subdivide(V, S, F, 1)
        for tag, t in [("", tex), ("_tissot", tex_t)]:
            img = render_textured(Vs, Ss, Fs, t, camera=scene["cam"], size=(1100, 1100), rotate=R,
                                **scene.get("look", {}))
            Image.fromarray(crop(img)).save(f"{OUT}/d4_{name}_{key}{tag}.png")
    # sigma-clouds on the sigma-plane (log axes: conformal maps span decades)
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    for key, c, label in [("conformal", style.BLUE, "conformal map"), ("symdir", style.AQUA, "Symmetric Dirichlet")]:
        s = sigmas(V, M[key], F)
        ax.scatter(s[:, 0], s[:, 1], s=3, color=c, alpha=0.35, lw=0, label=label, rasterized=True)
    g = np.geomspace(1e-3, 1e2, 2)
    ax.plot(g, g, ls="--", color=style.MUTED, lw=1)
    ax.plot(g, 1 / g, ls=":", color=style.MUTED, lw=1.3)
    ax.plot(1, 1, "o", ms=11, color=style.ACCENT, zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-2, 3e1)
    ax.set_ylim(1e-2, 3e1)
    style.mirror_half(ax, 1e-2, 3e1)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\sigma_1$ (largest stretch)", fontsize=16)
    ax.set_ylabel(r"$\sigma_2$", fontsize=18, rotation=0, labelpad=14)
    leg = ax.legend(frameon=False, loc="lower right", markerscale=5, fontsize=13)
    for h in leg.legend_handles:
        h.set_alpha(1)
    fig.savefig(f"{OUT}/d4_{name}_sigma_cloud.png")
    plt.close(fig)
    print(name, "done", flush=True)
