"""Poisson's ratio sweep figures: metrics vs nu, and cube-checker renders per nu."""
import glob, os, sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from distortion import style
from distortion.render import render_textured, subdivide

SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
style.use()
NUS = ["0.00", "0.10", "0.20", "0.30", "0.40", "0.49"]


def face_stats(V, S, F):
    def frame(P):
        e1, e2 = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
        x = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
        n = np.cross(e1, e2)
        y = np.cross(n / np.linalg.norm(n, axis=1, keepdims=True), x)
        return np.stack([np.stack([np.einsum("ij,ij->i", e, x), np.einsum("ij,ij->i", e, y)], 1) for e in (e1, e2)], 2)
    s = np.linalg.svd(frame(S[F]) @ np.linalg.inv(frame(V[F])), compute_uv=False)
    A = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    w = A / A.sum()
    la = np.log2(s[:, 0] * s[:, 1])
    spread = np.sqrt(w @ (la - w @ la) ** 2)
    return spread, float(w @ (s[:, 0] / s[:, 1]))


def cube_texture(n=4, h=1024):
    """Equirectangular image of a tri-color checkered cube, centrally projected (as in the viewer)."""
    w = 2 * h
    lon = (np.arange(w) + 0.5) / w * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi
    LON, LAT = np.meshgrid(lon, lat)
    e = np.stack([np.cos(LAT) * np.cos(LON), np.cos(LAT) * np.sin(LON), np.sin(LAT)], -1)
    a = np.abs(e)
    ax = np.argmax(a, -1)
    cols = np.array([[42, 120, 214], [235, 104, 52], [27, 175, 122]], float)
    img = np.empty(e.shape)
    for k, (i, j) in enumerate([(1, 2), (2, 0), (0, 1)]):
        m = ax == k
        u = e[..., i][m] / a[..., k][m]
        v = e[..., j][m] / a[..., k][m]
        chk = (np.floor((u + 1) / 2 * n) + np.floor((v + 1) / 2 * n)) % 2
        img[m] = np.where(chk[:, None] > 0, cols[k], [244, 243, 239])
    return img.astype(np.uint8)


from distortion.textures import cube_texture as _cube
TEX = _cube(checks=16, conformal=True)
CAMS = {"bunny": [(0.4, 0.8, 5.2), (0, 0.25, 0), (0, 1, 0)],
        "armadillo": [(0.04, -0.19, -6.0), (0.04, -0.19, 0), (0, 1, 0)]}

metrics = {}
for p in sorted(glob.glob(os.path.join(SRC, "*.npz"))):
    name = os.path.basename(p)[:-4]
    d = np.load(p)
    V, F = d["V"], d["F"]
    metrics[name] = np.array([face_stats(V, d[f"nu_{nu}"], F) for nu in NUS])
    if name in CAMS:
        row = []
        for nu in NUS:
            Vs, Ss, Fs = subdivide(V, d[f"nu_{nu}"], F, 1)
            img = render_textured(Vs, Ss, Fs, TEX, camera=CAMS[name], size=(700, 700))
            a = np.asarray(img)
            ys, xs = np.nonzero((a < 250).any(-1))
            row.append(a[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
        hh = max(r.shape[0] for r in row)
        row = [np.pad(r, ((0, hh - r.shape[0]), (20, 20), (0, 0)), constant_values=255) for r in row]
        Image.fromarray(np.concatenate(row, 1)).save(os.path.join(OUT, f"poisson_{name}_strip.png"))
        for nu, r in zip(NUS, row):
            Image.fromarray(r).save(os.path.join(OUT, f"poisson_{name}_nu{nu}.png"))

nus = [float(n) for n in NUS]
HIGHLIGHT = {"bunny": style.BLUE, "hand": style.ORANGE, "armadillo": style.AQUA}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for j, (ax, title) in enumerate(zip(axes, ["area distortion: spread of log₂(area ratio)", "shape distortion: mean σ₁/σ₂"])):
    for name, m in metrics.items():
        if name not in HIGHLIGHT:
            ax.plot(nus, m[:, j], "-", lw=1, color=style.MUTED, alpha=0.6)
    for name, c in HIGHLIGHT.items():
        if name in metrics:
            m = metrics[name]
            ax.plot(nus, m[:, j], "o-", lw=2.2, ms=6, color=c, mec="white", mew=1.2)
            ax.annotate(name, (nus[-1], m[-1, j]), (8, 0), textcoords="offset points", va="center", fontsize=12, color=style.INK2)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Poisson's ratio ν")
    ax.set_xticks(nus)
    ax.set_xlim(-0.02, 0.6)
axes[0].set_ylim(bottom=0)
axes[0].text(0.01, 0.06, "gray: other meshes", transform=axes[0].transAxes, fontsize=11, color=style.MUTED)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "poisson_metrics.png"))
for name, m in metrics.items():
    print(f"{name:22s} area-spread " + " ".join(f"{x:.2f}" for x in m[:, 0]) + "   shape " + " ".join(f"{x:.2f}" for x in m[:, 1]))
