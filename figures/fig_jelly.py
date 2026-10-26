"""Jelly gingerbread man dropped on the floor: implicit Euler as minimization,

    x_{n+1} = argmin_x  1/(2h^2) ||x - x~||_M^2 + E(x) + B(x),    x~ = x_n + h v_n + h^2 g,

with E Neo-Hookean (planar.MeshEnergy) and B an IPC-style log barrier against the floor. Each step
is solved by projected Newton with flip-free and floor-safe step limits. Writes a video (mesh +
its sigma-cloud) and a strip of key frames.

    python figures/fig_jelly.py <out-dir>
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from matplotlib.collections import PolyCollection

from distortion import style
from distortion.eigensystem import NeoHookean, signed_svd
from distortion.planar import MeshEnergy, dets, gingerbread, max_step

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
os.makedirs(OUT, exist_ok=True)
style.use()

V, F, G = gingerbread(max_area=0.006)
E = MeshEnergy(V, F, NeoHookean(mu=260.0, lam=520.0))
n = len(V)
m = np.zeros(n)
np.add.at(m, F.ravel(), np.repeat(E.A / 3, 3))           # lumped mass, density 1
M = np.repeat(m, 2)
FLOOR, DHAT, KAPPA = -3.0, 0.06, 2e3
grav = np.array([0.0, -9.8])
h = 1 / 120


def barrier(y):
    d = y - FLOOR
    on = d < DHAT
    dd = np.where(on, d, DHAT)
    L = np.log(dd / DHAT)
    b = np.where(on, -(dd - DHAT) ** 2 * L, 0)
    g = np.where(on, -2 * (dd - DHAT) * L - (dd - DHAT) ** 2 / dd, 0)
    H = np.where(on, -2 * L - 4 * (dd - DHAT) / dd + (dd - DHAT) ** 2 / dd**2, 0)
    return KAPPA * b.sum(), KAPPA * g, KAPPA * np.maximum(H, 0)


def step(X, Vel):
    Xt = X + h * Vel + h * h * grav
    Y = X.copy()
    for it in range(30):
        e, gE, Fd = E.grad(Y)
        b, gb, Hb = barrier(Y[:, 1])
        f = 0.5 * np.sum(M.reshape(-1, 2) * (Y - Xt) ** 2) / h**2 + e + b
        g = M.reshape(-1, 2) * (Y - Xt) / h**2 + gE
        g[:, 1] += gb
        Hc = np.zeros(2 * n)
        Hc[1::2] = Hb
        H = (sp.diags(M / h**2 + Hc) + E.hessian(Fd, "clamp")).tocsc()
        d = spla.spsolve(H, -g.ravel()).reshape(-1, 2)
        slope = float(np.sum(g * d))
        if -slope < 1e-10:
            break
        a = min(1.0, 0.9 * max_step(Y, d, F))
        down = d[:, 1] < 0
        if down.any():
            a = min(a, 0.9 * np.min((Y[down, 1] - FLOOR) / -d[down, 1]))
        tot = lambda Z: 0.5 * np.sum(M.reshape(-1, 2) * (Z - Xt) ** 2) / h**2 + E.value(Z) + barrier(Z[:, 1])[0]
        while tot(Y + a * d) > f + 1e-4 * a * slope and a > 1e-12:
            a *= 0.5
        Y = Y + a * d
        if np.abs(a * d).max() < 1e-7:
            break
    return Y, (Y - X) / h


FR = os.path.join(OUT, "jelly_frames.npy")
if os.path.exists(FR) and "--resim" not in sys.argv:
    frames = list(np.load(FR))
else:
    tilt = np.radians(4)
    R = np.array([[np.cos(tilt), -np.sin(tilt)], [np.sin(tilt), np.cos(tilt)]])
    X = V @ R.T + [0, 0.9]
    Vel = np.tile([0.0, -2.0], (n, 1))
    frames = [X.copy()]
    T = 2.4
    for k in range(int(T / h)):
        X, Vel = step(X, Vel)
        if k % 2 == 1:
            frames.append(X.copy())
        assert (dets(X, F) > 0).all() and (X[:, 1] > FLOOR).all()
    print("simulated", len(frames), "frames, min det", dets(X, F).min())
    np.save(os.path.join(OUT, "jelly_frames.npy"), np.array(frames))


def sigmas(X):
    Ds = np.stack([X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]], 2)
    _, s, _ = signed_svd(Ds @ E.Dinv)
    return s


def draw(axm, axs, X):
    axm.clear(); axs.clear()
    s = sigmas(X)
    J = s.prod(1)
    t = np.clip(np.log2(J) / 0.6, -1, 1)[:, None]
    lo, mid, hi = np.array([0.165, 0.471, 0.839]), np.array([0.94, 0.937, 0.925]), np.array([0.922, 0.408, 0.204])
    col = np.where(t < 0, mid + (lo - mid) * -t, mid + (hi - mid) * t)
    axm.add_collection(PolyCollection(X[F], fc=col, ec="none"))
    axm.axhspan(FLOOR - 1, FLOOR, color=style.GRID)
    axm.set_xlim(-3.2, 3.6); axm.set_ylim(FLOOR - 0.3, 3.4)
    axm.set_aspect("equal"); axm.axis("off")
    axs.scatter(s[:, 0], s[:, 1], s=4, color=style.BLUE, alpha=0.4, lw=0)
    axs.plot([1], [1], "o", color=style.ACCENT, ms=8)
    q = np.linspace(0.3, 2, 100)
    axs.plot(q, q, ":", color=style.MUTED, lw=1); axs.plot(q, 1 / q, ":", color=style.MUTED, lw=1)
    axs.set_xlim(0.4, 1.8); axs.set_ylim(0.4, 1.8); axs.set_aspect("equal")
    style.mirror_half(axs, 0.4, 1.8, fs=11)
    axs.set_xlabel(r"$\sigma_1$", fontsize=14); axs.set_ylabel(r"$\sigma_2$", fontsize=14)


fig, (axm, axs) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
import imageio_ffmpeg
w = None
for X in frames:
    draw(axm, axs, X)
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    if w is None:
        H_, W_ = img.shape[:2]
        H_, W_ = H_ - H_ % 2, W_ - W_ % 2
        w = imageio_ffmpeg.write_frames(os.path.join(OUT, "jelly.mp4"), (W_, H_), fps=60, quality=8, macro_block_size=1)
        w.send(None)
    w.send(np.ascontiguousarray(img[:H_, :W_]))
w.close()
plt.close(fig)

# key frames strip (for the PDF and the opening slide)
ys = np.array([f[:, 1].min() for f in frames])
hit = int(np.argmin(ys))
picks = [0, hit // 2, hit, min(len(frames) - 1, hit + (len(frames) - hit) // 3), len(frames) - 1]
fig, axs = plt.subplots(1, len(picks), figsize=(16, 4.6))
for ax, k in zip(axs, picks):
    X = frames[k]
    s = sigmas(X)
    t = np.clip(np.log2(s.prod(1)) / 0.6, -1, 1)[:, None]
    lo, mid, hi = np.array([0.165, 0.471, 0.839]), np.array([0.94, 0.937, 0.925]), np.array([0.922, 0.408, 0.204])
    col = np.where(t < 0, mid + (lo - mid) * -t, mid + (hi - mid) * t)
    ax.add_collection(PolyCollection(X[F], fc=col, ec="none"))
    ax.axhspan(FLOOR - 1, FLOOR, color=style.GRID)
    ax.set_xlim(-3.0, 3.4); ax.set_ylim(FLOOR - 0.3, 3.4); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"t = {2 * k * h:.2f} s", fontsize=15, color=style.INK2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "jelly_strip.png"))
print("hit frame", hit, "of", len(frames))
