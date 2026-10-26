"""Conformal hand painted with textures drawn directly on the sphere, to show "keeps angles" without the
cube's 8 corners (Gauss-Bonnet: a square grid on the sphere must have defects somewhere).
  mercator: checkerboard square in Mercator coordinates (lon, ln tan(pi/4 + lat/2)); right angles everywhere,
            all the curvature goes into the two poles, put at the wrist and the middle fingertip.
  dots:     evenly spread round dots (Fibonacci points); small circles map to small circles.
    python figures/fig_hand_textures.py"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from distortion.render import render_textured  # noqa: E402
from distortion.textures import COLORS, PAPER  # noqa: E402

ASSETS = os.path.join(HERE, "..", "..", "assets")
OUT = os.path.join(ASSETS, "fig")
H = 2048


def lonlat_grid(h=H):
    w = 2 * h
    lon = (np.arange(w) + 0.5) / w * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi
    return np.meshgrid(lon, lat)


def mercator_texture(n=64, fade=82):
    """Square checks in Mercator coordinates, n around the equator; fades to paper near the poles where
    the checks shrink below a texel."""
    LON, LAT = lonlat_grid()
    psi = np.log(np.tan(np.pi / 4 + np.clip(LAT, -1.5607, 1.5607) / 2))
    d = 2 * np.pi / n
    chk = (np.floor((LON + np.pi) / d) + np.floor(psi / d)) % 2
    band = np.floor(psi / d / 8) % 3                       # color bands every 8 checks along the meridian
    col = COLORS[band.astype(int)]
    img = np.where(chk[..., None] > 0, col, PAPER)
    t = np.clip((np.degrees(np.abs(LAT)) - fade) / (89.5 - fade), 0, 1)[..., None]
    return (img * (1 - t) + 0.5 * (col + PAPER) * t).astype(np.uint8)


def dots_texture(n=1500, radius_frac=0.32):
    """n round dots spread evenly on the sphere; radius = radius_frac x mean spacing."""
    k = np.arange(n) + 0.5
    z = 1 - 2 * k / n
    phi = np.pi * (1 + 5 ** 0.5) * k
    P = np.c_[np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi), z]
    LON, LAT = lonlat_grid()
    E = np.stack([np.cos(LAT) * np.cos(LON), np.cos(LAT) * np.sin(LON), np.sin(LAT)], -1).reshape(-1, 3)
    dist, idx = cKDTree(P).query(E)
    r = radius_frac * np.sqrt(4 * np.pi / n)
    inside = (dist < r).reshape(LON.shape)
    col = COLORS[(idx % 3).reshape(LON.shape)]
    return np.where(inside[..., None], col, PAPER).astype(np.uint8)


def frame(a, b):
    """Rotation taking unit vector a to the north pole (texture z) and keeping b's direction as lon 0."""
    z = a / np.linalg.norm(a)
    x = b - z * (b @ z); x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])                # rows: new axes -> S @ R.T gives new coords


def main():
    d = np.load(os.path.join(ASSETS, "viewer", "cache100", "hand.npz"))
    V, F = d["V"], d["F"]
    S = d["conformal"] / np.linalg.norm(d["conformal"], axis=1, keepdims=True)
    tip = S[V[:, 1].argmax()]
    wrist = S[V[:, 1] < V[:, 1].min() + 0.05].mean(0)
    R = frame(tip - wrist, np.array([0.0, 0.0, 1.0]))      # Mercator poles: middle fingertip (N), wrist (S)
    c = V.mean(0)
    cam = [tuple(c + np.array([0, 0.3, 6.0])), tuple(c + np.array([0, 0.05, 0])), (0, 1, 0)]
    for name, tex in [("mercator", mercator_texture()), ("dots", dots_texture())]:
        plt.imsave(os.path.join(OUT, f"tex_{name}.png"), tex[::4, ::4])
        img = render_textured(V, S, F, tex, camera=cam, size=(1000, 1000), rotate=R, shape=True)
        plt.imsave(os.path.join(OUT, f"aux_hand_conf_{name}.png"), img)
        print("wrote", f"aux_hand_conf_{name}.png")


if __name__ == "__main__":
    main()
