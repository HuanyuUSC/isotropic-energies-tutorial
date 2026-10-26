"""Equirectangular textures for the static renders, matching the interactive viewer's shader:
the tri-color checkered cube, by central (gnomonic) or conformal (Rancic et al. 1996) projection.
Face layout and colors follow viewer_template.html (x-faces blue, y-faces orange, z-faces green)."""
import numpy as np

COLORS = np.array([[42, 120, 214], [235, 104, 52], [27, 175, 122]], float)
PAPER = np.array([244, 243, 239], float)
_LUT = {}


def _conformal_lut(N=257):
    if N not in _LUT:
        import os
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "figures"))
        from conformal_cube import table
        from scipy.interpolate import RegularGridInterpolator
        t = np.linspace(-1, 1, N)
        _LUT[N] = RegularGridInterpolator((t, t), table(N))       # (row = y, col = x)
    return _LUT[N]


def cube_texture(checks=16, conformal=True, h=2048):
    """(h, 2h, 3) uint8 equirectangular image of the checkered cube seen from its center."""
    w = 2 * h
    lon = (np.arange(w) + 0.5) / w * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi
    LON, LAT = np.meshgrid(lon, lat)
    e = np.stack([np.cos(LAT) * np.cos(LON), np.cos(LAT) * np.sin(LON), np.sin(LAT)], -1)
    a = np.abs(e)
    face = np.where((a[..., 0] >= a[..., 1]) & (a[..., 0] >= a[..., 2]), 0, np.where(a[..., 1] >= a[..., 2], 1, 2))
    img = np.empty(e.shape)
    for k, (i, j) in enumerate([(1, 2), (2, 0), (0, 1)]):       # shader: e.yz/|x|, e.zx/|y|, e.xy/|z|
        m = face == k
        u, v = e[..., i][m] / a[..., k][m], e[..., j][m] / a[..., k][m]
        if conformal:
            f = _conformal_lut()
            u, v = f(np.c_[v, u]), f(np.c_[u, v])                  # u = U(x, y), v = U(y, x)
        chk = (np.floor((u + 1) / 2 * checks) + np.floor((v + 1) / 2 * checks)) % 2
        img[m] = np.where(chk[:, None] > 0, COLORS[k], PAPER)
    return img.astype(np.uint8)
