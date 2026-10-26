"""The process behind the icosahedral map: globe -> icosahedron (central projection) -> net.
Same orientation and cuts as fig_d0_polyhedral.py (reads d0_icosa_net.npz). Cut edges in the
accent color, folded edges in gray, on all three pictures.

    python figures/fig_d0_icosa3d.py <out-dir>
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

from distortion import maps, polyhedral as ph, style
from distortion.render import render_textured, subdivide

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
hexrgb = lambda h: np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float)

net = np.load(os.path.join(OUT, "d0_icosa_net.npz"))
R, cuts = net["R"], [tuple(e) for e in net["cuts"]]
V, F = ph.icosahedron()
VR = V @ R.T
folds = [e for e in ph.edges_of(F) if e not in cuts]

# ---- equirectangular texture: sea / land / Greenland / Africa, then the edges on top
EW, EH = 4096, 2048
lab = Image.new("L", (EW, EH), 0)
d = ImageDraw.Draw(lab)
to_px = lambda r: [((lo + np.pi) / (2 * np.pi) * EW, (np.pi / 2 - la) / np.pi * EH) for lo, la in r]
for r in maps.land():
    d.polygon(to_px(r), fill=1)
for rings, v in [(maps.countries(lambda p: p["NAME"] == "Greenland"), 2),
                 (maps.countries(lambda p: p["CONTINENT"] == "Africa"), 3)]:
    for outer, r in rings:
        if outer:
            d.polygon(to_px(r), fill=v)
PAL = np.array([hexrgb("#eef3f8"), hexrgb(style.LAND), hexrgb(style.ORANGE), hexrgb(style.BLUE)])
tex = PAL[np.asarray(lab)]

lon = (np.arange(EW) + 0.5) / EW * 2 * np.pi - np.pi
lat = np.pi / 2 - (np.arange(EH) + 0.5) / EH * np.pi
for rows in np.array_split(np.arange(EH), 16):
    LO, LA = np.meshgrid(lon, lat[rows])
    p = ph.lonlat_to_xyz(LO, LA).reshape(-1, 3)
    for edges, width, color in [(folds, 0.35, hexrgb("#9a9a9a")), (cuts, 0.9, hexrgb(style.ACCENT))]:
        hit = np.zeros(len(p), bool)
        for a_, b_ in edges:
            a, b = VR[a_], VR[b_]
            n = np.cross(a, b)
            n /= np.linalg.norm(n)
            mid = (a + b) / np.linalg.norm(a + b)
            hit |= (np.abs(p @ n) < np.sin(np.radians(width))) & (p @ mid > a @ mid)
        t = tex[rows].reshape(-1, 3)
        t[hit] = color
        tex[rows] = t.reshape(len(rows), EW, 3)
tex = tex.astype(np.uint8)

# ---- camera: look at Africa / Europe
look = ph.lonlat_to_xyz(np.radians(20), np.radians(20))
cam = [tuple(5.2 * look), (0, 0, 0), (0, 0, 1)]

def crop(img):
    a = np.asarray(img)
    ys, xs = np.nonzero((a < 250).any(-1))
    return a[ys.min() - 10:ys.max() + 10, xs.min() - 10:xs.max() + 10]

# globe: a finely subdivided icosahedron projected to the sphere
Vs, Ss, Fs = subdivide(VR, VR, F, 5)
globe = crop(render_textured(Ss, Ss, Fs, tex, camera=cam, size=(1100, 1100)))
# icosahedron: flat faces, each showing its patch of the globe (midpoints stay on the face)
Vi, Si, Fi = subdivide(VR, VR, F, 5)
ico = crop(render_textured(Vi, Si, Fi, tex, camera=cam, size=(1100, 1100)))
Image.fromarray(globe).save(os.path.join(OUT, "d0_icosa_step_globe.png"))
Image.fromarray(ico).save(os.path.join(OUT, "d0_icosa_step_solid.png"))

# net: same layout as d0_icosa.png, sampled from the same texture (cuts land on its boundary)
pos = {f: T for f, T in enumerate(net["pos"])}
def sample(p):
    lo, la = ph.xyz_to_lonlat(p)
    i = np.clip(((np.pi / 2 - la) / np.pi * EH).astype(int), 0, EH - 1)
    j = np.clip(((lo + np.pi) / (2 * np.pi) * EW).astype(int), 0, EW - 1)
    return tex[i, j] / 255
img, ok, _, _ = ph.render(VR, F, pos, sample, res=1800)
rgba = np.dstack([img, ok.astype(float)])
Image.fromarray((rgba * 255).astype(np.uint8)).save(os.path.join(OUT, "d0_icosa_step_net.png"))
print("saved globe, icosahedron and net")
