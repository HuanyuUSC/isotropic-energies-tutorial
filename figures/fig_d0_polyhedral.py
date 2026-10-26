"""Act 0/1 figures: a third way to lie, cut the globe. Icosahedral (per-face gnomonic) world map,
unfolded into a net whose cuts avoid land, with Tissot circles, plus the sigma clouds of
Mercator / Gall-Peters / icosahedral maps.

    python figures/fig_d0_polyhedral.py <out-dir>
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from distortion import maps, polyhedral as ph, style

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
os.makedirs(OUT, exist_ok=True)
style.use()
hexrgb = lambda h: np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)]) / 255

# ---- land as an equirectangular raster (0 sea, 1 land, 2 Greenland, 3 Africa)
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
LAB = np.asarray(lab)


def label_at(p):
    lon, lat = ph.xyz_to_lonlat(p)
    i = np.clip(((np.pi / 2 - lat) / np.pi * EH).astype(int), 0, EH - 1)
    j = np.clip(((lon + np.pi) / (2 * np.pi) * EW).astype(int), 0, EW - 1)
    return LAB[i, j]


PAL = np.array([hexrgb("#ffffff"), hexrgb(style.LAND), hexrgb(style.ORANGE), hexrgb(style.BLUE)])
SEA = hexrgb("#eef3f8")

# Tissot centers: the same grid as the other maps (every 30 deg lon, lat -60..60)
TC = np.array([ph.lonlat_to_xyz(lo, la) for la in np.radians([-60, -30, 0, 30, 60])
               for lo in np.radians(np.arange(-165, 181, 30))])
TR = np.radians(4)


def sampler(tissot):
    def f(p):
        c = PAL[label_at(p)]
        c[label_at(p) == 0] = SEA
        if tissot:
            near = (p @ TC.T).max(1) > np.cos(TR)
            c[near] = 0.55 * hexrgb(style.ACCENT) + 0.45 * c[near]
        return c
    return f


# ---- land sample points for choosing the net
rng = np.random.default_rng(1)
P = rng.normal(size=(60000, 3))
P /= np.linalg.norm(P, axis=1, keepdims=True)
land_pts = P[label_at(P) > 0]

V, F = ph.icosahedron()
cut_land, R, tree, pos, cuts = ph.best_net(V, F, land_pts, rng, n_rot=1500)
VR = V @ R.T
print(f"best net: land samples within 1.5 deg of a cut = {cut_land:.0f} of {len(land_pts)}")
np.savez(os.path.join(OUT, "d0_icosa_net.npz"), R=R, cuts=np.array(cuts))
# turn the net so that Africa's north points up (a net has no global "up")
c, nn = ph.to_net(VR, F, pos, np.array([ph.lonlat_to_xyz(np.radians(20), 0.0),
                                        ph.lonlat_to_xyz(np.radians(20), np.radians(5))]))
dv = nn - c
pos = ph.rotate_net(pos, np.pi / 2 - np.arctan2(dv[1], dv[0]))
np.savez(os.path.join(OUT, "d0_icosa_net.npz"), R=R, cuts=np.array(cuts), pos=np.array([pos[f] for f in range(len(F))]))

for tag, tis in [("", False), ("_tissot", True)]:
    img, ok, _, _ = ph.render(VR, F, pos, sampler(tis), res=2400)
    rgba = np.dstack([img, ok.astype(float)])
    Image.fromarray((rgba * 255).astype(np.uint8)).save(os.path.join(OUT, f"d0_icosa{tag}.png"))

# ---- Greenland vs Africa on the net (pixel counts = apparent areas)
img, ok, pts, _ = ph.render(VR, F, pos, sampler(False), res=2400)
labs = label_at(pts[ok])
print(f"icosahedral: Africa / Greenland apparent area = {(labs == 3).sum() / (labs == 2).sum():.1f} (true 13.9)")

# ---- distortion statistics over the whole globe (uniform samples)
s_ico = ph.gnomonic_sigmas(VR, F, P)
lat = np.arcsin(P[:, 2])
print(f"icosahedral: sigma1/sigma2 max {np.max(s_ico[:, 0] / s_ico[:, 1]):.2f}, "
      f"area max/min {np.max(s_ico.prod(1)) / np.min(s_ico.prod(1)):.2f}")

# ---- sigma plane: three maps as clouds of points (land-only, uniform on the globe)
fig, ax = plt.subplots(figsize=(6.4, 6.4))
m = label_at(P) > 0
latm = lat[m & (np.abs(lat) < np.radians(84))]
sm = np.stack([1 / np.cos(latm)] * 2, 1)
sg = np.stack([1 / (np.sqrt(2) * np.cos(latm)), np.sqrt(2) * np.cos(latm)], 1)
si = s_ico[m]
for S, c, name in [(sm, style.BLUE, "Mercator"), (sg, style.ORANGE, "Gall–Peters"), (si, style.AQUA, "icosahedral")]:
    k = rng.choice(len(S), min(len(S), 2500), replace=False)
    ax.scatter(S[k].max(1), S[k].min(1), s=4, color=c, alpha=0.35, lw=0, label=name)
ax.plot([1], [1], "o", color=style.ACCENT, ms=9, zorder=5)
ax.annotate("no distortion", (1, 1), (0.2, 1.6), color=style.ACCENT, fontsize=13,
            arrowprops=dict(arrowstyle="-", color=style.ACCENT))
ax.set_xlim(0, 4)
ax.set_ylim(0, 4)
style.mirror_half(ax, 0, 4)
ax.set_aspect("equal")
ax.set_xlabel(r"$\sigma_1$", fontsize=16)
ax.set_ylabel(r"$\sigma_2$", fontsize=16)
leg = ax.legend(loc="lower right", markerscale=4, frameon=False, fontsize=12)
for t in leg.get_texts():
    t.set_color(style.INK2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "d0_sigma_three_maps.png"))
