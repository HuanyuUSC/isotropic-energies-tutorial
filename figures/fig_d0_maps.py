"""Act 0 figures: Mercator vs. equal-area, Greenland vs. Africa, Tissot's indicatrix."""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from distortion import maps, style

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
os.makedirs(OUT, exist_ok=True)
style.use()

LAT_RANGE = {"mercator": np.radians([-70, 84]), "gall_peters": np.radians([-90, 90])}
LAND = maps.land()
GREENLAND = maps.countries(lambda p: p["NAME"] == "Greenland")
AFRICA = maps.countries(lambda p: p["CONTINENT"] == "Africa")


def draw(ax, proj, lat_range, highlight=True, tissot=False):
    clip = lambda r: np.c_[r[:, 0], np.clip(r[:, 1], *lat_range)]
    for r in LAND:
        r = clip(r)
        ax.add_patch(Polygon(np.c_[proj(r[:, 0], r[:, 1])], fc=style.LAND, ec="none"))
    if highlight:
        for rings, c in [(GREENLAND, style.ORANGE), (AFRICA, style.BLUE)]:
            for outer, r in rings:
                if outer:
                    r = clip(r)
                    ax.add_patch(Polygon(np.c_[proj(r[:, 0], r[:, 1])], fc=c, ec="none"))
    if tissot:
        for lat in np.radians([-60, -30, 0, 30, 60]):
            for lon in np.radians(np.arange(-165, 181, 30)):
                lo, la = maps.small_circle(lon, lat, np.radians(4))
                ax.add_patch(Polygon(np.c_[proj(lo, la)], fc=style.ACCENT, alpha=0.45, ec=style.ACCENT, lw=0.8))
    x0, y0 = proj(np.array([-np.pi, np.pi]), lat_range)
    ax.set_xlim(*x0)
    ax.set_ylim(*y0)
    ax.set_aspect("equal")
    ax.axis("off")


def apparent_area(rings, proj):
    a = 0.0
    for outer, r in rings:
        x, y = proj(r[:, 0], r[:, 1])
        s = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        a += s if outer else -s
    return a


for name, proj in [("mercator", maps.mercator), ("gall_peters", maps.gall_peters)]:
    lat_range = LAT_RANGE[name]
    for tag, kw in [("", {}), ("_tissot", dict(tissot=True)), ("_tissot_plain", dict(tissot=True, highlight=False))]:
        x0, y0 = proj(np.array([-np.pi, np.pi]), lat_range)
        w = 12
        fig, ax = plt.subplots(figsize=(w, w * np.ptp(y0) / np.ptp(x0)))
        draw(ax, proj, lat_range, **kw)
        fig.savefig(f"{OUT}/d0_{name}{tag}.png")
        plt.close(fig)
    ratio = apparent_area(AFRICA, proj) / apparent_area(GREENLAND, proj)
    print(f"{name}: Africa / Greenland apparent area = {ratio:.2f}")
print(f"true: Africa / Greenland = {maps.area_km2(AFRICA) / maps.area_km2(GREENLAND):.1f}")
