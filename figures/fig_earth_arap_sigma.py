"""The ARAP Earth's stretch distribution on the sigma-plane: every triangle's principal stretches (sigma1 >= sigma2),
binned and weighted by its area on the globe. Reads assets/fig/d1_earth_arap_cut180_sigma.npz (fig_earth_arap.py 180)."""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from distortion import style

style.use()
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "fig")
D = np.load(os.path.join(FIG, "d1_earth_arap_cut180_sigma.npz"))
s, w = D["sigma"], D["area"] / D["area"].sum()
L = 2.2
fig, ax = plt.subplots(figsize=(5.4, 5.4))
# one dot per triangle, dot area proportional to the triangle's area on the globe. Color = where it is on the globe:
# hue = distance from Greenwich |lon| (the map is mirror-symmetric east-west), lightness = |lat| (north-south symmetric)
HUE = LinearSegmentedColormap.from_list("lon", ["#2a78d6", "#1baf7a", "#e8a21a", "#d6452a"])


def geo_color(alon, alat):
    c = HUE(np.asarray(alon) / 180.0)[..., :3]
    t = (np.asarray(alat) / 90.0)[..., None] ** 1.3 * 0.8
    return c * (1 - t) + t                                  # toward white at the poles


alon, alat = np.abs(D["lon"]), np.abs(D["lat"])
order = np.argsort(alat)[::-1]                              # poles first, low latitudes on top
ax.scatter(s[order, 0], s[order, 1], s=6.0 * w[order] / w.mean(), c=geo_color(alon[order], alat[order]),
           alpha=0.45, lw=0, zorder=3, rasterized=True)
# legend: the same colors on a longitude x latitude patch
cax = ax.inset_axes([0.25, 0.58, 0.36, 0.2])
gx, gy = np.meshgrid(np.linspace(0, 180, 181), np.linspace(0, 90, 91))
cax.imshow(geo_color(gx, gy), origin="lower", extent=(0, 180, 0, 90), aspect="auto")
cax.set_xticks([0, 90, 180]); cax.set_xticklabels(["0°", "90°", "180°"], fontsize=9)
cax.set_yticks([0, 45, 90]); cax.set_yticklabels(["0°", "45°", "90°"], fontsize=9)
cax.set_xlabel("|longitude| from Greenwich", fontsize=10, labelpad=1)
cax.set_ylabel("|latitude|", fontsize=10, labelpad=1)
for sp in cax.spines.values():
    sp.set_visible(False)
d = np.linspace(0, L, 10); t = np.linspace(1 / L, L, 300)
ax.plot(d, d, ls="--", color=style.MUTED, lw=1.2)
ax.plot(t, 1 / t, ls=":", color=style.MUTED, lw=1.5)
ax.text(1.72, 1.86, "same shape", fontsize=12, color=style.MUTED, rotation=45)
ax.text(1.7, 0.47, "same area", fontsize=12, color=style.MUTED)
ax.plot(1, 1, "o", ms=9, color=style.ACCENT, zorder=5)
ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal")
ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
ax.set_xlabel(r"$\sigma_1$", fontsize=20); ax.set_ylabel(r"$\sigma_2$", fontsize=20, rotation=0, labelpad=14)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
style.mirror_half(ax, 0, L)
fig.savefig(os.path.join(FIG, "d1_earth_arap_sigma.png"))
inside = w[(s[:, 0] < 1.3) & (s[:, 1] > 0.7)].sum()
print(f"area within sigma1<1.3, sigma2>0.7: {inside:.0%}; sigma1 {np.percentile(s[:,0],[5,50,95]).round(2)}, "
      f"sigma2 {np.percentile(s[:,1],[5,50,95]).round(2)}; off-plot area {w[s[:, 0] > L].sum():.2%}")
