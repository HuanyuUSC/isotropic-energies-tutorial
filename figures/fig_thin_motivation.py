"""Motivation for the shell and rod projects.
  aux_thin_bend.png     a bent slab: the stretch along the mid-surface varies linearly across the thickness,
                        sigma(xi) = 1 + kappa xi; integrating psi over the cross section gives a bending energy
                        ~ kappa^2 h^3 (stiffness grows with thickness)
  aux_thin_shellmodes.png  the two through-thickness modes of the shell: linear (thickness stretch rho) and
                        quadratic (bending-induced thickness change zeta)
  aux_thin_rodmodes.png    the four cross-section modes of the rod (from the ANIME-Rod paper's figure, color bar cut)
    python figures/fig_thin_motivation.py <out-dir>"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image

from distortion import style

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
style.use()
CM = LinearSegmentedColormap.from_list("st", ["#2a78d6", "#d9d6cf", "#c0392b"])

# ---------------------------------------------------------------- bent slab
R, h, th = 2.2, 0.55, np.radians(70)                      # mid-surface radius, thickness, opening angle
fig = plt.figure(figsize=(8.6, 5.0))
ax = fig.add_axes([0.0, 0.0, 0.62, 1.0])
t = np.linspace(np.pi / 2 - th / 2, np.pi / 2 + th / 2, 200)
kappa = 1 / R
for xi in np.linspace(-h / 2, h / 2, 13):                 # fibers, colored by their stretch 1 + kappa xi
    r = R + xi
    pts = np.c_[r * np.cos(t), r * np.sin(t)]
    seg = np.stack([pts[:-1], pts[1:]], 1)
    ax.add_collection(LineCollection(seg, colors=[CM(Normalize(1 - kappa * h / 2, 1 + kappa * h / 2)(1 + kappa * xi))],
                                     linewidths=4.2 if abs(xi) > 1e-9 else 2.2, capstyle="butt"))
for a in (t[0], t[-1]):
    ax.plot([(R - h / 2) * np.cos(a), (R + h / 2) * np.cos(a)], [(R - h / 2) * np.sin(a), (R + h / 2) * np.sin(a)],
            color=style.INK, lw=1.6)
mid = np.c_[R * np.cos(t), R * np.sin(t)]
ax.plot(*mid.T, color=style.INK, lw=1.4, ls="--")
# one cross section, with the thickness coordinate xi
a0 = np.pi / 2
ax.annotate("", xy=((R + h / 2) * np.cos(a0), (R + h / 2) * np.sin(a0) + 0.02),
            xytext=((R - h / 2) * np.cos(a0), (R - h / 2) * np.sin(a0) - 0.02),
            arrowprops=dict(arrowstyle="<->", color=style.INK, lw=1.6))
ax.text(0.08, R + h / 2 + 0.12, r"thickness $h$", fontsize=15, color=style.INK, ha="left", va="bottom")
ax.text(-(R + 0.05) * np.sin(th / 2) - 0.1, (R) * np.cos(th / 2) - 0.35, "mid-surface\n(σ = 1)", fontsize=13,
        color=style.INK2, ha="right")
ax.text(1.55, 2.55, "outside: stretched", fontsize=14, color="#c0392b", ha="left")
ax.text(1.0, 1.35, "inside: compressed", fontsize=14, color="#2a78d6", ha="left")
ax.set_xlim(-2.0, 2.7); ax.set_ylim(1.0, 3.2); ax.set_aspect("equal"); ax.axis("off")
# stretch across the thickness
ax2 = fig.add_axes([0.70, 0.22, 0.27, 0.62])
xi = np.linspace(-0.5, 0.5, 50)
ax2.plot(1 + 0.25 * xi, xi, color=style.INK, lw=2.4)
ax2.scatter(1 + 0.25 * xi[::5], xi[::5], c=CM(np.linspace(0, 1, 10)), s=40, zorder=3)
ax2.axvline(1, color=style.GRID, lw=1)
ax2.set_yticks([-0.5, 0, 0.5]); ax2.set_yticklabels([r"$-\frac{h}{2}$", "0", r"$\frac{h}{2}$"], fontsize=14)
ax2.set_xticks([1]); ax2.set_xticklabels(["1"], fontsize=13)
ax2.set_xlabel(r"stretch $\sigma = 1 + \kappa\,\xi$", fontsize=14)
ax2.set_ylabel(r"$\xi$", fontsize=16, rotation=0, labelpad=10)
for sp in ("top", "right"):
    ax2.spines[sp].set_visible(False)
fig.savefig(os.path.join(OUT, "aux_thin_bend.png"))
plt.close(fig)

# ---------------------------------------------------------------- shell: two through-thickness modes
fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.9))
xi = np.linspace(-0.5, 0.5, 100)
for ax, (phi, lab, c) in zip(axs, [(xi, r"$\rho$: thickness stretch", style.BLUE),
                                   (4 * (xi ** 2 - 1 / 12), r"$\zeta$: bending thickness", style.ACCENT)]):
    ax.axhspan(-0.5, 0.5, color="#ecebe6", zorder=0)
    ax.axvline(0, color=style.MUTED, lw=1)
    ax.plot(phi, xi, color=c, lw=3)
    ax.set_xlim(-0.65, 0.65); ax.set_ylim(-0.62, 0.62)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(lab, fontsize=14, color=c, pad=4)
axs[0].text(-0.62, 0.5, r"$\frac{h}{2}$", fontsize=13, va="center")
axs[0].text(-0.62, -0.5, r"$-\frac{h}{2}$", fontsize=13, va="center")
fig.tight_layout(w_pad=1.5)
fig.savefig(os.path.join(OUT, "aux_thin_shellmodes.png"))
plt.close(fig)

# ---------------------------------------------------------------- rod: the paper's four modes, color bar cut off
im = np.asarray(Image.open(r"F:/Dropbox/writeup/fcurves/fig/ablation/deformationFields.png").convert("RGB"))
im = im[60:290]
ys, xs = np.nonzero((im < 245).any(-1))
Image.fromarray(im[ys.min() - 4:ys.max() + 4, xs.min() - 4:xs.max() + 4]).save(os.path.join(OUT, "aux_thin_rodmodes.png"))
print("ok")
