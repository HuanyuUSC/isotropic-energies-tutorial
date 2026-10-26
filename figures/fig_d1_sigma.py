"""Act 1 figures: circle -> ellipse, the SVD as rotate-stretch-rotate, the sigma-plane
(master visual), map projections on it, and energy landscapes on it."""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch

from distortion import maps, style

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
os.makedirs(OUT, exist_ok=True)
style.use()

rot = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
th = np.linspace(0, 2 * np.pi, 200)
CIRCLE = np.c_[np.cos(th), np.sin(th)].T
U, SIG, V = rot(np.radians(35)), np.array([1.8, 0.6]), rot(np.radians(-25))
F = U @ np.diag(SIG) @ V.T


def shape(ax, M, title=None, labels=None):
    """Image of the unit disk under M, with the right singular vectors' images colored."""
    P = M @ CIRCLE
    ax.fill(P[0], P[1], color=style.GRID, zorder=0)
    ax.plot(P[0], P[1], color=style.INK2, lw=1.5)
    for v, c in [(V[:, 0], style.BLUE), (V[:, 1], style.ORANGE)]:
        a, b = M @ v, -(M @ v)
        ax.plot([b[0], 0], [b[1], 0], color=c, lw=3, solid_capstyle="round")
        # arrowhead on the + end: a rotation keeps the pair's handedness, a reflection would not
        ax.add_patch(FancyArrowPatch((0, 0), tuple(a), arrowstyle="-|>", mutation_scale=22, lw=3, color=c,
                                     shrinkA=0, shrinkB=0))
    if labels:
        for v, c, lab in [(V[:, 0], style.BLUE, labels[0]), (V[:, 1], style.ORANGE, labels[1])]:
            a = M @ v
            n = np.linalg.norm(a)
            ax.text(*(a * (1 + 0.32 / max(n, 0.3))), lab, color=c, fontsize=20, ha="center", va="center")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=18, pad=8)


# ---- circle -> ellipse
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
shape(axes[0], np.eye(2), "before", (r"$v_1$", r"$v_2$"))
shape(axes[1], F, r"after $x \mapsto Fx$", (r"$\sigma_1 u_1$", r"$\sigma_2 u_2$"))
fig.savefig(f"{OUT}/d1_circle_to_ellipse.png")
plt.close(fig)

# ---- SVD as rotate -> stretch -> rotate, as a 4-step build (same layout every step)
steps = [(np.eye(2), "unit circle"), (V.T, r"rotate: $V^\top$"),
         (np.diag(SIG) @ V.T, r"stretch: $\Sigma$"), (F, r"rotate: $U$")]
for k in range(4):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    for j, (M, t) in enumerate(steps):
        if j <= k:
            shape(axes[j], M, t, [(r"$v_1$", r"$v_2$"), (r"$e_1$", r"$e_2$"),
                                  (r"$\sigma_1 e_1$", r"$\sigma_2 e_2$"), (r"$\sigma_1 u_1$", r"$\sigma_2 u_2$")][j])
        else:
            axes[j].axis("off")
    fig.savefig(f"{OUT}/d1_svd_step{k}.png")
    plt.close(fig)

# ---- signed SVD: a flipped triangle keeps U, V rotations and puts the sign on sigma_2
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
for ax, sg, t in [(axes[0], SIG, r"$\sigma_2 > 0$: det F > 0"),
                  (axes[1], SIG * np.array([1, -1]), r"$\sigma_2 < 0$: flipped, det F < 0")]:
    M = U @ np.diag(sg) @ V.T
    shape(ax, M, None, (r"$\sigma_1 u_1$", r"$\sigma_2 u_2$"))
    ax.set_title(t, fontsize=19, pad=8, color=style.INK if sg[1] > 0 else style.ACCENT)
fig.savefig(f"{OUT}/d1_svd_flip.png")
plt.close(fig)

# ---- the sigma-plane (master visual)
LIM = 2.6


def sigma_axes(ax, glyphs=True, notes=True, fold=True):
    ax.set_xlim(0, LIM)
    ax.set_ylim(0, LIM)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\sigma_1$", fontsize=20)
    ax.set_ylabel(r"$\sigma_2$", fontsize=20, rotation=0, labelpad=14)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    if glyphs:
        for s1 in [0.4, 1.0, 1.6, 2.2]:
            for s2 in [0.4, 1.0, 1.6, 2.2]:
                ax.add_patch(Ellipse((s1, s2), 0.22 * s1, 0.22 * s2, fc=style.GRID, ec=style.INK2, lw=1))
    if notes:
        ax.plot(1, 1, "o", ms=11, color=style.ACCENT, zorder=5)
        ax.annotate("no distortion", (1, 1), (1.25, 0.62), fontsize=15, color=style.ACCENT,
                    arrowprops=dict(arrowstyle="-", color=style.ACCENT, lw=1))
        ax.text(0.05, LIM - 0.12, "collapse", fontsize=13, color=style.MUTED, va="top")
        ax.text(LIM - 0.05, 0.06, "collapse", fontsize=13, color=style.MUTED, ha="right")
    if fold:
        style.mirror_half(ax, 0, LIM)


fig, ax = plt.subplots(figsize=(6.2, 6.2))
sigma_axes(ax, fold=False)
fig.savefig(f"{OUT}/d1_sigma_plane.png")
plt.close(fig)

# the fold: (s1, s2) and (s2, s1) are the same ellipse turned 90 degrees
for k, name in enumerate(["d1_sigma_fold_pair", "d1_sigma_fold"]):
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    sigma_axes(ax, notes=False, fold=bool(k))
    ax.plot(1, 1, "o", ms=11, color=style.ACCENT, zorder=5)
    for (a, b) in [(1.6, 0.4), (0.4, 1.6)]:
        ax.add_patch(Ellipse((a, b), 0.22 * a, 0.22 * b, fc="none", ec=style.ACCENT, lw=3, zorder=6))
    ax.annotate("", (1.45, 0.5), (0.5, 1.45), zorder=7,
                arrowprops=dict(arrowstyle="<->", color=style.ACCENT, lw=2, connectionstyle="arc3,rad=-0.35"))
    ax.text(0.55, 0.62, "same ellipse,\nturned 90°", fontsize=14, color=style.ACCENT, zorder=7)
    fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig)

# map projections on the sigma-plane
lat = np.radians(np.linspace(0, 70, 200))
fig, ax = plt.subplots(figsize=(6.2, 6.2))
sigma_axes(ax, glyphs=False, notes=False, fold=False)
d = np.linspace(0.05, LIM, 2)
s = np.linspace(1 / LIM, LIM, 200)
ax.plot(d, d, ls="--", color=style.MUTED, lw=1.2)
ax.plot(s, 1 / s, ls=":", color=style.MUTED, lw=1.5)
ax.text(2.12, 2.3, "same shape", fontsize=13, color=style.MUTED, rotation=45)
ax.text(1.95, 0.33, "same area", fontsize=13, color=style.MUTED)
ax.plot(1, 1, "o", ms=13, color=style.ACCENT, zorder=4)
LABEL_OFFSET = {("Mercator", 30): (10, -10), ("Mercator", 60): (8, -14),
                ("Gall–Peters", 0): (8, 6), ("Gall–Peters", 60): (0, 0)}
for proj, c, name, degs in [(maps.mercator, style.BLUE, "Mercator", [30, 60]),
                            (maps.gall_peters, style.ORANGE, "Gall–Peters", [0, 60])]:
    s1, s2 = maps.SIGMAS[proj](lat)
    s1, s2 = np.maximum(s1, s2), np.minimum(s1, s2)          # sorted: sigma_1 >= sigma_2
    ax.plot(s1, s2, color=c, lw=3, label=name)
    for deg in degs:
        a, b = maps.SIGMAS[proj](np.radians(deg))
        a, b = max(a, b), min(a, b)
        ax.plot(a, b, "o", color=c, ms=8, mec="white", mew=1.5, zorder=6)
        if (name, deg) == ("Gall–Peters", 60):
            continue                                          # same point as the equator
        lab = "0° and 60°" if name == "Gall–Peters" else f"{deg}°"
        ax.annotate(lab, (a, b), LABEL_OFFSET[(name, deg)], textcoords="offset points",
                    fontsize=12, color=style.INK2)
ax.annotate("Mercator 0°\nGall–Peters 45°", (1, 1), (-118, 18), textcoords="offset points",
            fontsize=12, color=style.INK2, zorder=6)
style.mirror_half(ax, 0, LIM)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.86), fontsize=14)
fig.savefig(f"{OUT}/d1_sigma_maps.png")
plt.close(fig)

# ---- energy landscapes on the sigma-plane
g = np.linspace(0.04, LIM, 400)
S1, S2 = np.meshgrid(g, g)
J = S1 * S2
ENERGIES = {
    "conformal": (S1 / S2 + S2 / S1, r"$\sigma_1/\sigma_2+\sigma_2/\sigma_1$", "diag"),
    "area": ((J - 1) ** 2, r"$(\sigma_1\sigma_2-1)^2$", "hyper"),
    "arap": ((S1 - 1) ** 2 + (S2 - 1) ** 2, r"$\sum_i(\sigma_i-1)^2$", "point"),
    "symdir": (S1**2 + S1**-2 + S2**2 + S2**-2, r"$\sum_i\sigma_i^2+\sigma_i^{-2}$", "point"),
    "hencky": (np.log(S1) ** 2 + np.log(S2) ** 2, r"$\sum_i\log^2\sigma_i$", "point"),
    "neohookean": (0.5 * (S1**2 + S2**2 - 2) - np.log(J) + 0.5 * np.log(J) ** 2,
                   r"$\frac{1}{2}(\sum_i\sigma_i^2-2)-\log J+\frac{1}{2}\log^2 J$", "point"),
}
levels = np.geomspace(1e-2, 1e2, 13)
for key, (E, label, zero) in ENERGIES.items():
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    Z = E - E.min()
    cf = ax.contourf(S1, S2, np.clip(Z, levels[0], levels[-1]), levels=levels,
                     cmap=style.SEQ, norm=plt.matplotlib.colors.LogNorm(), extend="both")
    ax.contour(S1, S2, Z, levels=levels[::2], colors="white", linewidths=0.6, alpha=0.7)
    sigma_axes(ax, glyphs=False, notes=False, fold=False)
    style.mirror_half(ax, 0, LIM, label=False)
    if zero == "diag":
        ax.plot([0, LIM], [0, LIM], color=style.ACCENT, lw=3)
    elif zero == "hyper":
        ax.plot(s, 1 / s, color=style.ACCENT, lw=3)
    else:
        ax.plot(1, 1, "o", ms=11, color=style.ACCENT)
    ax.set_title(label, fontsize=18, pad=10)
    fig.savefig(f"{OUT}/d1_energy_{key}.png")
    plt.close(fig)
print("ok")


# the same, with every Tissot circle of the two maps (latitudes 0, +-30, +-60) drawn at its own stretches, NOT
# sorted: x = stretch along the parallel, y = along the meridian. A tall ellipse (y > x) stays in the mirror half,
# a wide one below the diagonal, so each map circle can be matched to its point by eye.
LIM2 = 4.2                                         # Mercator at 75 deg stretches by sec 75 = 3.86
lat2 = np.radians(np.linspace(0, 77, 300))
fig, ax = plt.subplots(figsize=(6.2, 6.2))
sigma_axes(ax, glyphs=False, notes=False, fold=False)
ax.set_xlim(0, LIM2); ax.set_ylim(0, LIM2); ax.set_xticks([0, 1, 2, 3, 4]); ax.set_yticks([0, 1, 2, 3, 4])
ax.plot([0.05, LIM2], [0.05, LIM2], ls="--", color=style.MUTED, lw=1.2)
s2 = np.linspace(1 / LIM2, LIM2, 300)
ax.plot(s2, 1 / s2, ls=":", color=style.MUTED, lw=1.5)
ax.text(2.75, 3.0, "same shape", fontsize=13, color=style.MUTED, rotation=45)
ax.text(3.2, 0.42, "same area", fontsize=13, color=style.MUTED)
ax.plot(1, 1, "o", ms=10, color=style.ACCENT, zorder=4)
LAB = {("Mercator", 0): ("0°", (-30, -22)), ("Mercator", 30): ("30°", (14, -6)), ("Mercator", 60): ("60°", (32, 16)),
       ("Gall–Peters", 0): ("0° (tall)", (12, 10)), ("Gall–Peters", 30): ("30°", (14, 4)),
       ("Gall–Peters", 60): ("60° (wide)", (-6, -34)), ("Mercator", 75): ("75°", (30, 22)),
       ("Gall–Peters", 75): ("75°", (-8, 10))}
for proj, c, name in [(maps.mercator, style.BLUE, "Mercator"), (maps.gall_peters, style.ORANGE, "Gall–Peters")]:
    a_c, b_c = maps.SIGMAS[proj](lat2)
    ax.plot(a_c, b_c, color=c, lw=2.2, alpha=0.5, label=name)
    for deg in (0, 30, 60, 75):
        a, b = maps.SIGMAS[proj](np.radians(deg))                 # (along the parallel, along the meridian)
        ax.add_patch(Ellipse((a, b), 0.16 * a, 0.16 * b, fc=c, ec=c, alpha=0.35, lw=1.5, zorder=5))
        t, off = LAB[(name, deg)]
        ax.annotate(t, (a, b), off, textcoords="offset points", fontsize=12, color=c, zorder=6)
style.mirror_half(ax, 0, LIM2)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.86), fontsize=14)
fig.savefig(f"{OUT}/d1_sigma_maps_tissot.png")
plt.close(fig)
