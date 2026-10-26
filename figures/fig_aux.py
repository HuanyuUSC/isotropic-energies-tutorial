"""Auxiliary panels so that no slide is text-only (slide standard rule 18b).

Each function writes one PNG named aux_<what>.png. Most are schematics (matplotlib); a few
reuse the real gingerbread mesh, energies and renders.

    python figures/fig_aux.py <out-dir> [name ...]     # all panels, or only the named ones
"""
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon

from distortion import style
from distortion.eigensystem import ARAP, SymmetricDirichlet, WithVolume, hessian_eigensystem, projected_hessian, signed_svd

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
ONLY = set(sys.argv[2:])
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)
style.use()
PANELS = []


def panel(f):
    PANELS.append(f)
    return f


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"aux_{name}.png"))
    plt.close(fig)


def clean(ax):
    ax.set_aspect("equal")
    ax.axis("off")


def arrow(ax, a, b, color=None, lw=2.2, ms=18, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=ms, lw=lw,
                                 color=color or style.INK2, shrinkA=0, shrinkB=0, **kw))


def ellipse_pts(F, n=200):
    t = np.linspace(0, 2 * np.pi, n)
    return (F @ np.stack([np.cos(t), np.sin(t)])).T


def sigma_plane(ax, lim=3.0, curves=True):
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\sigma_1$", fontsize=16)
    ax.set_ylabel(r"$\sigma_2$", fontsize=16)
    if curves:
        s = np.linspace(1 / lim, lim, 300)
        ax.plot(s, s, ":", color=style.MUTED, lw=1)
        ax.plot(s, 1 / s, ":", color=style.MUTED, lw=1)
    ax.plot([1], [1], "o", color=style.ACCENT, ms=9, zorder=6)
    style.mirror_half(ax, 0, lim, fs=11)


def box(ax, xy, w, h, text, fc, ec=None, fs=15, tc=None):
    ax.add_patch(FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec or fc, lw=1.5))
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=fs, color=tc or style.INK)


# ------------------------------------------------------------------ Act 0
@panel
def gores():
    """Twelve globe gores: a sphere lies flat only once it is cut (or stretched)."""
    from shapely.geometry import Polygon as SP, box as sbox
    from distortion import maps
    land = [SP(np.degrees(r)).buffer(0) for r in maps.land()]
    fig, ax = plt.subplots(figsize=(13, 4.6))
    W = 30
    for k in range(12):
        lon0 = -180 + W / 2 + k * W
        lat = np.linspace(-90, 90, 200)
        half = W / 2 * np.cos(np.radians(lat))
        xo = k * (W + 3)
        ax.add_patch(Polygon(np.c_[np.r_[xo - half, xo + half[::-1]], np.r_[lat, lat[::-1]]],
                             fc="#eef3f8", ec=style.GRID, lw=0.8))
        clip = sbox(lon0 - W / 2, -90, lon0 + W / 2, 90)
        for g in land:
            g = g.intersection(clip)
            for p in getattr(g, "geoms", [g]):
                if p.is_empty or p.geom_type != "Polygon":
                    continue
                lo, la = np.asarray(p.exterior.coords).T
                ax.add_patch(Polygon(np.c_[xo + (lo - lon0) * np.cos(np.radians(la)), la], fc=style.LAND, ec="none"))
    ax.set_xlim(-W, 12 * (W + 3))
    ax.set_ylim(-92, 92)
    clean(ax)
    save(fig, "gores")


# ------------------------------------------------------------------ Act 1
@panel
def ellipse_axes():
    """Unit circle -> ellipse with semi-axes sigma_1 u_1 and sigma_2 u_2."""
    th = np.radians(30)
    U = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    F = U @ np.diag([1.7, 0.75])
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, M, lab in [(axs[0], np.eye(2), "tiny circle on the globe"), (axs[1], F, "its image: Tissot's ellipse")]:
        P = ellipse_pts(M)
        ax.add_patch(Polygon(P, fc=style.LAND, ec=style.INK2, lw=1.8))
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1.7, 1.7)
        clean(ax)
        ax.set_title(lab, fontsize=15, color=style.INK2)
    for k, (c, name) in enumerate([(style.BLUE, r"$v_1$"), (style.ORANGE, r"$v_2$")]):
        e = np.eye(2)[:, k]                                    # here V = I, so v_i = e_i
        arrow(axs[0], (0, 0), tuple(e), color=c, lw=2.6)
        axs[0].text(*(e * 1.28 + [0.0, 0.0]), name, color=c, fontsize=18, ha="center", va="center")
    for k, (s, c) in enumerate([(1.7, style.BLUE), (0.75, style.ORANGE)]):
        arrow(axs[1], (0, 0), tuple(U[:, k] * s), color=c, lw=2.6)
        axs[1].text(*(U[:, k] * s * 1.22 + [[0.05, 0.05], [-0.4, 0.05]][k]), [r"$\sigma_1 u_1$", r"$\sigma_2 u_2$"][k], color=c, fontsize=18)
    save(fig, "ellipse_axes")


@panel
def svd_algebra():
    """[F] = [U][Sigma][V^T] as colored blocks, each with a glyph of what it holds:
    U and V^T are one angle each (a dial: U = exp(theta J)), Sigma is two stretches. 4 = 1 + 2 + 1."""
    th_u, th_v, s = np.radians(35), np.radians(-25), np.array([1.8, 0.6])
    rot = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    F = rot(th_u) @ np.diag(s) @ rot(th_v).T
    fig, ax = plt.subplots(figsize=(10, 4.4))

    def mat(x, label, diag=False, c=style.BLUE):
        for i in range(2):
            for j in range(2):
                on = (not diag) or i == j
                ax.add_patch(plt.Rectangle((x + j * 0.5, 0.5 - i * 0.5), 0.46, 0.46,
                                           fc=c if on else "white", ec=style.GRID, alpha=0.85 if on else 1))
        ax.text(x + 0.48, -0.32, label, ha="center", fontsize=17, color=style.INK2)

    def dial(cx, ang, c, lab):
        """A rotation by ang: unit circle, the turned frame (two arrows), and the angle arc."""
        cy, r = -1.35, 0.42
        t = np.linspace(0, 2 * np.pi, 100)
        ax.plot(cx + r * np.cos(t), cy + r * np.sin(t), color=style.GRID, lw=1.5)
        ax.plot([cx, cx + r * 1.05], [cy, cy], color=style.MUTED, lw=1, ls=(0, (3, 2)))
        R = rot(ang)
        for k, lw in [(0, 3), (1, 2)]:
            d = R[:, k] * r
            arrow(ax, (cx, cy), (cx + d[0], cy + d[1]), color=c, lw=lw, ms=14)
        a = np.linspace(0, ang, 30)
        ax.plot(cx + 0.22 * np.cos(a), cy + 0.22 * np.sin(a), color=style.INK2, lw=1.5)
        ax.text(cx + 0.2, cy - 0.17, lab, fontsize=19, color=style.INK2, ha="center", va="center")

    mat(0, "F", c=style.MUTED)
    ax.text(1.35, 0.48, "=", fontsize=30, ha="center", va="center")
    mat(1.8, "U  (rotation)", c=style.BLUE)
    mat(3.4, "Σ  (stretch)", diag=True, c=style.ACCENT)
    mat(5.0, "Vᵀ  (rotation)", c=style.ORANGE)
    # glyphs (second row)
    P = ellipse_pts(F) * 0.3
    ax.fill(0.48 + P[:, 0], -1.35 + P[:, 1], color=style.LAND)
    ax.plot(0.48 + P[:, 0], -1.35 + P[:, 1], color=style.INK2, lw=1.5)
    dial(2.28, th_u, style.BLUE, r"$\theta_U$")
    E = ellipse_pts(np.diag(s)) * 0.3
    ax.fill(3.88 + E[:, 0], -1.35 + E[:, 1], color=style.LAND)
    ax.plot(3.88 + E[:, 0], -1.35 + E[:, 1], color=style.INK2, lw=1.2)
    ax.plot([3.88, 3.88 + 0.3 * s[0]], [-1.35, -1.35], color=style.ACCENT, lw=3)
    ax.plot([3.88, 3.88], [-1.35, -1.35 + 0.3 * s[1]], color=style.ACCENT, lw=3)
    ax.text(3.88 + 0.3 * s[0] + 0.05, -1.35, "σ₁", fontsize=15, color=style.ACCENT, va="center")
    ax.text(3.88, -1.35 + 0.3 * s[1] + 0.12, "σ₂", fontsize=15, color=style.ACCENT, ha="center")
    dial(5.48, -th_v, style.ORANGE, r"$-\theta_V$")
    # counts (third row)
    for x, txt, c in [(0.48, "4 numbers", style.INK2), (1.35, "=", style.INK2), (2.28, "1 angle", style.BLUE),
                      (3.08, "+", style.INK2), (3.88, "2 stretches", style.ACCENT), (4.68, "+", style.INK2),
                      (5.48, "1 angle", style.ORANGE)]:
        ax.text(x, -2.15, txt, ha="center", fontsize=17, color=c, weight="bold" if txt not in "=+" else "normal")
    ax.set_xlim(-0.2, 6.2)
    ax.set_ylim(-2.35, 1.1)
    clean(ax)
    save(fig, "svd_algebra")


@panel
def signed_svd():
    """Backup: the signed-SVD convention on a flipped triangle. Standard SVD (sigma >= 0) hides the flip in a
    reflection V; negating v2 and sigma2 makes U, V rotations and puts the flip in sigma2 < 0."""
    rot = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    U, V = rot(np.radians(35)), rot(np.radians(-25))
    R = np.diag([1.0, -1.0])
    rows = [("standard SVD: σ ≥ 0", U, np.array([1.8, 0.6]), V @ R),
            ("signed SVD: U, V rotations", U, np.array([1.8, -0.6]), V)]
    GOOD, BAD = style.AQUA, "#d6402f"
    fig, ax = plt.subplots(figsize=(11, 6.2))

    def frame(cx, cy, M, name):
        r = 0.55
        for k, c in [(0, style.BLUE), (1, style.ORANGE)]:
            d = M[:, k] * r
            arrow(ax, (cx, cy), (cx + d[0], cy + d[1]), color=c, lw=3, ms=16)
            ax.text(cx + d[0] * 1.35, cy + d[1] * 1.35, "%s%s" % (name, "₁₂"[k]), color=c, fontsize=16,
                    ha="center", va="center")
        det = np.linalg.det(M)
        a0 = np.arctan2(M[1, 0], M[0, 0])
        a = np.linspace(a0, a0 + np.sign(det) * np.pi / 2, 30)
        ok = det > 0
        ax.plot(cx + 0.25 * np.cos(a), cy + 0.25 * np.sin(a), color=GOOD if ok else BAD, lw=2)
        ax.text(cx, cy - 1.05, "rotation" if ok else "reflection", ha="center", fontsize=16,
                color=GOOD if ok else BAD, weight="bold")

    def sig(cx, cy, s):
        k = 0.5
        arrow(ax, (cx - 0.3, cy), (cx - 0.3 + k * s[0], cy), color=style.ACCENT, lw=3, ms=16)
        c2 = style.ACCENT if s[1] > 0 else BAD
        arrow(ax, (cx - 0.3, cy), (cx - 0.3, cy + k * s[1]), color=c2, lw=3, ms=16)
        ax.text(cx - 0.3 + k * s[0] + 0.1, cy, "σ₁", color=style.ACCENT, fontsize=16, va="center")
        ax.text(cx - 0.15, cy + k * s[1] + (0.1 if s[1] > 0 else -0.1), "σ₂ = %+.1f" % s[1], fontsize=16,
                va="center", color=c2)

    for x, t in [(0.3, "U"), (2.5, "Σ"), (4.7, "V")]:
        ax.text(x, 3.2, t, fontsize=22, ha="center", color=style.INK2)
    for i, (title, Ui, Si, Vi) in enumerate(rows):
        y = 1.6 - i * 3.0
        ax.text(-1.25, y + 0.95, title, fontsize=19, color=style.INK, weight="bold", ha="left")
        frame(0.3, y, Ui, "u")
        sig(2.5, y, Si)
        frame(4.7, y, Vi, "v")
    ax.annotate("", xy=(4.7, -0.55), xytext=(4.7, 0.3),
                arrowprops=dict(arrowstyle="-|>", lw=2, color=style.INK2, mutation_scale=20))
    ax.text(4.85, -0.1, "negate v₂ and σ₂", fontsize=16, color=style.INK2, va="center")
    ax.set_xlim(-1.3, 6.6)
    ax.set_ylim(-2.6, 3.6)
    clean(ax)
    save(fig, "signed_svd")


@panel
def svd_basis():
    """F = sigma_1 u_1 v_1^T + sigma_2 u_2 v_2^T: two rank-one maps add up to the ellipse."""
    th, ph_ = np.radians(30), np.radians(-20)
    R = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    U, V = R(th), R(ph_)
    s = [1.7, 0.75]
    terms = [s[k] * np.outer(U[:, k], V[:, k]) for k in range(2)]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.0))
    for ax, M, lab, c in [(axs[0], terms[0], r"$\sigma_1\, u_1 v_1^\top$", style.BLUE),
                          (axs[1], terms[1], r"$\sigma_2\, u_2 v_2^\top$", style.ORANGE),
                          (axs[2], terms[0] + terms[1], r"$F$", style.INK2)]:
        P = ellipse_pts(M)
        ax.plot(*ellipse_pts(np.eye(2)).T, color=style.GRID, lw=1)
        ax.add_patch(Polygon(P, fc=style.LAND if c == style.INK2 else "none", ec=c, lw=3, closed=True))
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1.8, 1.8)
        clean(ax)
        ax.set_title(lab, fontsize=20, color=c)
    fig.text(0.355, 0.5, "+", fontsize=34, ha="center", va="center")
    fig.text(0.675, 0.5, "=", fontsize=34, ha="center", va="center")
    save(fig, "svd_basis")


@panel
def rotated_ellipses():
    """The same ellipse, rotated three ways: same distortion?"""
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, a in zip(axs, [0, 50, 110]):
        t = np.radians(a)
        Rm = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        ax.add_patch(Polygon(ellipse_pts(Rm @ np.diag([1.6, 0.6])), fc=style.LAND, ec=style.INK2, lw=2))
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        clean(ax)
    fig.text(0.36, 0.5, "=", fontsize=34, ha="center", va="center", color=style.ACCENT)
    fig.text(0.64, 0.5, "=", fontsize=34, ha="center", va="center", color=style.ACCENT)
    save(fig, "rotated_ellipses")


@panel
def sigma_goals():
    """Master visual, annotated with the three goals."""
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    sigma_plane(ax, curves=False)
    s = np.linspace(0.34, 3, 300)
    ax.plot(s, s, color=style.BLUE, lw=3)
    ax.plot(s, 1 / s, color=style.ORANGE, lw=3)
    ax.text(1.75, 2.45, "keep angles\nσ₁ = σ₂", color=style.BLUE, fontsize=15, ha="center")
    ax.text(2.45, 0.08, "keep area\nσ₁σ₂ = 1", color=style.ORANGE, fontsize=15, ha="center")
    ax.annotate("keep everything\n(1, 1)", (1, 1), (1.85, 1.15), color=style.ACCENT, fontsize=15,
                arrowprops=dict(arrowstyle="-", color=style.ACCENT))
    fig.tight_layout()
    save(fig, "sigma_goals")


# ------------------------------------------------------------------ Act 2
def gb():
    from distortion.planar import gingerbread
    V, F, g = gingerbread()
    X = pickle.load(open(os.path.join(HERE, "..", "scratch", "act2.pkl"), "rb"))["raise_arap"]
    return V, F, g, X


def draw_mesh(ax, V, F, fc=style.LAND, lw=0.25):
    from matplotlib.collections import PolyCollection
    ax.add_collection(PolyCollection(V[F], fc=fc, ec=style.MUTED, lw=lw))
    ax.autoscale_view()
    clean(ax)


@panel
def gb_drag():
    """Feet pinned, hand dragged: what should the rest do?"""
    V, F, g, X = gb()
    fig, ax = plt.subplots(figsize=(5.6, 6.2))
    draw_mesh(ax, V, F)
    ax.plot(*V[g["feet"]].T, "s", color=style.INK, ms=4)
    h = V[g["right_hand"]].mean(0)
    arrow(ax, tuple(h), tuple(X[g["right_hand"]].mean(0)), color=style.ACCENT, lw=3, ms=26,
          connectionstyle="arc3,rad=0.35")
    ax.text(0, -2.25, "pinned", ha="center", fontsize=14, color=style.INK)
    ax.text(1.7, 1.9, "drag", fontsize=16, color=style.ACCENT)
    ax.set_xlim(-2, 2.3)
    ax.set_ylim(-2.4, 2.4)
    save(fig, "gb_drag")


@panel
def gb_twist():
    """Feet and hands pinned, the head turned half a turn: the setup of the twist experiment."""
    from matplotlib.patches import FancyArrowPatch
    V, F, g, X = gb()
    head = pickle.load(open(os.path.join(HERE, "..", "scratch", "twist.pkl"), "rb"))["head"]
    fig, ax = plt.subplots(figsize=(5.6, 6.2))
    draw_mesh(ax, V, F)
    for k in ("feet", "left_hand", "right_hand"):
        ax.plot(*V[g[k]].T, "s", color=style.INK, ms=4)
    ax.plot(*V[head].T, "o", color=style.ACCENT, ms=2.5, alpha=0.8)
    c = V[head].mean(0)
    r = 0.75 * np.ptp(V[head][:, 0])
    ax.add_patch(FancyArrowPatch((c[0] + r, c[1]), (c[0] - r, c[1]), connectionstyle="arc3,rad=1.0",     # counterclockwise
                                 arrowstyle="-|>", mutation_scale=26, lw=3, color=style.ACCENT, zorder=6))
    ax.text(c[0], c[1] + r + 0.45, "turn the head 180°", ha="center", fontsize=16, color=style.ACCENT)
    ax.text(0, -2.25, "feet and hands pinned", ha="center", fontsize=14, color=style.INK)
    ax.set_xlim(-2, 2.3)
    ax.set_ylim(-2.4, 3.2)
    save(fig, "gb_twist")


@panel
def gb_cloud():
    """A deformed shape is a cloud of points on the sigma-plane."""
    V, F, g, X = gb()
    Dm = np.stack([V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]], 2)
    Ds = np.stack([X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]], 2)
    _, s, _ = signed_svd(Ds @ np.linalg.inv(Dm))
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw=dict(width_ratios=[1, 1.05]))
    draw_mesh(axs[0], X, F)
    sigma_plane(axs[1], lim=2.0)
    axs[1].scatter(s[:, 0], s[:, 1], s=6, color=style.BLUE, alpha=0.45, lw=0)
    fig.text(0.49, 0.5, "→", fontsize=40, ha="center", va="center", color=style.INK2)
    fig.tight_layout()
    save(fig, "gb_cloud")


@panel
def rotation_penalty():
    """A rotated square: ||F - I||^2 = 4 (penalized!), ARAP = 0."""
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float) - 0.5
    Rm = np.array([[0, -1], [1, 0]])
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    axs[0].add_patch(Polygon(sq, fc=style.LAND, ec=style.INK2, lw=2))
    axs[0].plot(*sq[[0, 1]].T, color=style.BLUE, lw=5)
    axs[1].add_patch(Polygon(sq @ Rm.T, fc=style.LAND, ec=style.INK2, lw=2))
    axs[1].plot(*(sq[[0, 1]] @ Rm.T).T, color=style.BLUE, lw=5)
    for ax, t in zip(axs, ["rest", "rotated 90°:  F = R"]):
        ax.set_xlim(-1, 1)
        ax.set_ylim(-0.9, 0.9)
        ax.set_title(t, fontsize=15, color=style.INK2)
        clean(ax)
    fig.subplots_adjust(right=0.72)
    fig.text(0.86, 0.58, r"$\|F - I\|^2 = 4$", fontsize=19, color="#c0392b", ha="center")
    fig.text(0.86, 0.38, r"ARAP $= 0$", fontsize=19, color=style.AQUA, ha="center")
    save(fig, "rotation_penalty")


# ------------------------------------------------------------------ Act 3
def f1d(x):
    return 0.08 * x**4 - 0.6 * x**2 + 0.3 * x + 2


def df(x):
    return 0.32 * x**3 - 1.2 * x + 0.3


def d2f(x):
    return 0.96 * x**2 - 1.2


@panel
def newton_1d():
    """Newton: fit a parabola (gradient + Hessian), jump to its bottom. A single-well, non-quadratic E,
    so the figure says nothing about local vs global minima (the double well is kept for newton_uphill)."""
    x = np.linspace(-3.2, 3.2, 400)
    c = -0.6
    f1d = lambda t: 0.03 * (t - c) ** 4 + 0.2 * (t - c) ** 2 + 0.4
    df = lambda t: 0.12 * (t - c) ** 3 + 0.4 * (t - c)
    d2f = lambda t: 0.36 * (t - c) ** 2 + 0.4
    x0 = 2.0
    q = lambda t: f1d(x0) + df(x0) * (t - x0) + 0.5 * d2f(x0) * (t - x0) ** 2
    x1 = x0 - df(x0) / d2f(x0)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x, f1d(x), color=style.INK, lw=2.6, label="E(x)")
    ax.plot(x, q(x), "--", color=style.AQUA, lw=2.2, label=r"quadratic model from $\nabla E$, $\nabla^2 E$")
    ax.plot([x0], [f1d(x0)], "o", color=style.INK, ms=9)
    ax.plot([x1], [q(x1)], "o", color=style.AQUA, ms=9)
    arrow(ax, (x0, f1d(x0) + 0.15), (x1 + 0.05, q(x1) + 0.15), color=style.AQUA, connectionstyle="arc3,rad=0.3")
    ax.set_ylim(0, 4.5)
    ax.set_xlim(-3.3, 3.3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(frameon=False, fontsize=13, loc="upper center")
    fig.tight_layout()
    save(fig, "newton_1d")


@panel
def line_search():
    """Backtracking line search along a Newton direction d: try alpha = 1, halve until the energy drops."""
    a = np.linspace(0, 1.08, 400)
    phi = lambda t: 1.0 - 2.2 * t + 2.0 * t ** 2 + 1.6 * t ** 4          # E(x + alpha d), overshoots at alpha = 1
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(a, phi(a), color=style.INK, lw=2.6)
    ax.axhline(phi(0), color=style.MUTED, lw=1.2, ls=":")
    ax.text(1.07, phi(0) + 0.06, "E(x)", color=style.MUTED, fontsize=14, ha="right")
    tries = [1.0, 0.5]
    for k, t in enumerate(tries):
        ok = phi(t) < phi(0)
        ax.plot([t], [phi(t)], "o" if ok else "X", ms=13, color=style.AQUA if ok else "#c0392b", zorder=5)
        ax.text(t - (0.05 if not ok else 0), phi(t) + (0.0 if not ok else -0.3), (f"α = {t:g}: " + ("accept" if ok else "uphill, halve")),
                ha="right" if not ok else "center", va="center", fontsize=15, color=style.AQUA if ok else "#c0392b")
    ax.plot([0], [phi(0)], "o", ms=10, color=style.INK, zorder=5)
    ax.text(0.02, phi(0) - 0.25, "x", fontsize=16, color=style.INK)
    ax.set_xlim(-0.02, 1.1); ax.set_ylim(0, 3.0)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([])
    ax.set_xlabel(r"step size $\alpha$ along the Newton direction $d$", fontsize=15)
    ax.set_ylabel(r"$E(x + \alpha d)$", fontsize=15)
    fig.tight_layout()
    save(fig, "line_search")


@panel
def newton_uphill():
    """Where the Hessian is negative, the 'bottom' of the model is a top: Newton climbs."""
    x = np.linspace(-3.2, 3.2, 400)
    x0 = 0.75
    q = lambda t: f1d(x0) + df(x0) * (t - x0) + 0.5 * d2f(x0) * (t - x0) ** 2
    x1 = x0 - df(x0) / d2f(x0)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x, f1d(x), color=style.INK, lw=2.6, label="E(x)")
    ax.plot(x, q(x), "--", color="#c0392b", lw=2.2, label=r"model with $\nabla^2 E < 0$")
    ax.plot([x0], [f1d(x0)], "o", color=style.INK, ms=9)
    ax.plot([x1], [f1d(x1)], "o", color="#c0392b", ms=9)
    arrow(ax, (x0, f1d(x0) + 0.12), (x1 + 0.02, f1d(x1) + 0.12), color="#c0392b", connectionstyle="arc3,rad=0.5")
    ax.text(x1 + 0.35, f1d(x1) + 0.75, "uphill", color="#c0392b", fontsize=16, ha="center")
    ax.set_ylim(0, 4.5)
    ax.set_xlim(-3.3, 3.3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(frameon=False, fontsize=13, loc="upper center")
    fig.tight_layout()
    save(fig, "newton_uphill")


def basis_matrix(U, V, order):
    return np.array([np.outer(U[:, i], V[:, j]).ravel() for i, j in order])


def blocks(d, energy=None):
    rng = np.random.default_rng(4)
    s = np.array([1.6, 0.7, 1.2][:d])
    Q = lambda: np.linalg.qr(rng.normal(size=(d, d)))[0] * np.r_[np.ones(d - 1), -1] if False else np.linalg.qr(rng.normal(size=(d, d)))[0]
    U, V = Q(), Q()
    U[:, -1] *= np.sign(np.linalg.det(U))
    V[:, -1] *= np.sign(np.linalg.det(V))
    F = U @ np.diag(s) @ V.T
    energy = energy or WithVolume(SymmetricDirichlet(), 1.0)                      # volume term couples the stretches
    H = projected_hessian(energy, F[None], "none")[0]
    order = [(i, i) for i in range(d)] + [p for i in range(d) for j in range(i + 1, d) for p in ((i, j), (j, i))]
    B = basis_matrix(U, V, order)
    return H, B @ H @ B.T


@panel
def hessian_blocks():
    # 2D: ARAP (the only energy introduced by then): its scaling block is diagonal. 3D: a coupled energy.
    for d, name in [(2, "hessian_blocks"), (3, "hessian_blocks3d")]:
        H, Hb = blocks(d, ARAP() if d == 2 else None)
        n = d * d
        pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
        std = [f"$E_{{{i + 1}{j + 1}}}$" for i in range(d) for j in range(d)]
        new = [f"$u_{i + 1}v_{i + 1}^\\top$" for i in range(d)] + \
              [f"$u_{a + 1}v_{b + 1}^\\top$" for i, j in pairs for a, b in ((i, j), (j, i))]
        fs = 24 if d == 2 else 14
        fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.6))
        for ax, M, t, labs in [(axs[0], H, "standard basis", std), (axs[1], Hb, r"basis $u_i v_j^\top$", new)]:
            A = np.abs(M) / np.abs(M).max()
            ax.imshow(np.where(A < 1e-9, np.nan, A ** 0.4), cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels(labs, fontsize=fs, color=style.INK2)
            ax.set_yticklabels(labs, fontsize=fs, color=style.INK2)
            ax.xaxis.tick_top()
            ax.tick_params(length=0)
            ax.set_title(t, fontsize=24 if d == 2 else 18, color=style.INK2, pad=48 if d == 2 else 32)
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(style.GRID)
            ax.set_facecolor("white")
        # name the entries of the block-diagonal form
        A = np.abs(Hb) / np.abs(Hb).max()
        ink = lambda r, c: "white" if A[r, c] ** 0.4 > 0.6 else style.INK
        R = plt.Rectangle
        axs[1].add_patch(R((-0.5, -0.5), d, d, fill=False, ec=style.ACCENT, lw=2.5))
        axs[1].text(d - 0.5 + 0.12, d / 2 - 0.5, r"$\nabla^2_\sigma\hat\psi$", color=style.ACCENT,
                    fontsize=30 if d == 2 else 20, ha="left", va="center")
        for m, (i, j) in enumerate(pairs):
            k = d + 2 * m
            axs[1].add_patch(R((k - 0.5, k - 0.5), 2, 2, fill=False, ec=style.ACCENT, lw=2.5))
            sub = "" if d == 2 else f"_{{{i + 1}{j + 1}}}"
            for r in range(2):
                for c in range(2):
                    axs[1].text(k + c, k + r, f"${'a' if r == c else 'b'}{sub}$", color=ink(k + r, k + c),
                                fontsize=34 if d == 2 else 17, ha="center", va="center")
        if d == 2:                                   # ARAP: the off-diagonal of the scaling block is exactly 0
            for r, c in [(0, 1), (1, 0)]:
                axs[1].text(c, r, "$0$", color=style.MUTED, fontsize=30, ha="center", va="center")
        fig.text(0.5, 0.45, "→", fontsize=40, ha="center", va="center", color=style.INK2)
        fig.tight_layout(w_pad=4)
        save(fig, name)


@panel
def hessian_diag():
    """One step past hessian_blocks: the eigenbasis D1, D2 (scaling), T12 (twist), L12 (flip) makes the
    2D ARAP Hessian diagonal, with the closed-form eigenvalues on the diagonal."""
    d = 2
    rng = np.random.default_rng(4)
    s = np.array([1.6, 0.7])
    U, V = np.linalg.qr(rng.normal(size=(d, d)))[0], np.linalg.qr(rng.normal(size=(d, d)))[0]
    U[:, -1] *= np.sign(np.linalg.det(U)); V[:, -1] *= np.sign(np.linalg.det(V))
    F = U @ np.diag(s) @ V.T
    H = projected_hessian(ARAP(), F[None], "none")[0]
    o = lambda i, j: np.outer(U[:, i], V[:, j]).ravel()
    Bu = np.array([o(0, 0), o(1, 1), o(0, 1), o(1, 0)])
    Be = np.array([o(0, 0), o(1, 1), (o(0, 1) - o(1, 0)) / np.sqrt(2), (o(0, 1) + o(1, 0)) / np.sqrt(2)])
    Hu, He = Bu @ H @ Bu.T, Be @ H @ Be.T
    lam = np.diag(He)
    psi = 2 * (s - 1)
    assert np.allclose(He - np.diag(lam), 0, atol=1e-9)
    assert np.isclose(lam[2], (psi[0] + psi[1]) / (s[0] + s[1])) and np.isclose(lam[3], (psi[0] - psi[1]) / (s[0] - s[1]))
    print("ARAP eigenvalues (D1, D2, T12, L12):", lam.round(4))
    vmax = max(np.abs(H).max(), np.abs(Hu).max(), np.abs(He).max())
    std = ["$E_{11}$", "$E_{12}$", "$E_{21}$", "$E_{22}$"]
    uv = [r"$u_1v_1^\top$", r"$u_2v_2^\top$", r"$u_1v_2^\top$", r"$u_2v_1^\top$"]
    eig = ["$D_1$", "$D_2$", "$T_{12}$", "$L_{12}$"]
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.8))
    for ax, M, t, labs in [(axs[0], H, "standard basis", std), (axs[1], Hu, r"basis $u_i v_j^\top$", uv),
                           (axs[2], He, "eigenbasis", eig)]:
        A = np.abs(M) / vmax
        ax.imshow(np.where(A < 1e-9, np.nan, A ** 0.4), cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(labs, fontsize=21, color=style.INK2); ax.set_yticklabels(labs, fontsize=21, color=style.INK2)
        ax.xaxis.tick_top(); ax.tick_params(length=0)
        ax.set_title(t, fontsize=22, color=style.INK2, pad=44)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(style.GRID)
    R = plt.Rectangle
    # middle: the blocks, as on the previous slide
    A = np.abs(Hu) / vmax
    axs[1].add_patch(R((-0.5, -0.5), 2, 2, fill=False, ec=style.ACCENT, lw=2.5))
    axs[1].add_patch(R((1.5, 1.5), 2, 2, fill=False, ec=style.ACCENT, lw=2.5))
    for r, c in [(0, 1), (1, 0)]:
        axs[1].text(c, r, "$0$", color=style.MUTED, fontsize=26, ha="center", va="center")
    for r in range(2):
        for c in range(2):
            axs[1].text(2 + c, 2 + r, f"${'a' if r == c else 'b'}$", fontsize=28, ha="center", va="center",
                        color="white" if A[2 + r, 2 + c] ** 0.4 > 0.6 else style.INK)
    # right: only the diagonal, each entry its eigenvalue
    A = np.abs(He) / vmax
    for k, l in enumerate([r"$\lambda_{D_1}$", r"$\lambda_{D_2}$", r"$\lambda_{T}$", r"$\lambda_{L}$"]):
        axs[2].add_patch(R((k - 0.5, k - 0.5), 1, 1, fill=False, ec=style.ACCENT, lw=2.5))
        axs[2].text(k, k, l, fontsize=24, ha="center", va="center", color="white" if A[k, k] ** 0.4 > 0.6 else style.INK)
    for x in (0.318, 0.672):
        fig.text(x, 0.45, "→", fontsize=40, ha="center", va="center", color=style.INK2)
    fig.tight_layout(w_pad=5)
    save(fig, "hessian_diag")


@panel
def modes():
    """The four Hessian modes in 2D as ellipse perturbations."""
    F = np.diag([1.5, 0.8])
    Qs = [("scale σ₁", np.diag([1.0, 0])), ("scale σ₂", np.diag([0, 1.0])),
          ("twist", np.array([[0, -1], [1, 0]]) / np.sqrt(2)), ("flip", np.array([[0, 1], [1, 0]]) / np.sqrt(2))]
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, (t, Q) in zip(axs, Qs):
        ax.add_patch(Polygon(ellipse_pts(F), fc=style.LAND, ec=style.INK2, lw=1.6))
        ax.add_patch(Polygon(ellipse_pts(F + 0.45 * Q), fc="none", ec=style.ACCENT, lw=2.6, ls="--"))
        ax.set_xlim(-2.1, 2.1)
        ax.set_ylim(-1.6, 1.6)
        clean(ax)
        ax.set_title(t, fontsize=18, color=style.INK2)
    save(fig, "modes")


def twist_region(energy, name, title):
    s1, s2 = np.meshgrid(np.linspace(0.02, 3, 500), np.linspace(0.02, 3, 500))
    S = np.stack([s1.ravel(), s2.ravel()], 1)
    g = energy.grad(S)
    lam = ((g[:, 0] + g[:, 1]) / (S[:, 0] + S[:, 1])).reshape(s1.shape)
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.contourf(s1, s2, lam < 0, levels=[0.5, 1.5], colors=["#f6c9c4"])
    ax.contour(s1, s2, lam, levels=[0], colors=["#c0392b"], linewidths=2)
    sigma_plane(ax)
    ax.text(0.55, 0.45, "twist\n< 0", color="#c0392b", fontsize=16, ha="center")
    ax.set_title(title, fontsize=15, color=style.INK2)
    fig.tight_layout()
    save(fig, name)


@panel
def twist_regions():
    twist_region(ARAP(), "twist_arap", "ARAP: twist eigenvalue < 0 where σ₁ + σ₂ < 2")
    twist_region(SymmetricDirichlet(), "twist_symdir", "Symmetric Dirichlet: twist eigenvalue < 0")


@panel
def clamp():
    lam = np.array([5.2, 1.4, -0.9, 2.3])
    labels = ["scale 1", "scale 2", "twist", "flip"]
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, L, t in [(axs[0], lam, "true Hessian"), (axs[1], np.maximum(lam, 0), "clamped")]:
        ax.bar(labels, L, color=[style.BLUE if v >= 0 else "#c0392b" for v in L], width=0.6)
        ax.axhline(0, color=style.INK2, lw=1)
        ax.set_title(t, fontsize=15, color=style.INK2)
        ax.tick_params(axis="x", labelsize=13)
    axs[0].set_ylabel("eigenvalue λ", fontsize=13)
    fig.tight_layout()
    save(fig, "clamp")


@panel
def maxstep():
    a, b, c = -1.0, 0.2, 1.0
    al = np.linspace(0, 1.6, 300)
    det = a * al**2 + b * al + c
    root = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
    fig, ax = plt.subplots(figsize=(7, 4.0))
    ax.axvspan(root, 1.6, color="#f6c9c4", lw=0)
    ax.plot(al, det, color=style.INK, lw=2.6)
    ax.axhline(0, color=style.INK2, lw=1)
    ax.plot([root], [0], "o", color="#c0392b", ms=9)
    ax.annotate("collapse: det F = 0", (root, 0), (root - 0.75, -0.55), color="#c0392b", fontsize=14,
                arrowprops=dict(arrowstyle="-", color="#c0392b"))
    ax.plot([0.9 * root], [a * (0.9 * root) ** 2 + b * 0.9 * root + c], "o", color=style.AQUA, ms=9)
    ax.text(0.9 * root - 0.05, 0.05, "max step", color=style.AQUA, fontsize=14, ha="right", va="bottom")
    ax.text(root + 0.05, 0.6, "flipped", color="#c0392b", fontsize=14)
    ax.set_xlabel("step length α", fontsize=14)
    ax.set_ylabel("det F(x + α d)", fontsize=14)
    ax.set_xlim(0, 1.6)
    ax.set_ylim(-1.2, 1.3)
    ax.set_yticks([0])
    fig.tight_layout()
    save(fig, "maxstep")


@panel
def tri_tet():
    """2D: triangle, F is 2x2. 3D: tetrahedron, F is 3x3."""
    fig = plt.figure(figsize=(10, 4.4))
    ax = fig.add_subplot(1, 2, 1)
    T = np.array([[0, 0], [1.2, 0.1], [0.4, 1.0]])
    ax.add_patch(Polygon(T, fc=style.LAND, ec=style.INK2, lw=2))
    ax.set_xlim(-0.3, 1.5)
    ax.set_ylim(-0.3, 1.3)
    clean(ax)
    ax.set_title(r"triangle: $F \in \mathbb{R}^{2\times2}$", fontsize=17, color=style.INK2)
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    P = np.array([[0, 0, 0], [1.2, 0.1, 0], [0.4, 1.0, 0], [0.5, 0.4, 1.0]])
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    faces = [[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]]
    ax3.add_collection3d(Poly3DCollection([P[f] for f in faces], fc=style.LAND, ec=style.INK2, lw=1.5, alpha=0.75))
    ax3.set_xlim(0, 1.2); ax3.set_ylim(0, 1.2); ax3.set_zlim(0, 1.0)
    ax3.view_init(18, -60)
    ax3.set_axis_off()
    ax3.set_title(r"tetrahedron: $F \in \mathbb{R}^{3\times3}$", fontsize=17, color=style.INK2)
    save(fig, "tri_tet")


@panel
def flow_invariants():
    fig, ax = plt.subplots(figsize=(14, 2.9))
    rows = [(1.0, ["ψ(F)", "rewrite in\nI₁, I₂, I₃", "9 scalar\nderivatives", "chain rule with\ng₁…g₃, H₁…H₃", "3×3 block,\n7 terms / entry"],
             "#f3e3d9", "invariants"),
            (0.0, ["ψ̂(σ)", "∂ψ̂/∂σᵢ,\n∂²ψ̂/∂σᵢ∂σⱼ", r"done:  $\nabla^2_\sigma\hat\psi$  and  $(\psi_i \pm \psi_j)/(\sigma_i \pm \sigma_j)$"], "#d6f0e6", "stretches")]
    for y, steps, fc, lab in rows:
        ax.text(-1.25, y, lab, ha="right", va="center", fontsize=18, color=style.INK2)
        xs = np.arange(len(steps)) * 3.1 if len(steps) > 3 else np.array([0, 3.1, 7.75])
        ws = [2.6] * len(steps) if len(steps) > 3 else [2.6, 2.6, 6.0]
        for k, (x, t) in enumerate(zip(xs, steps)):
            box(ax, (x, y), ws[k], 0.82, t, fc, fs=16)
            if k:
                arrow(ax, (xs[k - 1] + ws[k - 1] / 2 + 0.05, y), (x - ws[k] / 2 - 0.05, y), ms=14, lw=1.8)
    ax.set_xlim(-3.9, 13.8)
    ax.set_ylim(-0.55, 1.55)
    clean(ax)
    save(fig, "flow_invariants")


@panel
def opcount():
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.barh([1, 0], [122, 4], color=[style.MUTED, style.ACCENT], height=0.55)
    ax.set_yticks([1, 0], ["invariants", "stretches"], fontsize=14)
    for y, v in [(1, 122), (0, 4)]:
        ax.text(v + 2, y, str(v), va="center", fontsize=14, color=style.INK2)
    ax.set_xlabel("operations in one scaling entry (Sym. Dirichlet)", fontsize=12)
    ax.set_xlim(0, 140)
    fig.tight_layout()
    save(fig, "opcount")


# ------------------------------------------------------------------ Act 4
@panel
def recipe():
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ins = [("unknowns x", 1.5), ("energy ψ(σ)", 0.5), ("constraints", -0.5)]
    for t, y in ins:
        box(ax, (0, y), 2.7, 0.75, t, "#eef3f8", fs=21)
        arrow(ax, (1.4, y), (3.15, 0.5), ms=16, lw=2)
    box(ax, (4.6, 0.5), 2.8, 1.1, "projected\nNewton", style.ACCENT, fs=23, tc="white")
    outs = [("spherical maps", 1.5), ("shape editing", 0.5), ("physics", -0.5)]
    for t, y in outs:
        arrow(ax, (6.05, 0.5), (7.6, y), ms=16, lw=2)
        box(ax, (9.1, y), 2.9, 0.75, t, "#d6f0e6", fs=21)
    ax.set_xlim(-1.4, 10.6)
    ax.set_ylim(-1.0, 2.0)
    clean(ax)
    save(fig, "recipe")


@panel
def sphere_step():
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    t = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(t), np.sin(t), color=style.INK2, lw=2)
    a = np.radians(100)
    x = np.array([np.cos(a), np.sin(a)])
    tang = np.array([np.sin(a), -np.cos(a)])
    y = x + 1.15 * tang
    p = y / np.linalg.norm(y)
    ax.plot(*np.c_[x - 0.4 * tang, x + 1.25 * tang], color=style.GRID, lw=1.5)
    arrow(ax, tuple(x), tuple(y), color=style.BLUE, lw=2.6)
    ax.plot([0, y[0]], [0, y[1]], ":", color=style.MUTED)
    arrow(ax, tuple(y), tuple(p), color=style.ACCENT, lw=2.6)
    ax.plot(*x, "o", color=style.INK, ms=9)
    ax.plot(*p, "o", color=style.ACCENT, ms=9)
    ax.text(*(x + [-0.22, 0.05]), "x", fontsize=20)
    ax.text(*((x + y) / 2 + [-0.15, 0.14]), "tangent step", fontsize=16, color=style.BLUE)
    ax.text(*(p + [0.3, -0.15]), "project back\nonto the sphere", fontsize=16, color=style.ACCENT)
    ax.plot(0, 0, "o", color=style.INK2, ms=5)
    ax.set_xlim(-1.3, 2.2)
    ax.set_ylim(-1.2, 1.7)
    clean(ax)
    save(fig, "sphere_step")


@panel
def implicit_euler():
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    r = np.array([0, 0])
    for rad in [0.5, 1.0, 1.5, 2.0, 2.5]:
        ax.add_patch(Ellipse(r, 2 * rad * 1.3, 2 * rad, fc="none", ec=style.GRID, lw=1))
    xn = np.array([-0.6, 1.0])
    xt = np.array([1.9, 1.6])
    xnew = np.array([1.0, 0.95])
    ax.plot(*r, "o", color=style.AQUA, ms=10)
    ax.text(r[0] - 0.1, r[1] - 0.45, "rest shape\n(min of E)", fontsize=13, color=style.AQUA, ha="center")
    ax.plot(*xn, "o", color=style.INK2, ms=9)
    ax.text(xn[0] - 0.15, xn[1] + 0.15, r"$x_n$", fontsize=18, ha="right")
    arrow(ax, tuple(xn), tuple(xt), color=style.BLUE, lw=2.2)
    ax.plot(*xt, "o", mfc="white", mec=style.BLUE, ms=10, mew=2)
    ax.text(xt[0] + 0.1, xt[1] + 0.1, r"$\tilde x$ (inertia)", fontsize=16, color=style.BLUE)
    arrow(ax, tuple(xt), tuple(xnew), color=style.ACCENT, lw=2.2)
    ax.plot(*xnew, "o", color=style.ACCENT, ms=10)
    ax.text(xnew[0] + 0.12, xnew[1] - 0.35, r"$x_{n+1}$", fontsize=18, color=style.ACCENT)
    ax.set_xlim(-2.6, 3.6)
    ax.set_ylim(-1.6, 2.4)
    clean(ax)
    save(fig, "implicit_euler")


@panel
def implicit_euler_steps():
    """Time stepping: a strip x0 -> x1 -> ... -> xn -> xn+1 (steps of h), and the one step xn -> xn+1 as a
    minimization: inertia carries xn to x-tilde, the elastic energy pulls it back toward rest."""
    fig = plt.figure(figsize=(10.5, 4.3))
    top = fig.add_axes([0.02, 0.70, 0.96, 0.28])
    ax = fig.add_axes([0.08, 0.0, 0.84, 0.70])
    # --- the time line
    xs = [0.0, 1.0, 2.0, 4.4, 5.6]
    labs = [r"$x_0$", r"$x_1$", r"$x_2$", r"$x_n$", r"$x_{n+1}$"]
    top.plot([-0.4, 6.2], [0, 0], color=style.GRID, lw=2, zorder=0)
    for x, l in zip(xs, labs):
        last = l == r"$x_{n+1}$"
        top.plot([x], [0], "o", ms=12, color=style.ACCENT if last else style.INK2, zorder=3)
        top.text(x, 0.42, l, ha="center", va="bottom", fontsize=21, color=style.ACCENT if last else style.INK)
    top.text(3.2, 0.0, "· · ·", ha="center", va="center", fontsize=24, color=style.INK2,
             bbox=dict(fc="white", ec="none", pad=1))
    for a, b in [(0.0, 1.0), (1.0, 2.0), (4.4, 5.6)]:
        c = style.ACCENT if b == 5.6 else style.BLUE
        top.annotate("", xy=(b - 0.08, -0.45), xytext=(a + 0.08, -0.45), arrowprops=dict(arrowstyle="-|>", color=c, lw=2))
        top.text((a + b) / 2, -0.62, r"$h$", ha="center", va="top", fontsize=19, color=c)
    top.text(6.25, 0.0, "time", ha="left", va="center", fontsize=16, color=style.MUTED)
    top.set_xlim(-0.5, 6.9); top.set_ylim(-1.3, 1.2); top.axis("off")
    # --- one step, as a minimization
    r = np.array([0, 0])
    for rad in [0.5, 1.0, 1.5, 2.0, 2.5]:
        ax.add_patch(Ellipse(r, 2 * rad * 1.3, 2 * rad, fc="none", ec=style.GRID, lw=1))
    xn = np.array([-0.6, 1.0]); xt = np.array([1.9, 1.6]); xnew = np.array([1.0, 0.95])
    ax.plot(*r, "o", color=style.AQUA, ms=11)
    ax.text(r[0] + 0.22, r[1], "rest shape (min of E)", fontsize=17, color=style.AQUA, ha="left", va="center")
    ax.plot(*xn, "o", color=style.INK2, ms=10)
    ax.text(xn[0] - 0.15, xn[1] + 0.12, r"$x_n$", fontsize=22, ha="right")
    arrow(ax, tuple(xn), tuple(xt), color=style.BLUE, lw=2.4)
    ax.text(0.45, 1.62, r"inertia: $x_n + h\,v_n$", fontsize=18, color=style.BLUE, ha="center", rotation=12)
    ax.plot(*xt, "o", mfc="white", mec=style.BLUE, ms=11, mew=2)
    ax.text(xt[0] + 0.12, xt[1] + 0.02, r"$\tilde x$", fontsize=22, color=style.BLUE)
    arrow(ax, tuple(xt), tuple(xnew), color=style.ACCENT, lw=2.4)
    ax.text(1.95, 1.0, "elastic pull", fontsize=18, color=style.ACCENT)
    ax.plot(*xnew, "o", color=style.ACCENT, ms=11)
    ax.text(xnew[0] - 0.1, xnew[1] - 0.42, r"$x_{n+1}$", fontsize=22, color=style.ACCENT, ha="center")
    ax.set_xlim(-2.2, 3.4); ax.set_ylim(-0.75, 2.0)
    clean(ax)
    ax.set_aspect("auto")
    save(fig, "implicit_euler_steps")


@panel
def taylor_zoom():
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    s1, s2 = np.meshgrid(np.linspace(0.05, 3, 400), np.linspace(0.05, 3, 400))
    S = np.stack([s1.ravel(), s2.ravel()], 1)
    E = SymmetricDirichlet().psi(S).reshape(s1.shape) - 4
    ax.contour(s1, s2, np.log1p(E), levels=14, colors=[style.BLUE], linewidths=1, alpha=0.7)
    sigma_plane(ax, curves=False)
    ax.add_patch(plt.Circle((1, 1), 0.28, fc="none", ec=style.ACCENT, lw=2.5))
    ins = ax.inset_axes([0.55, 0.55, 0.42, 0.42])
    z1, z2 = np.meshgrid(np.linspace(0.8, 1.2, 200), np.linspace(0.8, 1.2, 200))
    Q = 4 * ((z1 - 1) ** 2 + (z2 - 1) ** 2)
    ins.contour(z1, z2, Q, levels=8, colors=[style.ACCENT], linewidths=1.2)
    ins.set_xticks([]); ins.set_yticks([])
    ins.set_title("near (1, 1): quadratic", fontsize=12, color=style.ACCENT)
    for sp in ins.spines.values():
        sp.set_visible(True); sp.set_color(style.ACCENT)
    fig.tight_layout()
    save(fig, "taylor_zoom")


@panel
def renders():
    """Plain bunny and a Blue Marble globe, for the 'Earth on a bunny?' question."""
    from distortion.render import earth_texture, render_textured, subdivide
    from distortion import polyhedral as ph
    d = np.load(os.path.join(HERE, "..", "..", "assets", "fig", "cache", "bunny.npz"))
    gray = np.full((8, 16, 3), 215, np.uint8)
    img = render_textured(d["V"], d["symdir"], d["F"], gray, camera=[(0.4, 0.8, 5.2), (0, 0.25, 0), (0, 1, 0)],
                          size=(1000, 1000))
    plt.imsave(os.path.join(OUT, "aux_bunny_plain.png"), img)
    V, F = ph.icosahedron()
    _, S, Fs = subdivide(V, V, F, 5)
    look = ph.lonlat_to_xyz(np.radians(20), np.radians(15))
    img = render_textured(S, S, Fs, earth_texture(), camera=[tuple(5.2 * look), (0, 0, 0), (0, 0, 1)], size=(1000, 1000))
    plt.imsave(os.path.join(OUT, "aux_globe.png"), img)


@panel
def angle_defect():
    """Backup: angle sum < 2 pi (cone, bulge) vs > 2 pi (saddle)."""
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, n, ang, t in [(axs[0], 5, 60, "angles sum < 360°: bulge"), (axs[1], 7, 60, "angles sum > 360°: saddle")]:
        tot = n * ang
        for k in range(n):
            a0, a1 = np.radians(k * ang), np.radians((k + 1) * ang)
            if np.degrees(a1) > 360:
                c = "#f6c9c4"
            else:
                c = style.LAND
            ax.add_patch(Polygon([[0, 0], [np.cos(a0), np.sin(a0)], [np.cos(a1), np.sin(a1)]], fc=c, ec=style.INK2, lw=1.2, alpha=0.9))
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        clean(ax)
        ax.set_title(t, fontsize=14, color=style.INK2)
    save(fig, "angle_defect")




@panel
def gradient_force():
    """The gradient pulls along the ellipse's axes; on the sigma-plane it is an arrow downhill."""
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6))
    F = np.diag([1.6, 0.7])
    ax = axs[0]
    ax.add_patch(Polygon(ellipse_pts(F), fc=style.LAND, ec=style.INK2, lw=2))
    for v, c in [((1.6, 0), style.BLUE), ((-1.6, 0), style.BLUE), ((0, 0.7), style.ORANGE), ((0, -0.7), style.ORANGE)]:
        v = np.array(v)
        d = -0.45 * v / np.linalg.norm(v) if c == style.BLUE else 0.45 * v / np.linalg.norm(v)
        arrow(ax, tuple(v), tuple(v + d), color=c, lw=3, ms=20)
    ax.text(0, -1.45, "too long: pull in    too short: push out", ha="center", fontsize=13, color=style.INK2)
    ax.text(2.05, 0.18, r"$u_1$", color=style.BLUE, fontsize=18)
    ax.text(0.12, 0.98, r"$u_2$", color=style.ORANGE, fontsize=18)
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-1.7, 1.4)
    clean(ax)
    ax = axs[1]
    s1, s2 = np.meshgrid(np.linspace(0.2, 3, 300), np.linspace(0.2, 3, 300))
    S = np.stack([s1.ravel(), s2.ravel()], 1)
    E = SymmetricDirichlet().psi(S).reshape(s1.shape) - 4
    ax.contour(s1, s2, np.log1p(E), levels=12, colors=[style.BLUE], linewidths=0.9, alpha=0.6)
    sigma_plane(ax, curves=False)
    p = np.array([1.6, 0.7])
    g = SymmetricDirichlet().grad(p[None])[0]
    arrow(ax, tuple(p), tuple(p - 0.12 * g), color=style.ACCENT, lw=3, ms=20)
    ax.plot(*p, "o", color=style.INK, ms=8)
    ax.text(p[0] + 0.08, p[1] - 0.25, r"$-\nabla\hat\psi$", color=style.ACCENT, fontsize=17)
    fig.tight_layout()
    save(fig, "gradient_force")


@panel
def two_guards():
    """Signed stretch: the barrier makes the minimizer flip-free, but a full Newton step can jump over
    the wall (the energy is finite again on the inverted side); the max step stops it."""
    from distortion.eigensystem import ARAP, StVK, WithBarrier
    sg = np.linspace(-1.6, 2.6, 2000)
    tiny = np.geomspace(1e-12, 0.05, 3000)
    sg = np.unique(np.r_[sg, tiny, -tiny])
    abs_ = np.abs(sg)
    arap = (sg - 1) ** 2                                               # no wall: smooth through sigma = 0
    wb = WithBarrier(ARAP(), k=12.0)                                  # stiffness exaggerated for the drawing
    bar = wb.psi(abs_[:, None]) - ARAP().psi(abs_[:, None]) + (sg - 1) ** 2   # barrier on |sigma|
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.axvspan(-1.6, 0, color="#f6c9c4", alpha=0.6, lw=0)
    ax.text(-0.8, 5.6, "inverted\n(σ₂ < 0)", ha="center", fontsize=14, color="#c0392b")
    ax.plot(sg, arap, color=style.MUTED, lw=2.2, ls="--", label="ARAP: finite at σ = 0")
    ax.plot(sg, bar, color=style.ACCENT, lw=3, label="ARAP + barrier: ∞ at σ = 0")
    x0, x1 = 0.55, -0.55
    f = lambda x: np.interp(x, sg, bar)
    arrow(ax, (x0, f(x0) + 0.15), (x1, f(x1) + 0.15), color="#c0392b", lw=2.2, ms=18, connectionstyle="arc3,rad=-0.5")
    ax.text(-0.95, 0.35, "a full Newton step can\njump over the wall", ha="center", fontsize=13, color="#c0392b")
    xs = 0.3
    arrow(ax, (x0, f(x0) - 0.05), (xs, f(xs) if f(xs) < 6 else 5.8), color=style.AQUA, lw=2.4, ms=18)
    ax.text(0.33, 2.3, "max step +\nbacktracking stop it", fontsize=13, color=style.AQUA)
    ax.plot([1], [0], "o", color=style.AQUA, ms=9, zorder=6)
    ax.text(1.0, 0.9, "minimizer:\nflip-free", ha="center", fontsize=13, color=style.AQUA)
    ax.set_xlim(-1.6, 2.6)
    ax.set_ylim(0, 6.5)
    ax.set_yticks([])
    ax.set_xlabel("signed stretch σ₂", fontsize=15)
    ax.set_ylabel(r"$\hat\psi$", fontsize=16, rotation=0, labelpad=12)
    ax.legend(frameon=False, fontsize=13, loc="upper right")
    fig.tight_layout()
    save(fig, "two_guards")


@panel
def ds_dm():
    """Where F comes from: one triangle of the gingerbread, at rest and after the arm-raise drag.
    Both are images of one reference triangle: D_m sends it to the rest triangle, D_s to the deformed
    one, so F = D_s D_m^{-1} is the unique linear map taking rest edges to deformed edges."""
    import pickle
    from matplotlib.collections import PolyCollection
    from distortion.planar import gingerbread
    V, F, _ = gingerbread()
    X = pickle.load(open(os.path.join(HERE, "..", "scratch", "act2.pkl"), "rb"))["raise_arap"]
    t = int(np.argmin(np.linalg.norm(V[F].mean(1) - [1.0, 0.68], axis=1)))
    RED = "#c0392b"
    fig = plt.figure(figsize=(15, 5.4))
    A = [fig.add_axes([0.0, 0.08, 0.2, 0.8]), fig.add_axes([0.21, 0.0, 0.58, 0.92]), fig.add_axes([0.8, 0.08, 0.2, 0.8])]
    for ax, P, lab in [(A[0], V, "rest"), (A[2], X, "deformed")]:
        ax.add_collection(PolyCollection(P[F], fc=style.LAND, ec=style.MUTED, lw=0.15))
        ax.add_collection(PolyCollection(P[F[t:t + 1]], fc=RED, ec=RED, lw=1.5, zorder=3))
        c = P[F[t]].mean(0)
        ax.plot([c[0]], [c[1]], "o", ms=26, mfc="none", mec=RED, mew=2.5, zorder=4)
        ax.autoscale_view(); clean(ax)
        ax.set_title(lab, fontsize=18, color=style.INK2)
    ax = A[1]
    # the three triangles, each with its two edge vectors (columns): blue = edge 1, orange = edge 2
    rest = V[F[t]] - V[F[t]].mean(0)
    dfm = X[F[t]] - X[F[t]].mean(0)
    s = 1.0 / np.abs(np.r_[rest, dfm]).max()
    tris = [(rest * s * 1.2 + [-2.6, 0.75], r"$X_0$", r"$X_1$", r"$X_2$", r"$D_m = [\,X_1 - X_0,\; X_2 - X_0\,]$"),
            (np.array([[0, 0], [1, 0], [0, 1]]) * 1.0 + [-0.5, -2.1], "", "", "", "reference triangle"),
            (dfm * s * 1.2 + [2.6, 0.75], r"$x_0$", r"$x_1$", r"$x_2$", r"$D_s = [\,x_1 - x_0,\; x_2 - x_0\,]$")]
    for P, l0, l1, l2, cap in tris:
        ax.add_patch(plt.Polygon(P, fc="#f6c9c4", ec=RED, lw=1.5, zorder=1))
        arrow(ax, tuple(P[0]), tuple(P[1]), color=style.BLUE, lw=3, ms=20)
        arrow(ax, tuple(P[0]), tuple(P[2]), color=style.ORANGE, lw=3, ms=20)
        c = P.mean(0)
        for p, l in zip(P, (l0, l1, l2)):
            if l:
                d = (p - c) / np.linalg.norm(p - c)
                ax.text(*(p + 0.28 * d), l, fontsize=17, ha="center", va="center", color=style.INK)
        ax.text(c[0], P[:, 1].min() - 0.62, cap, fontsize=16, ha="center", color=style.INK2)
    arrow(ax, (-0.7, -1.45), (-2.2, -0.75), color=style.INK2, lw=2.2, ms=20)
    ax.text(-1.55, -1.55, r"$D_m$", fontsize=24, color=style.INK2, ha="right", va="top")
    arrow(ax, (0.7, -1.45), (2.2, -0.75), color=style.INK2, lw=2.2, ms=20)
    ax.text(1.55, -1.55, r"$D_s$", fontsize=24, color=style.INK2, ha="left", va="top")
    ax.add_patch(__import__("matplotlib.patches", fromlist=["FancyArrowPatch"]).FancyArrowPatch(
        (-1.9, 1.55), (1.9, 1.55), connectionstyle="arc3,rad=-0.25", arrowstyle="-|>", mutation_scale=24,
        lw=3, color=style.ACCENT))
    ax.text(0, 2.15, r"$F = D_s\,D_m^{-1}$", fontsize=26, color=style.ACCENT, ha="center")
    ax.set_xlim(-3.9, 3.9); ax.set_ylim(-2.5, 2.55); clean(ax)
    save(fig, "ds_dm")


def conformal_plane(ax, lim=1.75):
    """Axes of the conformal plane {a I + b J}: a = uniform scale, b = rotation."""
    ax.axhline(0, color=style.GRID, lw=1); ax.axvline(0, color=style.GRID, lw=1)
    ax.text(lim - 0.05, -0.14, "scale", ha="right", va="top", fontsize=14, color=style.INK2)
    ax.text(0.06, lim - 0.05, "rotation", ha="left", va="top", fontsize=14, color=style.INK2)
    ax.set_xlim(-0.35 if lim < 2 else -lim, lim); ax.set_ylim(-0.35 if lim < 2 else -lim, lim)
    clean(ax)


def symdir_line(s, D, eps):
    """Symmetric Dirichlet along the straight line diag(s) + eps D, minus its value at eps = 0."""
    sv = lambda M: np.linalg.svd(M, compute_uv=False)
    E = lambda M: np.sum(sv(M) ** 2 + sv(M) ** -2.0)
    return np.array([E(np.diag(s) + e * D) for e in eps]) - E(np.diag(s))


@panel
def twist_orbit():
    """Why the twist eigenvalue goes negative: the twist orbit R Sigma R is a circle in the conformal
    plane (radius s = (s1 + s2)/2; the anti-conformal part stays fixed). A straight step along its
    tangent = a rotation (free) + an inflation s -> sqrt(s^2 + eps^2) of both stretches."""
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw=dict(width_ratios=[1, 1.15]))
    ax = axs[0]
    conformal_plane(ax)
    s, e = 1.0, 0.75
    t = np.linspace(-0.25, np.pi / 2 + 0.1, 200)
    ax.plot(s * np.cos(t), s * np.sin(t), color=style.INK, lw=2.6)
    r = np.hypot(s, e)
    ax.plot(r * np.cos(t), r * np.sin(t), color=style.MUTED, lw=1.2, ls="--")
    ax.plot([s, s], [-0.3, 1.2], color=style.ACCENT, lw=2.2, ls="--")
    th = np.arctan2(e, s)
    tt = np.linspace(0, th, 50)
    ax.plot(s * np.cos(tt), s * np.sin(tt), color=style.AQUA, lw=4, solid_capstyle="round")
    arrow(ax, (s * np.cos(th - 0.02), s * np.sin(th - 0.02)), (s * np.cos(th), s * np.sin(th)), color=style.AQUA, lw=3, ms=20)
    arrow(ax, (s * np.cos(th), s * np.sin(th)), (s, e), color="#c0392b", lw=3, ms=20)
    ax.plot([s], [0], "o", color=style.INK, ms=10, zorder=6)
    ax.plot([s], [e], "o", color=style.ACCENT, ms=10, zorder=6)
    ax.text(s + 0.07, -0.1, "F", fontsize=20, color=style.INK, va="top")
    ax.text(s + 0.07, e + 0.05, "straight twist step", fontsize=14, color=style.ACCENT)
    ax.text(0.5, 0.22, "rotate:\nE unchanged", fontsize=14, color=style.AQUA, ha="center")
    ax.text(1.06, 0.5, "inflate both\nstretches", fontsize=14, color="#c0392b", ha="left")
    ax.text(-0.3, 1.02, "twist orbit\n(E constant)", fontsize=14, color=style.INK, ha="left", va="bottom")
    ax.set_title("the twist orbit is a circle", fontsize=17, color=style.INK2)
    ax = axs[1]
    eps = np.linspace(-0.45, 0.45, 181)
    T = np.array([[0, -1], [1, 0]]) / np.sqrt(2)
    for sg, c, lab in [((1.3, 1.15), style.BLUE, "stretched: ψ₁ + ψ₂ > 0"), ((0.8, 0.7), "#c0392b", "compressed: ψ₁ + ψ₂ < 0")]:
        ax.plot(eps, symdir_line(np.array(sg), T, eps), color=c, lw=3, label=lab)
    ax.axhline(0, color=style.GRID, lw=1)
    ax.set_xlabel("step along the twist", fontsize=15)
    ax.set_ylabel("ΔE", fontsize=15, rotation=0, labelpad=16)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(frameon=False, fontsize=14, loc="lower left")
    ax.set_title("E along the straight step (Symmetric Dirichlet)", fontsize=17, color=style.INK2)
    fig.tight_layout(w_pad=3)
    save(fig, "twist_orbit")


@panel
def not_convex():
    """Rotations are the minimizers, and they form a circle: the chord between two of them is a shrunken
    rotation (compressed), so E rises in between. No rotation-invariant energy is convex."""
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw=dict(width_ratios=[1, 1.15]))
    ax = axs[0]
    conformal_plane(ax, lim=1.5)
    t = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(t), np.sin(t), color=style.AQUA, lw=3)
    ax.plot([1, 0], [0, 1], color="#c0392b", lw=2.4, ls="--")
    for (x, y), lab, dx in [((1, 0), "I", 0.1), ((0, 1), "R(90°)", 0.1)]:
        ax.plot([x], [y], "o", color=style.AQUA, ms=11, zorder=6)
        ax.text(x + dx, y + 0.08, lab, fontsize=18, color=style.INK)
    ax.plot([0.5], [0.5], "X", color="#c0392b", ms=14, zorder=6)
    ax.text(0.42, 0.36, "midpoint: rotation × 1/√2\n(compressed, E > 0)", fontsize=13, color="#c0392b", ha="right", va="top")
    ax.text(0.8, 0.86, "rotations: E = 0", fontsize=14, color=style.AQUA, ha="left")
    ax.set_title("the minimizers form a circle", fontsize=17, color=style.INK2)
    ax = axs[1]
    tt = np.linspace(0, 1, 201)
    r = np.sqrt((1 - tt) ** 2 + tt ** 2)
    ax.plot(tt, 2 * (r ** 2 + r ** -2) - 4, color="#c0392b", lw=3)
    ax.plot([0, 1], [0, 0], "o", color=style.AQUA, ms=11, zorder=6)
    ax.axhline(0, color=style.GRID, lw=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["I", "R(90°)"], fontsize=16)
    ax.set_yticks([])
    ax.set_ylabel("E", fontsize=16, rotation=0, labelpad=14)
    ax.set_xlabel("straight line between two rotations", fontsize=15)
    ax.text(0.5, 0.25, "a bump: not convex", fontsize=16, color="#c0392b", ha="center")
    ax.set_title("E along the chord (Symmetric Dirichlet)", fontsize=17, color=style.INK2)
    fig.tight_layout(w_pad=3)
    save(fig, "not_convex")


for f in PANELS:
    if not ONLY or f.__name__ in ONLY:
        f()
        print("done", f.__name__)
