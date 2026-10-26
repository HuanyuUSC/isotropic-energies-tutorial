"""The research hub for the UCLA talk (adapted from the thesis proposal's master visual): CT is replaced
by generative materials (ongoing work).

Center: the octant of principal-stretch space (lambda_1, lambda_2, lambda_3) with the rest state (1,1,1) and
one isotropic energy drawn as a level surface (Hencky, sum log^2 lambda_i = r^2, a sphere in log-stretch).
Right (forward, out-arrows): solid, shell, rod.  Left (inverse, in-arrows): poking, CT (dashed, proposed).

    python code/figures/fig_hub.py  -> assets/fig/hub_*.png

Variants: hub_all (plain), hub_<part> (one spoke lit, the rest dimmed), hub_forward, hub_inverse,
hub_reverse (the bridge: "where does psi come from?"), hub_coverage (poking curve vs CT cloud, schematic).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image
from distortion import style

style.use()
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "..", "assets", "research")

# (key, title, venue, image, crop box in relative coords (x0, y0, x1, y1), side, row)
PARTS = [
    ("solids", "Solids", "SIGGRAPH Asia 23 · SCA 26", os.path.join(FIG, "teaser_solids.png"), None, "R", 0),
    ("shells", "Thin shells", "SIGGRAPH Asia 26", os.path.join(FIG, "teaser_shells.png"), None, "R", 1),
    ("rods", "Rods", "SIGGRAPH 25", os.path.join(FIG, "teaser_rods.png"), None, "R", 2),
    ("poking", "Capture by poking", "SIGGRAPH Asia 23", os.path.join(FIG, "teaser_poking.png"), None, "L", 0),
    ("gen", "Generative materials", "ongoing", os.path.join(FIG, "gen_thumb.png"), None, "L", 1),
]
# spoke images are the writeup's teaser panels (fig/teaser/*.png), so the talk and the proposal share one hub
FORWARD = {"solids", "shells", "rods"}


def load(path, box, aspect=None):
    """Load an image; with aspect (width/height), pad it with white to that aspect instead of stretching."""
    im = Image.open(path).convert("RGB")
    if aspect:
        W, H = im.size
        tw, th = (int(round(H * aspect)), H) if W / H < aspect else (W, int(round(W / aspect)))
        canvas = Image.new("RGB", (tw, th), "white")
        canvas.paste(im, ((tw - W) // 2, (th - H) // 2))
        im = canvas
    if box:
        W, H = im.size
        im = im.crop((int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H)))
    return np.asarray(im)


def octant(ax, coverage=False, energy=True):
    ax.set_proj_type("ortho")
    L = 1.75
    for k, lab in enumerate(["σ₁", "σ₂", "σ₃"]):
        e = np.zeros(3); e[k] = L
        ax.plot([0, e[0]], [0, e[1]], [0, e[2]], color=style.INK2, lw=1.6)
        ax.text(*(e * 1.07), lab, fontsize=20, color=style.INK2, ha="center", va="center")
    if energy and not coverage:
        u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 60), np.linspace(0, np.pi, 30))
        r = 0.42
        X = np.exp(r * np.cos(u) * np.sin(v))
        Y = np.exp(r * np.sin(u) * np.sin(v))
        Z = np.exp(r * np.cos(v))
        ax.plot_surface(X, Y, Z, color=style.ACCENT, alpha=0.16 if coverage else 0.28, linewidth=0, shade=True)
        ax.plot_wireframe(X, Y, Z, color=style.ACCENT, alpha=0.18, lw=0.4, rstride=3, cstride=4)
    ax.scatter([1], [1], [1], s=60, color=style.INK, depthshade=False, zorder=10)
    ax.text(1.0, 1.0, 1.18, "rest\n(1,1,1)", fontsize=13, color=style.INK, ha="center")
    if coverage:
        # schematic only: poking = a thin tube near uniaxial compression; CT = a broad cloud (see coverage.pdf for data)
        t = np.linspace(0.62, 1.0, 40)
        lat = t ** -0.5
        ax.plot(lat, lat, t, color=style.ORANGE, lw=7, solid_capstyle="round", zorder=12)
        rng = np.random.default_rng(1)
        n = 900
        l3 = rng.uniform(0.59, 1.05, n)
        l1 = (l3 ** -0.5) * np.exp(rng.normal(0, 0.12, n))
        l2 = (l3 ** -0.5) * np.exp(rng.normal(0, 0.12, n))
        ax.scatter(l1, l2, l3, s=6, color=style.BLUE, alpha=0.35, depthshade=False)
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_zlim(0, L)
    ax.view_init(elev=22, azim=35)
    ax.set_box_aspect((1, 1, 1), zoom=1.2)
    ax.axis("off")


def hub(name, lit=None, reverse=False, coverage=False, caption=None):
    lit = set(lit) if lit is not None else {p[0] for p in PARTS}
    fig = plt.figure(figsize=(16, 8.4))
    ax = fig.add_axes([0.35, 0.03, 0.3, 0.78], projection="3d")
    octant(ax, coverage=coverage)
    fig.text(0.5, 0.93, "ψ(σ₁, σ₂, σ₃)", ha="center", fontsize=28, color=style.ACCENT)
    fig.text(0.5, 0.885, "one isotropic energy on the σ-space (principal stretches)", ha="center", fontsize=15,
             color=style.MUTED)
    fig.text(0.86, 0.95, "forward: simulate with ψ", ha="center", fontsize=19, color=style.INK2, weight="bold")
    fig.text(0.14, 0.95, "where ψ comes from", ha="center", fontsize=19, color=style.INK2, weight="bold")
    cx, cy = 0.5, 0.47
    for key, title, venue, img, box, side, row in PARTS:
        on = key in lit
        if side == "R":
            x0, y0, w, h = 0.745, 0.66 - row * 0.3, 0.23, 0.24
        else:
            x0, y0, w, h = 0.025, 0.55 - row * 0.4, 0.23, 0.3
        a = fig.add_axes([x0, y0 + 0.045, w, h - 0.05])
        a.imshow(load(img, box, aspect=(w * 16) / ((h - 0.05) * 8.4)), alpha=1.0 if on else 0.22)
        a.set_aspect("auto"); a.set_xticks([]); a.set_yticks([])
        dashed = key == "gen"
        for s in a.spines.values():
            s.set_visible(True)
            s.set_color(style.ACCENT if (on and lit != {p[0] for p in PARTS}) else style.MUTED)
            s.set_linewidth(3 if on and len(lit) < 5 else 1.2)
            if dashed:
                s.set_linestyle((0, (5, 3)))
        col = style.INK if on else style.GRID
        fig.text(x0 + w / 2, y0 + 0.012, title, ha="center", fontsize=17, color=col, weight="bold")
        fig.text(x0 + w / 2, y0 - 0.022, venue, ha="center", fontsize=13,
                 color=(style.ACCENT if venue == "ongoing" else style.MUTED) if on else style.GRID,
                 style="italic" if venue == "ongoing" else "normal")
        # spoke
        ey = y0 + h / 2
        if side == "R":
            p, q = (0.62, cy + (ey - cy) * 0.35), (x0 - 0.01, ey)
        else:
            p, q = (x0 + w + 0.01, ey), (0.38, cy + (ey - cy) * 0.35)
        if reverse and side == "R":
            p, q = q, p
        arr = FancyArrowPatch(p, q, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=28,
                              lw=3.2 if on else 1.5, color=style.ACCENT if on else style.GRID,
                              linestyle="--" if dashed else "-", connectionstyle="arc3,rad=0.0")
        fig.add_artist(arr)
    if reverse:
        fig.text(0.5, 0.02, "every simulator takes ψ as input: where does ψ come from?", ha="center",
                 fontsize=20, color=style.ACCENT)
    if coverage:
        fig.text(0.36, 0.06, "— poking: near one curve, under the indenter", ha="center", fontsize=16, color=style.ORANGE)
        fig.text(0.66, 0.06, "● CT: a cloud from one test", ha="center", fontsize=16, color=style.BLUE)
        fig.text(0.5, 0.025, "schematic; measured coverage on the next slides", ha="center", fontsize=13, color=style.MUTED)
    if caption:
        fig.text(0.5, 0.02, caption, ha="center", fontsize=20, color=style.ACCENT)
    fig.savefig(os.path.join(FIG, name), dpi=130, bbox_inches=None)
    plt.close(fig)


def main():
    hub("hub_all.png")
    for p in PARTS:
        hub("hub_%s.png" % p[0], lit=[p[0]])
    print("ok")


if __name__ == "__main__":
    main()
