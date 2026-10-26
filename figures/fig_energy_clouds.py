"""Stretch distribution of the hand's spherical map under six nu = 0 energies: one folded sigma-plane
per energy (log axes), same layout as the checker grid energies_hand_cube.png.

    python figures/fig_energy_clouds.py <out-dir>
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from distortion import style

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
HERE = os.path.dirname(os.path.abspath(__file__))
style.use()
d = np.load(os.path.join(HERE, "..", "..", "assets", "viewer", "cache100", "hand.npz"))
V, F = d["V"], d["F"]
KS = [("symdir", "Sym. Dirichlet"), ("en_symarap", "Sym. ARAP"), ("en_hencky", "Hencky"),
      ("en_neohookean", "Neo-Hookean"), ("en_arap_b", "ARAP + barrier"), ("en_stvk_b", "StVK + barrier")]


def sigmas(S):
    def frame(P):
        e1, e2 = P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]]
        x = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
        y = np.cross(np.cross(e1, e2), x)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        return np.stack([np.stack([(e1 * x).sum(1), (e1 * y).sum(1)], 1), np.stack([(e2 * x).sum(1), (e2 * y).sum(1)], 1)], 2)
    return np.linalg.svd(frame(S) @ np.linalg.inv(frame(V)), compute_uv=False)


def sig3(x):
    """Three significant figures, keeping trailing zeros (0.230, 7.41, 701)."""
    return f"{x:.{max(0, 2 - int(np.floor(np.log10(abs(x)))))}f}"


A = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
rng = np.random.default_rng(0)
pick = rng.choice(len(F), 9000, replace=False, p=A / A.sum())      # area-weighted sample
lo, hi = 0.08, 12
g = np.geomspace(lo, hi, 2)
fig, axs = plt.subplots(2, 3, figsize=(13.5, 9.2))
for ax, (k, name) in zip(axs.ravel(), KS):
    s = sigmas(d[k])[pick]
    ax.scatter(s[:, 0], s[:, 1], s=3, color=style.BLUE, alpha=0.3, lw=0, rasterized=True)
    ax.plot(g, g, ls="--", color=style.MUTED, lw=1)
    ax.plot(g, 1 / g, ls=":", color=style.MUTED, lw=1.2)
    ax.plot(1, 1, "o", ms=8, color=style.ACCENT, zorder=5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    style.mirror_half(ax, lo, hi, label=False)
    J = np.log2(s.prod(1))
    ax.set_title(f"{name}", fontsize=17, color=style.INK2)
    ax.text(0.04, 0.95, f"2% of area shrunk ≥ ×{2 ** -np.percentile(J, 2):.0f}\nmax stretch σ₁ = {s[:, 0].max():.0f}",
            transform=ax.transAxes, va="top", fontsize=12.5, color=style.INK2, zorder=6)
for ax in axs[1]:
    ax.set_xlabel(r"$\sigma_1$", fontsize=15)
for ax in axs[:, 0]:
    ax.set_ylabel(r"$\sigma_2$", fontsize=15, rotation=0, labelpad=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "energies_hand_clouds.png"))
print("ok")

# ---- merged figure: per energy, the checkered hand next to its cloud (2 x 3 cells)
from PIL import Image, ImageDraw, ImageFont
LEGEND = {"symdir": (style.AQUA, False), "en_symarap": (style.BLUE, True), "en_hencky": (style.INK, False),
          "en_neohookean": (style.ORANGE, False), "en_arap_b": (style.ACCENT, False), "en_stvk_b": (style.MUTED, False)}
SHOTS = sys.argv[2] if len(sys.argv) > 2 else None        # dir with encube_<key>.png viewer screenshots
if SHOTS:
    font = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 36)
    cells = []
    for k, name in KS:
        f2, ax = plt.subplots(figsize=(4.2, 4.2))
        s = sigmas(d[k])[pick]
        ax.scatter(s[:, 0], s[:, 1], s=3, color=style.BLUE, alpha=0.3, lw=0, rasterized=True)
        ax.plot(g, g, ls="--", color=style.MUTED, lw=1)
        ax.plot(g, 1 / g, ls=":", color=style.MUTED, lw=1.2)
        ax.plot(1, 1, "o", ms=8, color=style.ACCENT, zorder=5)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        style.mirror_half(ax, lo, hi, label=False)
        full = sigmas(d[k])                                   # all triangles, not the sample
        ax.text(0.04, 0.96, f"min σ = {sig3(full[:, 1].min())}\nmax σ = {sig3(full[:, 0].max())}",
                transform=ax.transAxes, va="top", fontsize=17, color=style.INK, zorder=6)
        print(f"{name:16s} min {full[:, 1].min():.3f}  max {full[:, 0].max():.1f}")
        ax.set_xlabel(r"$\sigma_1$", fontsize=14); ax.set_ylabel(r"$\sigma_2$", fontsize=14, rotation=0, labelpad=10)
        import io
        buf = io.BytesIO(); f2.savefig(buf, dpi=150, format="png"); plt.close(f2)
        buf.seek(0); cloud = Image.open(buf).convert("RGB")
        key = k.replace("en_", "")
        hand = Image.open(os.path.join(SHOTS, f"encube_{key}.png")).convert("RGB").crop((500, 130, 940, 610))
        H = 520
        hand = hand.resize((int(hand.width * H / hand.height), H)); cloud = cloud.resize((int(cloud.width * H / cloud.height), H))
        cell = Image.new("RGB", (hand.width + cloud.width + 10, H + 60), "white")
        cell.paste(hand, (0, 60)); cell.paste(cloud, (hand.width + 10, 60))
        dr = ImageDraw.Draw(cell)
        col, dashed = LEGEND[k]
        rgb = tuple(int(col[i:i + 2], 16) for i in (1, 3, 5))
        w = dr.textlength(name, font=font)
        sw = 70                                                  # line swatch, like a legend entry
        x0 = (cell.width - (w + sw + 16)) / 2
        y = 32
        if dashed:
            for xx in range(int(x0), int(x0 + sw), 22):
                dr.line([(xx, y), (min(xx + 13, x0 + sw), y)], fill=rgb, width=7)
        else:
            dr.line([(x0, y), (x0 + sw, y)], fill=rgb, width=7)
        dr.text((x0 + sw + 16, 8), name, fill=rgb, font=font)
        cells.append(cell)
    W, Hc = max(c.width for c in cells), max(c.height for c in cells)
    out = Image.new("RGB", (3 * W + 80, 2 * Hc + 40), "white")
    for i, c in enumerate(cells):
        out.paste(c, ((i % 3) * (W + 40), (i // 3) * (Hc + 40)))
    px = out.load()
    for y in range(out.size[1]):
        for x in range(out.size[0]):
            if px[x, y] in ((252, 252, 250), (251, 251, 249)): px[x, y] = (255, 255, 255)
    out.save(os.path.join(OUT, "energies_hand_merged.png"))
    print("merged", out.size)
