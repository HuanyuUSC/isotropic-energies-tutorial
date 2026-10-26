"""Title-slide teaser: the six applications of the talk (the master table's cells), as one strip of three
groups -- parameterization, deformation, dynamics -- each with its 2D and 3D example.
    python figures/fig_title_teaser.py      # writes assets/fig/title_teaser.png"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "..", "assets", "fig")
ACCENT = (181, 0, 91)
MUTED = (119, 119, 119)
H = 430           # thumbnail height (px)
GAP, GROUP_GAP = 40, 110

GROUPS = [("Parameterization", ["flatten_pair_a30.png", "sphere_pair_a30.png"]),
          ("Deformation", ["gb_raise.png", "demo3d_giraffe_sphere_cell.png"]),
          ("Dynamics", ["demo2d_crowd_cell.png", "demo3d_cantilever_cell.png"])]


def load(name, pad=6):
    a = np.asarray(Image.open(os.path.join(FIG, name)).convert("RGB"))
    ys, xs = np.nonzero((a < 245).any(-1))
    a = a[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]
    im = Image.fromarray(a)
    return im.resize((round(im.width * H / im.height), H), Image.LANCZOS)


def font(size):
    for f in ("seguisb.ttf", "segoeui.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main():
    """Grid: columns = applications, rows = 2D / 3D (the master table, transposed)."""
    cells = [[load(n) for n in names] for _, names in GROUPS]          # cells[col][row]
    colw = [max(c[0].width, c[1].width) for c in cells]
    left, top, CG, RG = 120, 96, 70, 36
    W = left + sum(colw) + CG * (len(cells) - 1) + 10
    Ht = top + 2 * H + RG
    canvas = Image.new("RGB", (W, Ht), "white")
    d = ImageDraw.Draw(canvas)
    ft, fr = font(66), font(54)
    x = left
    for (title, _), col, w in zip(GROUPS, cells, colw):
        tw = d.textlength(title, font=ft)
        d.text((x + (w - tw) / 2, 4), title, fill=ACCENT, font=ft)
        for r, im in enumerate(col):
            canvas.paste(im, (x + (w - im.width) // 2, top + r * (H + RG)))
        x += w + CG
    for r, lab in enumerate(("2D", "3D")):
        d.text((10, top + r * (H + RG) + H / 2 - 30), lab, fill=MUTED, font=fr)
    out = os.path.join(FIG, "title_grid_a30.png")             # (title_teaser.png is locked by Dropbox)
    canvas.save(out)
    print("wrote", out, canvas.size)


if __name__ == "__main__":
    main()
