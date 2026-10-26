"""Videos and stills of the 3D demos (distortion/contact.py, scratch/demo3d/*.py).

  squeeze     vid_squeeze_nu0 / _nu45: the pig pressed by a plate (deformation, quasistatic)
  cantilever  vid_cantilever: the armadillo falls on a cantilever (dynamics, IPC contact)
  hang        research/gen_hang: five bars, same E and nu, generated materials (research section)
  cells       stills for the master table: demo3d_pig_cell.png, demo3d_cantilever_cell.png

    python figures/fig_demo3d.py [squeeze|cantilever|hang|cells]
Surfaces are colored by the volume change J of their tet (blue = squeezed, orange = stretched), as in
the jelly videos; the bars are colored by material, as in gen_curves.png.
"""
import os
import sys

import imageio_ffmpeg
import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont

from distortion.contact import surface

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "scratch", "demo3d")
VID = os.path.join(HERE, "..", "..", "assets", "fig", "vid")
FIG = os.path.join(HERE, "..", "..", "assets", "fig")
RES = os.path.join(HERE, "..", "..", "assets", "research")
ONLY = set(sys.argv[1:])
LO, MID, HI = np.array([0.165, 0.471, 0.839]), np.array([0.94, 0.937, 0.925]), np.array([0.922, 0.408, 0.204])
FONT = "C:/Windows/Fonts/segoeui.ttf"


class Writer:
    def __init__(self, path, fps, poster=None):
        self.path, self.fps, self.poster, self.w, self.n = path, fps, poster, None, 0

    def add(self, img):
        img = np.ascontiguousarray(img[..., :3])
        if self.w is None:
            H, W = img.shape[:2]
            self.H, self.W = H - H % 2, W - W % 2
            self.w = imageio_ffmpeg.write_frames(self.path + ".mp4", (self.W, self.H), fps=self.fps, quality=8,
                                                 macro_block_size=1)
            self.w.send(None)
        img = np.ascontiguousarray(img[:self.H, :self.W])
        self.w.send(img)
        if self.n == self.poster:
            Image.fromarray(img).save(self.path + ".jpg", quality=92)
        self.n += 1

    def close(self):
        self.w.close()
        print(os.path.basename(self.path), self.n, "frames")


def j_colors(X, T, own):
    """Per-face RGB from the volume change of the owning tet."""
    D = np.stack([X[T[:, a]] - X[T[:, 0]] for a in (1, 2, 3)], 2)
    J = np.linalg.det(D) / np.linalg.det(np.stack([X0_CACHE[T[:, a]] - X0_CACHE[T[:, 0]] for a in (1, 2, 3)], 2))
    t = np.clip(np.log2(np.maximum(J, 1e-9))[own] / 0.6, -1, 1)[:, None]
    return (255 * np.where(t < 0, MID + (LO - MID) * -t, MID + (HI - MID) * t)).astype(np.uint8)


X0_CACHE = None


def polys(F):
    return np.hstack([np.full((len(F), 1), 3), F]).ravel()


def label(img, items, size=30):
    """Draw text items [(x, y, text, rgb)] on an image (pixel coordinates)."""
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, size)
    for x, y, t, c in items:
        w = d.textlength(t, font=f)
        d.text((x - w / 2, y), t, fill=c, font=f)
    return np.asarray(im)


# ------------------------------------------------------------------ squeeze
def squeeze_scene(D, size=(900, 760)):
    global X0_CACHE
    X0_CACHE = D["X0"]
    T = D["T"].astype(int)
    F, own = surface(T)
    p = pv.Plotter(off_screen=True, window_size=size)
    p.set_background("white")
    p.add_mesh(pv.Plane(center=(0, -0.002, 0), direction=(0, 1, 0), i_size=4, j_size=3), color="#e2e1dc",
               lighting=False)
    shadow = D["X0"].copy()                                  # the rest pig's footprint, flattened onto the floor
    shadow[:, 1] = 0.003
    p.add_mesh(pv.PolyData(shadow, polys(F)), color="#b9b6ae", lighting=False)
    mesh = pv.PolyData(D["frames"][0].copy(), polys(F))
    mesh.cell_data["rgb"] = j_colors(D["frames"][0], T, own)
    p.add_mesh(mesh, scalars="rgb", rgb=True, ambient=0.3, diffuse=0.8)
    plate = pv.Box(bounds=(-1.35, 1.35, 0, 0.04, -0.85, 0.85))
    pa = p.add_mesh(plate, color="#8f8d87", opacity=0.28, show_edges=True, edge_color="#6b6964", line_width=1.5)
    p.camera_position = [(1.0, 4.2, 2.1), (0.0, 0.15, 0.1), (0, 1, 0)]       # from above: the footprint shows the spread
    p.camera.zoom(1.1)
    return p, mesh, T, own, pa


def vid_squeeze():
    for tag in ["0.00", "0.45"]:
        D = np.load(os.path.join(DATA, f"squeeze_nu{tag}.npz"))
        p, mesh, T, own, pa = squeeze_scene(D)
        fr, plates = D["frames"], D["plates"]
        name = "vid_squeeze_nu0" if tag == "0.00" else "vid_squeeze_nu45"
        W = Writer(os.path.join(VID, name), fps=30, poster=int(np.argmin(plates)))
        seq = list(range(len(fr))) + [len(fr) - 1] * 20          # hold the released pig for a moment
        for k in seq:
            mesh.points = fr[k]
            mesh.cell_data["rgb"][:] = j_colors(fr[k], T, own)
            pa.SetPosition(0, plates[k], 0)
            mesh.Modified()
            p.render()
            W.add(np.asarray(p.screenshot(return_img=True)))
        p.close()
        W.close()


# ------------------------------------------------------------------ cantilever
def body_colors(T, nB, own):
    """Plain colors per body (the beam is gray, the armadillo tan): at nu = 0.4 the J coloring is ~neutral."""
    beam = T[own, 0] < nB
    return np.where(beam[:, None], np.array([196, 194, 188]), np.array([201, 138, 75])).astype(np.uint8)


def cantilever_scene(D, size=(1280, 760)):
    global X0_CACHE
    X0_CACHE = D["X0"]
    T = D["T"].astype(int)
    F, own = surface(T)
    nB = int(D["nB"])
    p = pv.Plotter(off_screen=True, window_size=size)
    p.set_background("white")
    p.add_mesh(pv.Plane(center=(2.2, -0.002, 0), direction=(0, 1, 0), i_size=8, j_size=4), color="#e2e1dc",
               lighting=False)
    p.add_mesh(pv.Box(bounds=(-0.25, 0.0, 0.0, 1.6, -0.45, 0.45)), color="#8f8d87")       # the wall / clamp
    mesh = pv.PolyData(D["frames"][0].copy(), polys(F))
    mesh.cell_data["rgb"] = body_colors(T, nB, own)
    p.add_mesh(mesh, scalars="rgb", rgb=True, ambient=0.3, diffuse=0.8, smooth_shading=False)
    p.camera_position = [(2.0, 1.9, 7.4), (1.9, 0.95, 0.0), (0, 1, 0)]
    return p, mesh, T, own


def vid_cantilever():
    D = np.load(os.path.join(DATA, "cantilever.npz"))
    p, mesh, T, own = cantilever_scene(D)
    fr = D["frames"]
    nB = int(D["nB"])
    poster = int(np.argmin(fr[:, :nB, 1].min(1)))              # the beam's deepest bend
    W = Writer(os.path.join(VID, "vid_cantilever"), fps=int(D["fps"]), poster=poster)
    for X in fr:
        mesh.points = X
        mesh.Modified()
        p.render()
        W.add(np.asarray(p.screenshot(return_img=True)))
    p.close()
    W.close()


# ------------------------------------------------------------------ generated materials: hanging bars
BAR_RGB = [(82, 81, 78), (42, 120, 214), (27, 175, 122), (235, 104, 52), (181, 0, 91)]   # NH, #1, #9, #2, #17


def to_px(p, xyz):
    """World point -> image pixel (x right, y down)."""
    import vtk
    c = vtk.vtkCoordinate()
    c.SetCoordinateSystemToWorld()
    c.SetValue(*xyz)
    x, y = c.GetComputedDoubleDisplayValue(p.renderer)
    return x, p.window_size[1] - y


def vid_hang():
    D = np.load(os.path.join(DATA, "generated_hang.npz"))
    V, T, fr, names, loads = D["V"], D["T"].astype(int), D["frames"], list(D["names"]), D["loads"]
    F, _ = surface(T)
    bottom = V[:, 1] < -1 + 1e-9
    p = pv.Plotter(off_screen=True, window_size=(1280, 820))
    p.set_background("white")
    gap = 0.92
    xs = (np.arange(len(names)) - (len(names) - 1) / 2) * gap
    p.add_mesh(pv.Box(bounds=(xs[0] - 0.35, xs[-1] + 0.35, 0.0, 0.08, -0.3, 0.3)), color="#8f8d87")
    p.add_mesh(pv.Line((xs[0] - 0.35, -1.0, 0.2), (xs[-1] + 0.35, -1.0, 0.2)), color="#b9b7b1", line_width=2)
    meshes, weights = [], []
    for i, x in enumerate(xs):
        m = pv.PolyData(fr[i, 0] + [x, 0, 0], polys(F))
        p.add_mesh(m, color=np.array(BAR_RGB[i]) / 255, ambient=0.3, diffuse=0.8, smooth_shading=False)
        meshes.append(m)
        wb = pv.Box(bounds=(-0.17, 0.17, -0.16, 0.0, -0.17, 0.17))
        weights.append((p.add_mesh(wb, color="#3d3c39"), wb))
    p.enable_parallel_projection()
    p.camera_position = [(0, -1.2, 8.0), (0, -1.2, 0), (0, 1, 0)]
    p.camera.parallel_scale = 1.8
    labels = ["Neo-Hookean"] + [f"generated #{n.strip('#')}" for n in names[1:]]
    W = Writer(os.path.join(RES, "gen_hang"), fps=24, poster=len(loads) - 1 + 12)
    seq = list(range(len(loads))) + [len(loads) - 1] * 36
    p.render()
    lab_px = [to_px(p, (x, 0, 0))[0] for x in xs]
    rest_px = to_px(p, (xs[-1] + 0.35, -1.0, 0.2))
    for k in seq:
        for i, m in enumerate(meshes):
            Y = fr[i, k]
            m.points = Y + [xs[i], 0, 0]
            m.Modified()
            c = Y[bottom].mean(0)
            weights[i][0].SetPosition(xs[i] + c[0], c[1], c[2])
        p.render()
        img = np.asarray(p.screenshot(return_img=True))
        H = img.shape[0]
        items = [(lab_px[i], H - 52, labels[i], BAR_RGB[i]) for i in range(len(labels))]
        items.append((img.shape[1] / 2, 10, f"weight / area = {loads[k]:.2f} E", (82, 81, 78)))
        items.append((rest_px[0] + 70, rest_px[1] - 16, "rest length", (150, 148, 142)))
        W.add(label(img, items, size=26))
    p.close()
    W.close()


# ------------------------------------------------------------------ stills for the master table
def crop(a, pad=10):
    ys, xs = np.nonzero((a[..., :3] < 245).any(-1))
    return a[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]


def cells():
    D = np.load(os.path.join(DATA, "squeeze_nu0.00.npz"))     # nu = 0: the lost volume shows as blue
    p, mesh, T, own, pa = squeeze_scene(D, size=(1100, 1000))
    k = int(np.argmin(D["plates"]))
    mesh.points = D["frames"][k]
    mesh.cell_data["rgb"][:] = j_colors(D["frames"][k], T, own)
    pa.SetPosition(0, D["plates"][k], 0)
    pa.GetProperty().SetOpacity(0.85)                       # a solid-looking plate pressing on the pig
    p.camera_position = [(1.6, 1.5, 6.8), (0.0, 0.25, 0.0), (0, 1, 0)]   # low side view: the squeeze reads
    p.render()
    Image.fromarray(crop(np.asarray(p.screenshot(return_img=True)))).save(os.path.join(FIG, "demo3d_pig_cell.png"))
    p.close()
    D = np.load(os.path.join(DATA, "cantilever.npz"))
    p, mesh, T, own = cantilever_scene(D, size=(1200, 1000))
    fr = D["frames"]
    k = int(np.argmin(fr[:, :int(D["nB"]), 1].min(1)))
    mesh.points = fr[k]
    p.add_mesh(pv.PolyData(fr[0], polys(surface(T)[0])), color="#d9d6cf", opacity=0.22)
    p.render()
    Image.fromarray(crop(np.asarray(p.screenshot(return_img=True)))).save(
        os.path.join(FIG, "demo3d_cantilever_cell.png"))
    p.close()


if __name__ == "__main__":
    for name, f in [("squeeze", vid_squeeze), ("cantilever", vid_cantilever), ("hang", vid_hang), ("cells", cells)]:
        if not ONLY or name in ONLY:
            f()
