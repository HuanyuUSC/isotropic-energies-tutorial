"""Video of the two-cantilever pile-up (scratch/pile/pile.py): beams from the simulated tets, each toy as its
textured hi-res surface carried by the simulated tets (barycentric embedding, scratch/pile/embed_toys.py).
Works on a partial pile.npz too (renders what is there).
    python figures/fig_pile.py [frames: all | a:b | k1,k2,...]   (a list writes stills instead of a video)"""
import os
import sys

import numpy as np
import pyvista as pv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "src")]
from distortion.contact import surface  # noqa: E402

PILE = os.path.join(HERE, "..", "scratch", "pile")
FIG = os.path.join(HERE, "..", "..", "assets", "fig")
VID = os.path.join(FIG, "vid")
FLOOR, BEAM, WALL = "#e2e1dc", (0.72, 0.71, 0.69), "#5f5d58"


def polys(F):
    return np.hstack([np.full((len(F), 1), 3), F]).ravel()


def load():
    D = np.load(os.path.join(PILE, os.environ.get("PILE_NPZ", "pile.npz")))
    names = [str(n) for n in D["names"]]
    offs, sizes = D["offsets"], D["sizes"]
    T = D["T"].astype(np.int64)
    toys = []
    for k, n in enumerate(names):
        if n == "beam":
            continue
        E = np.load(os.path.join(PILE, f"embed_{n}.npz"))
        idx = offs[k] + _local_T(D, offs[k], sizes[k], T)[E["tet"]]          # (m, 4) global vertex ids
        toys.append(dict(name=n, F=E["F"], idx=idx, bary=E["bary"], rgb=E["rgb"]))
    beams = [(offs[k], sizes[k]) for k, n in enumerate(names) if n == "beam"]
    return D, toys, beams, T


def _local_T(D, off, n, T):
    m = (T >= off).all(1) & (T < off + n).all(1)
    return T[m] - off


def scene(D, toys, beams, T, size=(1600, 900)):
    p = pv.Plotter(off_screen=True, window_size=size, lighting="none")
    p.add_light(pv.Light(position=(0.4, 1.0, 0.8), light_type="camera light", intensity=0.95))
    p.add_light(pv.Light(position=(-0.8, 0.3, 0.6), light_type="camera light", intensity=0.35))
    p.set_background("white")
    X0 = D["frames"][0]
    W = np.abs(X0[:, 0]).max() + 0.5
    p.add_mesh(pv.Box(bounds=(-W, W, -0.03, 0.0, -1.5, 1.5)), color=FLOOR, ambient=0.6, diffuse=0.4)
    meshes = []
    for off, n in beams:
        Tb = _local_T(D, off, n, T)
        Fb, _ = surface(Tb)
        m = pv.PolyData(X0[off:off + n].copy(), polys(Fb))
        p.add_mesh(m, color=BEAM, ambient=0.25, diffuse=0.8, specular=0.1)
        meshes.append(("beam", m, off, n))
        x = X0[off:off + n, 0]
        clamp = x.min() if x.mean() < 0 else x.max()                       # the clamped (outer) end
        y1 = X0[off:off + n, 1].max()
        z = np.abs(X0[off:off + n, 2]).max() + 0.12
        p.add_mesh(pv.Box(bounds=(clamp - 0.12 if clamp < 0 else clamp, clamp if clamp < 0 else clamp + 0.12,
                                  0.0, y1 + 0.05, -z, z)), color=WALL, ambient=0.3, diffuse=0.7)
    # the bin's walls (scratch/pile/pile.py): low translucent panels, so it's clear what keeps the toys together
    sys.path.insert(0, PILE)
    _argv, sys.argv = sys.argv, sys.argv[:1]          # pile.py parses argv at import: hide ours from it
    import pile as _pile
    sys.argv = _argv
    WX, WZ, Hw = _pile.TIP + _pile.L + 0.25, 1.3, 0.6
    for b in [(-WX - 0.04, -WX, 0, Hw, -WZ, WZ), (WX, WX + 0.04, 0, Hw, -WZ, WZ), (-WX, WX, 0, Hw, -WZ - 0.04, -WZ)]:
        p.add_mesh(pv.Box(bounds=b), color="#9fb4c8", opacity=0.35, ambient=0.5, diffuse=0.5)
    for t in toys:
        P = (D["frames"][0][t["idx"]] * t["bary"][..., None]).sum(1)
        m = pv.PolyData(P, polys(t["F"]))
        m.point_data["rgb"] = t["rgb"]
        p.add_mesh(m, scalars="rgb", rgb=True, smooth_shading=True, ambient=0.28, diffuse=0.8, specular=0.12,
                   specular_power=20)
        meshes.append(("toy", m, t, None))
    top = X0[:, 1].max()
    p.camera_position = [(0.0, 3.2, 12.5), (0.0, 2.55, 0.0), (0, 1, 0)]    # whole columns (5.3 m) in view
    p.camera.view_angle = 30.0
    return p, meshes


def update(meshes, X):
    for kind, m, a, b in meshes:
        if kind == "beam":
            m.points = X[a:a + b]
        else:
            m.points = (X[a["idx"]] * a["bary"][..., None]).sum(1)
            m.point_data.pop("Normals", None)
        m.Modified()


def main():
    D, toys, beams, T = load()
    fr = D["frames"]
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    p, meshes = scene(D, toys, beams, T)
    if "," in arg or arg.isdigit():
        from PIL import Image
        for k in [int(s) for s in arg.split(",")]:
            update(meshes, fr[min(k, len(fr) - 1)])
            p.render()
            Image.fromarray(p.screenshot(return_img=True)).save(os.path.join(PILE, f"still_{k:04d}.png"))
        print("stills written", len(fr), "frames available")
        return
    a, b = (0, len(fr)) if arg == "all" else map(int, arg.split(":"))
    from fig_demo3d import Writer
    W = Writer(os.path.join(VID, os.environ.get("PILE_OUT", "vid_pile")), fps=int(D["fps"]), poster=min(len(fr) - 1, b - 1))
    for k in range(a, min(b, len(fr))):
        update(meshes, fr[k])
        u = min(1.0, k / (1.5 * float(D["fps"])))        # camera: wide (full columns) -> close on the pile over 1.5 s
        u = u * u * (3 - 2 * u)
        ty, dist, ey = 2.55 + (1.2 - 2.55) * u, 12.5 + (9.0 - 12.5) * u, 3.2 + (3.4 - 3.2) * u
        p.camera_position = [(0.0, ey, dist), (0.0, ty, 0.0), (0, 1, 0)]
        p.render()
        W.add(np.asarray(p.screenshot(return_img=True)))
    p.close()
    W.close()
    print("video", b - a, "frames")


if __name__ == "__main__":
    main()
