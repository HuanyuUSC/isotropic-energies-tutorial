"""Giraffe texture comparison for the 3D-deformation section: a 2D atlas vs. one spherical map.

1. giraffe_atlas_white.png   the 2D atlas texture, white outside the patches, patch boundaries drawn
2. vid/vid_sphere_cover.mp4  the textured sphere turning randomly | the giraffe: the part of the giraffe that is
                             visible on the sphere right now is in full color, parts seen before stay faded,
                             parts not yet seen are grey -- until everything has been covered
3. vid/vid_giraffe_compare.mp4  the deformation video, atlas texture | spherical texture, frame-synchronized

    python figures/fig_giraffe_compare.py [atlas|cover|deform|all]
"""
import os
import pickle
import subprocess
import sys

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fig_demo3d import Writer  # noqa: E402

DATA = os.path.join(HERE, "..", "scratch", "demo3d")
FIG = os.path.join(HERE, "..", "..", "assets", "fig")
VID = os.path.join(FIG, "vid")
WHAT = sys.argv[1] if len(sys.argv) > 1 else "all"


def atlas_white():
    charts, UV = pickle.load(open(os.path.join(DATA, "giraffe_atlas.pkl"), "rb"))
    tex = np.asarray(Image.open(os.path.join(FIG, "giraffe_coat.png")).convert("RGB"))
    extent = max(np.concatenate(UV).max(0)) * 1.001
    res = tex.shape[0]
    mask = Image.new("L", (tex.shape[1], res), 0)
    dr = ImageDraw.Draw(mask)
    edges_all = []
    for (vs, Fl, _, _), X in zip(charts, UV):
        P = X / extent
        Q = np.stack([P[:, 0] * tex.shape[1], (1 - P[:, 1]) * res], 1)     # texture row 0 = v = 1
        for f in Fl:
            dr.polygon([tuple(q) for q in Q[f]], fill=255)
        e = np.sort(np.concatenate([Fl[:, [0, 1]], Fl[:, [1, 2]], Fl[:, [2, 0]]]), 1)
        u, c = np.unique(e, axis=0, return_counts=True)
        edges_all.append(Q[u[c == 1]])
    m = np.asarray(mask) > 0
    out = np.where(m[..., None], tex, 255).astype(np.uint8)
    img = Image.fromarray(out)
    dr = ImageDraw.Draw(img)
    for E in edges_all:
        for a, b in E:
            dr.line([tuple(a), tuple(b)], fill=(30, 30, 30), width=3)
    img = img.resize((1600, 1600), Image.LANCZOS)
    img.save(os.path.join(FIG, "giraffe_atlas_white.png"))
    print("saved giraffe_atlas_white.png", f"({len(charts)} patches, {100 * m.mean():.0f}% of the square used)")


def sphere_coat():
    """The sphere-mapped coat (same as fig_giraffe_sphere.py 'real'), cached."""
    cache = os.path.join(DATA, "giraffe_sphere_coat.npz")
    if os.path.exists(cache):
        C = np.load(cache)
        return C["F"], C["Vs"], C["S"], C["col"]
    sys.argv = [sys.argv[0], "0.00", "real"]
    import fig_giraffe_sphere as g
    G = np.load(os.path.join(DATA, "giraffe_sphere.npz"))
    F0 = G["F"]
    S0 = G["nu_0.00"]
    S0 = S0 / np.linalg.norm(S0, axis=1, keepdims=True)
    Vraw, _ = g.read_off(g.SRC)
    V0 = g.sim_frame(Vraw)
    F, Vs, S = F0, V0, S0
    for _ in range(3):
        F, (Vs, S) = g.subdivide(F, Vs, S)
    S = S / np.linalg.norm(S, axis=1, keepdims=True)
    Nf = pv.PolyData(Vs, np.hstack([np.full((len(F), 1), 3), F]).ravel()).compute_normals(
        cell_normals=False, auto_orient_normals=True)["Normals"]
    col = g.details_real(Vs, g.coat_real(V0, S0, F0, Vs), V0, Nf)
    np.savez_compressed(cache, F=F, Vs=Vs, S=S, col=col)
    return F, Vs, S, col


def random_rotations(n_frames, seed=3):
    """A smooth random tumble that brings every direction of the sphere to the camera: slerp through keyframe
    rotations whose front direction visits the 6 axis directions and a few random ones."""
    from scipy.spatial.transform import Rotation as Rot, Slerp
    rng = np.random.default_rng(seed)
    dirs = [np.array(d, float) for d in [(0, 0, 1), (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]]
    dirs = [dirs[0]] + [dirs[i] for i in rng.permutation(np.arange(1, 6))]
    dirs += [v / np.linalg.norm(v) for v in rng.standard_normal((3, 3))] + [dirs[0]]
    keys = []
    for d in dirs:                                           # the rotation taking direction d to the camera (+z)
        R, _ = Rot.align_vectors([[0, 0, 1]], [d])
        keys.append(Rot.from_rotvec(rng.uniform(-0.6, 0.6) * np.array([0, 0, 1.0])) * R)
    q = Rot.concatenate(keys)
    sl = Slerp(np.linspace(0, 1, len(keys)), q)
    t = np.linspace(0, 1, n_frames)
    t = 0.5 - 0.5 * np.cos(np.pi * t)                        # ease in and out
    return sl(t).as_matrix()


def tumble_with_head(S, head_dir, seed=3):
    """Rotations (one per frame) and a zoom weight per frame: a random tumble that visits every direction, with a
    long, zoomed-in pause in the middle where the sphere turns slowly around the head's region."""
    from scipy.spatial.transform import Rotation as Rot, Slerp
    rng = np.random.default_rng(seed)
    face = lambda d, roll=0.0: Rot.from_rotvec(roll * np.array([0, 0, 1.0])) * Rot.align_vectors([[0, 0, 1]], [d])[0]
    ax = [np.array(d, float) for d in [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]]
    ax = [ax[i] for i in rng.permutation(5)]
    before = [face(np.array([0, 0, 1.0]))] + [face(d, rng.uniform(-0.6, 0.6)) for d in ax[:3]]
    Hd = face(head_dir)
    wob = [Rot.from_rotvec(v) * Hd for v in ([0.35, 0, 0], [0, 0.35, 0], [-0.35, 0, 0], [0, -0.35, 0])]
    after = [face(d, rng.uniform(-0.6, 0.6)) for d in ax[3:]] +             [face(v / np.linalg.norm(v), rng.uniform(-0.6, 0.6)) for v in rng.standard_normal((2, 3))] +             [face(np.array([0, 0, 1.0]))]
    # (key, frames to reach it, zoom at it)
    plan = [(k, 40, 0.0) for k in before[1:]] + [(Hd, 50, 1.0)] + [(k, 45, 1.0) for k in wob] +            [(Hd, 45, 1.0)] + [(k, 40, 0.0) for k in after]
    Rs, Z = [before[0].as_matrix()], [0.0]
    prev, pz = before[0], 0.0
    for key, n, z in plan:
        sl = Slerp([0, 1], Rot.concatenate([prev, key]))
        t = np.linspace(0, 1, n + 1)[1:]
        e = 0.5 - 0.5 * np.cos(np.pi * t)
        Rs += list(sl(e).as_matrix())
        Z += list(pz + (z - pz) * e)
        prev, pz = key, z
    return np.array(Rs), np.array(Z)


def cover():
    F, Vs, S, col = sphere_coat()
    faces = np.hstack([np.full((len(F), 1), 3), F]).ravel()
    D = np.load(os.path.join(DATA, "giraffe_nh.npz"))
    from fig_giraffe import embed
    X0, T = D["X0"], D["T"].astype(int)
    tet, bary = embed(Vs, X0, T)
    P = np.einsum("ij,ijk->ik", bary, X0[T[tet]])            # the giraffe at rest
    H = np.ptp(P[:, 1])
    head = P[:, 1] > P[:, 1].min() + 0.88 * H
    hd = S[head].mean(0); hd /= np.linalg.norm(hd)
    grey = np.array([205, 203, 198], float)
    Rs, Z = tumble_with_head(S, hd)
    p = pv.Plotter(off_screen=True, window_size=(1600, 800), shape=(1, 2), border=False)
    p.set_background("white")
    p.subplot(0, 0)
    ms = pv.PolyData(S.copy(), faces); ms.point_data["rgb"] = col
    p.add_mesh(ms, scalars="rgb", rgb=True, smooth_shading=True, ambient=0.35, diffuse=0.75)
    p.camera_position = [(0, 0, 4.2), (0, 0, 0), (0, 1, 0)]
    cam_s = p.camera
    p.subplot(0, 1)
    mg = pv.PolyData(P, faces); mg.point_data["rgb"] = np.repeat(grey[None], len(P), 0).astype(np.uint8)
    p.add_mesh(mg, scalars="rgb", rgb=True, smooth_shading=True, ambient=0.35, diffuse=0.75)
    p.camera_position = [(4.6, 1.5, 2.2), (0, 0.95, 0.35), (0, 1, 0)]
    p.reset_camera(bounds=mg.bounds)
    p.camera.zoom(1.25)
    cam_g = p.camera
    pos0, foc0 = np.array(cam_g.position), np.array(cam_g.focal_point)
    hc = P[head].mean(0)
    foc1 = hc
    pos1 = hc + (pos0 - foc0) / np.linalg.norm(pos0 - foc0) * 1.25   # the head, close up, same viewing direction
    seen = np.zeros(len(S))
    n_end = 40
    W = Writer(os.path.join(VID, "vid_sphere_cover"), fps=30, poster=len(Rs) + n_end - 1)
    for k in range(len(Rs) + n_end):
        R, z = (Rs[k], Z[k]) if k < len(Rs) else (Rs[-1], 0.0)
        Sr = S @ R.T
        ms.points = Sr
        ms.Modified()
        cam_s.position = (0, 0, 4.2 - 1.0 * z)               # the sphere zooms in on the head's patch too
        cam_g.position = tuple(pos0 + (pos1 - pos0) * z)
        cam_g.focal_point = tuple(foc0 + (foc1 - foc0) * z)
        vis = np.clip((Sr[:, 2] - 0.15) / 0.25, 0, 1)        # facing the camera (soft rim)
        seen = np.maximum(seen, vis)
        if k >= len(Rs):                                     # the end: all of it, full color
            w = np.clip((k - len(Rs)) / 15, 0, 1)
            a = np.maximum(vis, seen * (0.3 + 0.7 * w))
        else:
            a = np.maximum(vis, 0.3 * seen)
        c = grey[None] + (col.astype(float) - grey[None]) * a[:, None]
        mg.point_data["rgb"] = c.astype(np.uint8)
        mg.Modified()
        p.render()
        W.add(np.asarray(p.screenshot(return_img=True)))
    W.close()
    p.close()
    print(f"vid_sphere_cover: {len(Rs) + n_end} frames ({(len(Rs) + n_end) / 30:.1f} s); head pause "
          f"{(Z > 0.99).sum() / 30:.1f} s; covered {100 * (seen > 0.5).mean():.1f}% of the vertices")


def plain_and_atlas():
    """The task (a plain mesh) and the industry answer (an atlas-painted mesh, seams drawn), side views facing
    left like the photo; plus the atlas texture next to it."""
    import fig_giraffe as fg
    V, F = fg.read_off(fg.SRC)
    V = fg.sim_frame(V)
    cam = [(5.5, 1.05, 0.08), (0, 1.0, 0.08), (0, 1, 0)]
    p = pv.Plotter(off_screen=True, window_size=(1000, 1000))
    p.set_background("white")
    p.add_mesh(pv.PolyData(V, np.hstack([np.full((len(F), 1), 3), F]).ravel()), color="#d8d5cf",
               smooth_shading=True, ambient=0.3, diffuse=0.8, specular=0.15)
    p.camera_position = cam
    img = np.asarray(p.screenshot(return_img=True))
    p.close()
    crop_save(img, "giraffe_plain.png")
    charts, UV = pickle.load(open(os.path.join(DATA, "giraffe_atlas.pkl"), "rb"))
    tex = np.asarray(Image.open(os.path.join(FIG, "giraffe_coat.png")).convert("RGB"))
    extent = max(np.concatenate(UV).max(0)) * 1.001
    offs = np.cumsum([0] + [len(c[0]) for c in charts])
    Vr = np.concatenate([V[c[0]] for c in charts])
    Fr = np.concatenate([c[1] + o for c, o in zip(charts, offs[:-1])])
    TC = np.concatenate(UV) / extent
    p = pv.Plotter(off_screen=True, window_size=(1000, 1000))
    p.set_background("white")
    m = pv.PolyData(Vr, np.hstack([np.full((len(Fr), 1), 3), Fr]).ravel())
    m.active_texture_coordinates = TC.astype(np.float32)
    p.add_mesh(m, texture=pv.Texture(tex), smooth_shading=True, ambient=0.35, diffuse=0.75, specular=0.05)
    Nv = pv.PolyData(V, np.hstack([np.full((len(F), 1), 3), F]).ravel()).compute_normals(
        cell_normals=False, auto_orient_normals=True)["Normals"]
    Vo = V + 0.004 * Nv                                       # lift the seam lines just off the surface
    segs = []
    for (vs, Fl, _, _) in charts:                            # patch boundaries, drawn on the mesh
        e = np.sort(np.concatenate([Fl[:, [0, 1]], Fl[:, [1, 2]], Fl[:, [2, 0]]]), 1)
        u, c = np.unique(e, axis=0, return_counts=True)
        segs.append(Vo[vs][u[c == 1]])
    segs = np.concatenate(segs)
    lines = pv.PolyData(segs.reshape(-1, 3), lines=np.hstack([np.full((len(segs), 1), 2),
                                                               np.arange(2 * len(segs)).reshape(-1, 2)]).ravel())
    p.add_mesh(lines, color="#1a1a1a", line_width=2.5, lighting=False)
    p.camera_position = cam
    img = np.asarray(p.screenshot(return_img=True))
    p.close()
    crop_save(img, "giraffe_atlas_mesh.png")


def crop_save(img, name, pad=10):
    ys, xs = np.nonzero((img[..., :3] < 245).any(-1))
    Image.fromarray(img[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]).save(
        os.path.join(FIG, name))
    print("saved", name)


def deform():
    ff = subprocess.run([sys.executable, "-c", "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"],
                        capture_output=True, text=True).stdout.strip()
    a, b = os.path.join(VID, "vid_giraffe.mp4"), os.path.join(VID, "vid_giraffe_sphere.mp4")
    out = os.path.join(VID, "vid_giraffe_compare.mp4")
    crop = "crop=760:500:110:230"                              # 900x760 frames: drop the empty sky
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", a, "-i", b, "-filter_complex",
                    f"[0:v]{crop}[l];[1:v]{crop}[r];[l][r]hstack=inputs=2,format=yuv420p[v]", "-map", "[v]",
                    "-c:v", "libx264", "-crf", "20", "-movflags", "+faststart", out], check=True)
    subprocess.run([ff, "-y", "-loglevel", "error", "-sseof", "-0.1", "-i", out, "-update", "1", "-frames:v", "1",
                    out.replace(".mp4", ".jpg")], check=True)
    # poster at the lowest pose (frame 55)
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", out, "-vf", "select=eq(n\\,55)", "-frames:v", "1",
                    out.replace(".mp4", ".jpg")], check=True)
    print("saved vid_giraffe_compare.mp4")


if __name__ == "__main__":
    if WHAT in ("atlas", "all"):
        atlas_white()
    if WHAT in ("deform", "all"):
        deform()
    if WHAT in ("cover", "all"):
        cover()
    if WHAT in ("plain", "all"):
        plain_and_atlas()
