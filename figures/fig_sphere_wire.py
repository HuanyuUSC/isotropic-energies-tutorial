"""Show a spherical parameterization directly: the mesh drawn on the sphere (wireframe), shaded with the
mesh's own surface detail, so features of the shape (fingers, nails, creases) appear on the sphere.
Low distortion = the details look like the shape; high distortion = they are squeezed or blown up.

Detail normals: the mesh normal relative to a heavily smoothed normal field, rotated onto the sphere normal
(a normal map), so the shading stays attached to the sphere when it turns.
    python figures/fig_sphere_wire.py still  [mesh] [maps...]
    python figures/fig_sphere_wire.py video  [mesh] [maps...]"""
import os
import subprocess
import sys

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "..", "assets")
OUT = os.path.join(ASSETS, "fig")


def load(mesh):
    for folder in (os.path.join(ASSETS, "viewer", "cache100"), os.path.join(ASSETS, "fig", "cache")):
        p = os.path.join(folder, f"{mesh}.npz")
        if os.path.exists(p):
            return np.load(p)
    raise FileNotFoundError(mesh)


def vertex_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], n)
    return N / np.linalg.norm(N, axis=1, keepdims=True)


def smooth(X, F, iters):
    n = int(F.max()) + 1
    i = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]])
    j = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
    A = sp.csr_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    A.data[:] = 1
    A = sp.diags(1 / np.asarray(A.sum(1)).ravel()) @ A
    for _ in range(iters):
        X = 0.5 * X + 0.5 * (A @ X)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def rotate_between(a, b, v):
    """Apply, per row, the minimal rotation taking unit a to unit b, to v (Rodrigues)."""
    k = np.cross(a, b)
    s = np.linalg.norm(k, axis=1, keepdims=True)
    c = (a * b).sum(1, keepdims=True)
    k = k / np.maximum(s, 1e-12)
    return v * c + np.cross(k, v) * s + k * (k * v).sum(1, keepdims=True) * (1 - c)


def detail_normals(V, F, S, iters=300, strength=1.0):
    N = vertex_normals(V, F)
    Ns = smooth(N.copy(), F, iters)
    D = rotate_between(Ns, S, N)
    D = S + strength * (D - S)
    return D / np.linalg.norm(D, axis=1, keepdims=True)


def plotter(size):
    import pyvista as pv
    p = pv.Plotter(off_screen=True, window_size=size, lighting="none")
    p.add_light(pv.Light(position=(0.5, 0.7, 1.0), light_type="camera light", intensity=1.0))
    p.add_light(pv.Light(position=(-1.0, -0.2, 0.6), light_type="camera light", intensity=0.3))
    p.set_background("white")
    return p


def add_sphere(p, S, F, Nd, R=np.eye(3), edge_rgb=(0.15, 0.15, 0.18), lw=0.6):
    import pyvista as pv
    X = S @ R.T
    m = pv.PolyData(X, np.hstack([np.full((len(F), 1), 3), F]).ravel())
    m.point_data["Normals"] = (Nd @ R.T).astype(np.float32)
    m.point_data.active_normals_name = "Normals"
    p.add_mesh(m, color=(0.86, 0.87, 0.92), smooth_shading=False, interpolation="gouraud",
               ambient=0.15, diffuse=0.85, specular=0.25, specular_power=25,
               show_edges=True, edge_color=edge_rgb, line_width=lw)
    return m


def sphere_data(mesh, key):
    d = load(mesh)
    V, F = d["V"], d["F"]
    S = d[key] / np.linalg.norm(d[key], axis=1, keepdims=True)
    return V, F, S, detail_normals(V, F, S)


def look_at(view):
    view = view / np.linalg.norm(view)
    up = np.array([0, 0, 1.0]) if abs(view[2]) < 0.95 else np.array([0, 1.0, 0])
    return [tuple(4.4 * view), (0, 0, 0), tuple(up)]


def still(mesh="hand", keys=("conformal", "symdir"), view=None, size=(1100, 1100)):
    imgs = []
    for key in keys:
        V, F, S, Nd = sphere_data(mesh, key)
        # default view: the sphere point of the shape's centroid-ish feature (palm for the hand)
        v = view if view is not None else S[np.argmin(np.linalg.norm(V - V.mean(0), axis=1))]
        p = plotter(size)
        add_sphere(p, S, F, Nd)
        p.camera_position = look_at(np.asarray(v, float))
        img = p.screenshot(return_img=True)
        p.close()
        plt.imsave(os.path.join(OUT, f"sphere_wire_{mesh}_{key}.png"), img)
        imgs.append(img)
        print("wrote", f"sphere_wire_{mesh}_{key}.png")
    return imgs


def spiral(n, turns=3.0):
    """Viewing directions on a pole-to-pole spiral and back (covers the whole sphere), eased at the ends."""
    t = np.linspace(0, 1, n)
    u = 0.5 - 0.5 * np.cos(2 * np.pi * t)                    # 0 -> 1 -> 0
    z = np.cos(np.pi * (0.06 + 0.88 * u))                    # stay a little off the poles
    phi = 2 * np.pi * turns * t
    r = np.sqrt(1 - z * z)
    return np.c_[r * np.cos(phi), r * np.sin(phi), z]


def video(mesh="hand", keys=("conformal", "symdir"), n=480, size=(800, 800), fps=30):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    data = [sphere_data(mesh, k) for k in keys]
    dirs = spiral(n)
    plots = []
    for (_, F, S, Nd) in data:
        p = plotter(size)
        add_sphere(p, S, F, Nd, lw=0.5)
        plots.append(p)
    out = os.path.join(OUT, "vid", f"sphere_wire_{mesh}.mp4")
    W, Hh = size[0] * len(keys), size[1]
    proc = subprocess.Popen([ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{Hh}", "-r", str(fps),
                             "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out],
                            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for k, v in enumerate(dirs):
        frame = []
        for p in plots:
            p.camera_position = look_at(v)
            p.render()
            frame.append(p.screenshot(return_img=True)[:, :, :3])
        proc.stdin.write(np.hstack(frame).astype(np.uint8).tobytes())
        if k % 60 == 0:
            print(k, "/", n)
    proc.stdin.close(); proc.wait()
    for p in plots:
        p.close()
    print("wrote", out)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "still"
    mesh = sys.argv[2] if len(sys.argv) > 2 else "hand"
    keys = tuple(sys.argv[3:]) or ("conformal", "symdir")
    (still if mode == "still" else video)(mesh, keys)


# ---------------------------------------------------------------- aligned video: both cameras look at the same hand point
class SphereLocator:
    """Locate a direction on a spherical map (triangle + barycentric), to carry it to another map of the same mesh."""

    def __init__(self, S, F):
        from scipy.spatial import cKDTree
        self.S, self.F = S, F
        C = S[F].mean(1)
        self.tree = cKDTree(C / np.linalg.norm(C, axis=1, keepdims=True))

    def locate(self, v, k=32):
        _, idx = self.tree.query(v, k)
        best = None
        for t in idx:
            A, B, C = self.S[self.F[t]]
            M = np.c_[A, B, C]
            w = np.linalg.solve(M, v)                    # v = a A + b B + c C (central projection)
            if w.min() >= -1e-9:
                return t, w / w.sum()
            m = w.min()
            if best is None or m > best[0]:
                best = (m, t, w)
        _, t, w = best
        w = np.clip(w, 0, None)
        return t, w / w.sum()

    def carry(self, v, S2):
        t, w = self.locate(v)
        p = w @ S2[self.F[t]]
        return p / np.linalg.norm(p)


def video_aligned(mesh="hand", keys=("conformal", "symdir"), driver="symdir", n=540, size=(800, 800), fps=30):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    data = {k: sphere_data(mesh, k) for k in keys}
    Sd, Fd = data[driver][2], data[driver][1]
    loc = SphereLocator(Sd, Fd)
    dirs = spiral(n)
    # camera up for the driver: toward +z (tangent); carried to the other map by mapping a nearby point
    views = {k: [] for k in keys}
    for v in dirs:
        up = np.array([0, 0, 1.0]) - v * v[2]
        up = up / max(np.linalg.norm(up), 1e-9)
        q = v + 0.02 * up; q /= np.linalg.norm(q)
        for k in keys:
            if k == driver:
                views[k].append((v, up))
            else:
                a = loc.carry(v, data[k][2]); b = loc.carry(q, data[k][2])
                u = b - a; u -= a * (a @ u)
                views[k].append((a, u / max(np.linalg.norm(u), 1e-12)))
    # light temporal smoothing of the carried cameras (the map is only C0 across triangles)
    for k in keys:
        if k == driver:
            continue
        A = np.array([a for a, _ in views[k]]); U = np.array([u for _, u in views[k]])
        ker = np.ones(9) / 9
        A = np.stack([np.convolve(np.pad(A[:, i], 4, mode="edge"), ker, "valid") for i in range(3)], 1)
        U = np.stack([np.convolve(np.pad(U[:, i], 4, mode="edge"), ker, "valid") for i in range(3)], 1)
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        U -= A * (A * U).sum(1, keepdims=True); U /= np.linalg.norm(U, axis=1, keepdims=True)
        views[k] = list(zip(A, U))
    plots = {}
    for k in keys:
        _, F, S, Nd = data[k]
        p = plotter(size)
        add_sphere(p, S, F, Nd, lw=0.5)
        plots[k] = p
    out = os.path.join(OUT, "vid", f"sphere_wire_{mesh}_aligned.mp4")
    W, Hh = size[0] * len(keys), size[1]
    proc = subprocess.Popen([ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{Hh}", "-r", str(fps),
                             "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "slow", out],
                            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for f in range(n):
        frame = []
        for k in keys:
            a, u = views[k][f]
            p = plots[k]
            p.camera_position = [tuple(4.4 * a), (0, 0, 0), tuple(u)]
            p.render()
            frame.append(p.screenshot(return_img=True)[:, :, :3])
        proc.stdin.write(np.hstack(frame).astype(np.uint8).tobytes())
        if f % 90 == 0:
            print(f, "/", n, flush=True)
    proc.stdin.close(); proc.wait()
    for p in plots.values():
        p.close()
    print("wrote", out)


# ---------------------------------------------------------------- slide video: aligned spheres + reference hands
FINGER_RGB = np.array([[42, 120, 214], [235, 104, 52], [27, 175, 122], [150, 80, 200], [200, 170, 30]], float) / 255
BASE_RGB = np.array([0.86, 0.87, 0.92])


def tint(lab, amount=0.55):
    c = np.tile(BASE_RGB, (len(lab), 1))
    m = lab >= 0
    c[m] = BASE_RGB + amount * (FINGER_RGB[lab[m]] - BASE_RGB)
    return c


def colored_mesh(X, F, rgb, normals=None):
    import pyvista as pv
    m = pv.PolyData(X, np.hstack([np.full((len(F), 1), 3), F]).ravel())
    m.point_data["rgb"] = (255 * rgb).astype(np.uint8)
    if normals is not None:
        m.point_data["Normals"] = normals.astype(np.float32)
        m.point_data.active_normals_name = "Normals"
    return m


def label_text(img, text, xy, size=34):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    w = dr.textlength(text, font=font)
    dr.text((xy[0] - w / 2, xy[1]), text, fill=(40, 40, 48), font=font)
    return np.asarray(im)


def video_slide(mesh="hand", keys=("conformal", "symdir"), titles=("Conformal", "Symmetric Dirichlet"),
                driver="symdir", n=540, S_SIZE=760, H_SIZE=(380, 330), fps=30):
    import imageio_ffmpeg
    import pyvista as pv
    sys.path.insert(0, HERE)
    from hand_fingers import fingers
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    d = load(mesh)
    V, F = d["V"], d["F"]
    lab = fingers(V, F)[0]
    base = tint(lab, 0.9)
    data = {k: sphere_data(mesh, k) for k in keys}
    # cameras: driver spiral, other carried to the same hand point (as in video_aligned)
    loc = SphereLocator(data[driver][2], F)
    views = {k: [] for k in keys}
    for v in spiral(n):
        up = np.array([0, 0, 1.0]) - v * v[2]; up /= max(np.linalg.norm(up), 1e-9)
        q = v + 0.02 * up; q /= np.linalg.norm(q)
        for k in keys:
            if k == driver:
                views[k].append((v, up))
            else:
                a = loc.carry(v, data[k][2]); b = loc.carry(q, data[k][2])
                u = b - a; u -= a * (a @ u)
                views[k].append((a, u / max(np.linalg.norm(u), 1e-12)))
    for k in keys:
        if k == driver:
            continue
        A = np.array([a for a, _ in views[k]]); U = np.array([u for _, u in views[k]])
        ker = np.ones(9) / 9
        A = np.stack([np.convolve(np.pad(A[:, i], 4, mode="edge"), ker, "valid") for i in range(3)], 1)
        U = np.stack([np.convolve(np.pad(U[:, i], 4, mode="edge"), ker, "valid") for i in range(3)], 1)
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        U -= A * (A * U).sum(1, keepdims=True); U /= np.linalg.norm(U, axis=1, keepdims=True)
        views[k] = list(zip(A, U))
    # plotters: one sphere per map; reference hand (palm + back) per map
    sph, hands, hmesh = {}, {}, {}
    c = V.mean(0)
    hand_cams = [[tuple(c + np.array([0, 0.3, 6.6])), tuple(c), (0, 1, 0)],
                 [tuple(c + np.array([0, 0.3, -6.6])), tuple(c), (0, 1, 0)]]
    Nh = vertex_normals(V, F)
    for k in keys:
        _, _, S, Nd = data[k]
        p = plotter((S_SIZE, S_SIZE))
        p.add_mesh(colored_mesh(S, F, base, Nd), scalars="rgb", rgb=True, smooth_shading=False, interpolation="gouraud",
                   ambient=0.15, diffuse=0.85, specular=0.25, specular_power=25,
                   show_edges=True, edge_color=(0.38, 0.38, 0.42), line_width=0.4)
        sph[k] = p
        hands[k] = []
        for cam in hand_cams:
            ph_ = plotter(H_SIZE)
            m = colored_mesh(V, F, base, Nh)
            ph_.add_mesh(m, scalars="rgb", rgb=True, smooth_shading=False, interpolation="gouraud",
                         ambient=0.2, diffuse=0.8, specular=0.1)
            ph_.camera_position = cam
            hands[k].append((ph_, m))
    W = S_SIZE * len(keys)
    Hh = S_SIZE + H_SIZE[1] + 50
    out = os.path.join(OUT, "vid", f"sphere_wire_{mesh}_slide.mp4")
    proc = subprocess.Popen([ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{Hh}", "-r", str(fps),
                             "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "slow", out],
                            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    dist = 4.4
    for f in range(n):
        cols = []
        for k, title in zip(keys, titles):
            a, u = views[k][f]
            p = sph[k]
            p.camera_position = [tuple(dist * a), (0, 0, 0), tuple(u)]
            p.render()
            top = p.screenshot(return_img=True)[:, :, :3]
            # visible part of this sphere -> full color on the hand, the rest faded
            S = data[k][2]
            vis = (S @ a) > 1.0 / dist + 0.02
            rgb = np.where(vis[:, None], base, 0.12 * base + 0.88 * np.array([0.98, 0.98, 0.97]))
            rgb[vis & (lab < 0)] = np.array([0.66, 0.68, 0.76])            # visible palm: clearly darker than faded
            row = []
            for ph_, m in hands[k]:
                m.point_data["rgb"][:] = (255 * rgb).astype(np.uint8)
                m.Modified()
                ph_.render()
                row.append(ph_.screenshot(return_img=True)[:, :, :3])
            bottom = np.hstack(row)
            col = np.vstack([np.full((50, S_SIZE, 3), 255, np.uint8), top, bottom])
            col = label_text(col, title, (S_SIZE // 2, 8))
            cols.append(col)
        proc.stdin.write(np.hstack(cols).astype(np.uint8).tobytes())
        if f % 90 == 0:
            print(f, "/", n, flush=True)
    proc.stdin.close(); proc.wait()
    for k in keys:
        sph[k].close()
        for ph_, _ in hands[k]:
            ph_.close()
    print("wrote", out)
