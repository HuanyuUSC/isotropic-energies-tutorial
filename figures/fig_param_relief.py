"""Parameterizations shown by the shape's own relief: the flattened face (ARAP) and the skull's sphere map
(Symmetric Dirichlet), shaded with the mesh's detail normals (a normal map), with or without a faint wireframe.
A low-distortion map keeps the features recognizable in the plane / on the sphere.
    python figures/fig_param_relief.py       # writes assets/fig/relief_*.png and a comparison sheet"""
import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import fig_sphere_wire as W  # noqa: E402

FIG = os.path.join(HERE, "..", "..", "assets", "fig")
GRAY = (0.86, 0.86, 0.88)


def crop(a, pad=8):
    ys, xs = np.nonzero((a[..., :3] < 245).any(-1))
    return a[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]


def render(X, F, N, cam, wire=None, size=(1100, 1100), parallel=False):
    import pyvista as pv
    p = W.plotter(size)
    m = pv.PolyData(X, np.hstack([np.full((len(F), 1), 3), F]).ravel())
    if N is not None:
        m.point_data["Normals"] = N.astype(np.float32)
        m.point_data.active_normals_name = "Normals"
    else:
        m = m.compute_normals(split_vertices=False)
    kw = dict(color=GRAY, smooth_shading=False, interpolation="gouraud", ambient=0.15, diffuse=0.85,
              specular=0.2, specular_power=25)
    if wire is not None:
        kw.update(show_edges=True, edge_color=wire[0], line_width=wire[1])
    p.add_mesh(m, **kw)
    p.camera_position = cam
    if parallel:
        p.enable_parallel_projection()
        p.reset_camera(); p.camera.zoom(1.25)
    img = p.screenshot(return_img=True)
    p.close()
    return crop(img)


def face():
    d = pickle.load(open(os.path.join(HERE, "..", "scratch", "flatten_face_arap.pkl"), "rb"))
    V, F, X = d["V3"], d["F"], d["pnewton"]["X"]
    # upright: best 2D rotation of the flat face onto the 3D face's front projection (xy)
    A = V[:, :2] - V[:, :2].mean(0); B = X - X.mean(0)
    U, _, Vt = np.linalg.svd(B.T @ A); R = U @ Vt           # reflection allowed: match what the camera sees
    X = B @ R
    N = W.vertex_normals(V, F)
    front = np.sign(N[:, 2].mean()) or 1.0
    Ns = W.smooth(N.copy(), F, 400)
    Nflat = W.rotate_between(Ns, np.tile([0, 0, front], (len(V), 1)), N)
    P = np.c_[X, np.zeros(len(X))]
    # the flat triangles must face the camera, or the renderer flips our normals (two-sided lighting)
    g = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])[:, 2]
    if np.sign(np.median(g)) != front:
        F = F[:, [0, 2, 1]]
    c3 = V.mean(0); c2 = P.mean(0)
    cam3 = [tuple(c3 + np.array([0, 0, 6 * front])), tuple(c3), (0, 1, 0)]
    cam2 = [tuple(c2 + np.array([0, 0, 6 * front])), tuple(c2), (0, 1, 0)]
    out = {"face3d": render(V, F, None, cam3)}
    out["faceflat"] = render(P, F, Nflat, cam2, parallel=True)
    out["faceflat_wire"] = render(P, F, Nflat, cam2, wire=((0.55, 0.55, 0.6), 0.5), parallel=True)
    return out


def skull():
    V, F, S, Nd = W.sphere_data("skull", "symdir")
    # look at the sphere point of the skull's face: the vertex farthest along the skull's front
    d = np.load(os.path.join(FIG, "cache", "skull.npz"))
    out = {}
    front = S[np.argmax(V[:, 2])]
    up = np.array([0, 1.0, 0]) - front * front[1]; up /= np.linalg.norm(up)
    cam = [tuple(4.4 * front), (0, 0, 0), tuple(up)]
    out["skullsphere"] = render(S, F, Nd, cam)
    out["skullsphere_wire"] = render(S, F, Nd, cam, wire=((0.55, 0.55, 0.6), 0.4))
    return out


def main():
    imgs = {**face(), **skull()}
    for k, a in imgs.items():
        plt.imsave(os.path.join(FIG, f"relief_{k}.png"), a)
    # comparison sheet at about twice the title-slide size
    cur = {k: crop(plt.imread(os.path.join(FIG, n))[..., :3] * 255) for k, n in
           (("face", "flatten_pair.png"), ("sphere", "sphere_pair.png"))}
    skull_tex = crop(plt.imread(os.path.join(FIG, "d4_skull_symdir.png"))[..., :3] * 255)
    rows = [("current: checker / Earth", [cur["face"], cur["sphere"]]),
            ("relief only", [imgs["face3d"], imgs["faceflat"], skull_tex, imgs["skullsphere"]]),
            ("relief + faint wireframe", [imgs["face3d"], imgs["faceflat_wire"], skull_tex, imgs["skullsphere_wire"]])]
    fig, axs = plt.subplots(3, 1, figsize=(16, 13))
    for ax, (title, ims) in zip(axs, rows):
        h = 340
        from PIL import Image
        res = [np.asarray(Image.fromarray(np.uint8(np.clip(i if i.max() > 1.5 else i * 255, 0, 255))).resize(
            (int(i.shape[1] * h / i.shape[0]), h), Image.LANCZOS)) for i in ims]
        gap = np.full((h, 40, 3), 255, np.uint8)
        strip = np.hstack(sum([[r, gap] for r in res], [])[:-1])
        ax.imshow(strip); ax.set_title(title, fontsize=16); ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "relief_compare.png"), dpi=90)
    print("done")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- textured 3D mesh + textured, relief-shaded domain
BLUE = np.array([42, 120, 214]); PAPER = np.array([238, 236, 230])
OPACITY = 0.3                                   # texture over a light-gray plaster model


def fade(tex, base=228):
    return (OPACITY * tex.astype(float) + (1 - OPACITY) * base).astype(np.uint8)


def checker_texture(n=12, px=1024):
    t = (np.arange(px) + 0.5) / px * n
    c = (np.floor(t)[:, None] + np.floor(t)[None, :]) % 2
    return np.where(c[..., None] > 0, BLUE, PAPER).astype(np.uint8)


def render_tex(X, F, uv, tex, N=None, cam=None, size=(1100, 1100), parallel=False, corner=False):
    """Textured render; N = per-vertex normals (or per-corner if corner=True); None = the mesh's own."""
    import pyvista as pv
    p = W.plotter(size)
    if corner:
        Xc = X[F].reshape(-1, 3); Fc = np.arange(len(Xc)).reshape(-1, 3)
        m = pv.PolyData(Xc, np.hstack([np.full((len(Fc), 1), 3), Fc]).ravel())
        m.active_texture_coordinates = uv.reshape(-1, 2).astype(np.float32)
        if N is not None:
            m.point_data["Normals"] = N[F].reshape(-1, 3).astype(np.float32)
    else:
        m = pv.PolyData(X, np.hstack([np.full((len(F), 1), 3), F]).ravel())
        m.active_texture_coordinates = uv.astype(np.float32)
        m.point_data["Normals"] = (N if N is not None else W.vertex_normals(X, F)).astype(np.float32)
    if corner and N is None:
        m.point_data["Normals"] = np.repeat(W.vertex_normals(X, F), 1, 0)[F].reshape(-1, 3).astype(np.float32)
    m.point_data.active_normals_name = "Normals"
    t = pv.Texture(np.ascontiguousarray(tex)); t.repeat = True; t.interpolate = True
    p.add_mesh(m, texture=t, smooth_shading=False, interpolation="gouraud", ambient=0.22, diffuse=0.8,
               specular=0.15, specular_power=25)
    p.camera_position = cam
    if parallel:
        p.enable_parallel_projection(); p.reset_camera(); p.camera.zoom(1.25)
    img = p.screenshot(return_img=True)
    p.close()
    return crop(img)


def face_textured(n=12):
    d = pickle.load(open(os.path.join(HERE, "..", "scratch", "flatten_face_arap.pkl"), "rb"))
    V, F, X = d["V3"], d["F"], d["pnewton"]["X"]
    A = V[:, :2] - V[:, :2].mean(0); B = X - X.mean(0)
    U, _, Vt = np.linalg.svd(B.T @ A); X = B @ (U @ Vt)
    N = W.vertex_normals(V, F)
    front = np.sign(N[:, 2].mean()) or 1.0
    Nflat = W.rotate_between(W.smooth(N.copy(), F, 400), np.tile([0, 0, front], (len(V), 1)), N)
    P = np.c_[X, np.zeros(len(X))]
    Fp = F
    g = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])[:, 2]
    if np.sign(np.median(g)) != front:
        Fp = F[:, [0, 2, 1]]
    uv = (X - X.min(0)) / np.ptp(X, 0).max()                 # one checker for both: square cells in the plane
    tex = fade(checker_texture(n))
    c3, c2 = V.mean(0), P.mean(0)
    a = render_tex(V, F, uv, tex, N=N, cam=[tuple(c3 + [0, 0, 6 * front]), tuple(c3), (0, 1, 0)])
    b = render_tex(P, Fp, uv, tex, N=Nflat, cam=[tuple(c2 + [0, 0, 6 * front]), tuple(c2), (0, 1, 0)], parallel=True)
    return a, b


def skull_textured():
    from distortion.render import corner_uv_mesh, earth_texture
    V, F, S, Nd = W.sphere_data("skull", "symdir")
    tex = fade(earth_texture())
    # turn the Earth so Africa sits on the skull's face, north up (as in d4_skull_symdir)
    f = S[np.argmax(V[:, 2])]; top = S[np.argmax(V[:, 1])]
    u = top - f * (f @ top); u /= np.linalg.norm(u)
    lon, lat = np.radians(20), np.radians(5)
    a_ = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    n_ = np.array([0, 0, 1.0]) - a_ * a_[2]; n_ /= np.linalg.norm(n_)
    Rot = np.c_[a_, n_, np.cross(a_, n_)] @ np.c_[f, u, np.cross(f, u)].T
    _, _, uv = corner_uv_mesh(S @ Rot.T, S @ Rot.T, F)
    uv = uv.reshape(-1, 3, 2)
    c, r = V.mean(0), np.ptp(V, 0).max()
    front = S[np.argmax(V[:, 2])]
    up = np.array([0, 1.0, 0]) - front * front[1]; up /= np.linalg.norm(up)
    a = render_tex(V, F, uv, tex, N=W.vertex_normals(V, F), cam=[tuple(c + np.array([0.9, 0.3, 3.0]) * r), tuple(c), (0, 1, 0)],
                   corner=True)
    b = render_tex(S, F, uv, tex, N=Nd, cam=[tuple(4.4 * front), (0, 0, 0), tuple(up)], corner=True)
    return a, b


def textured_pairs():
    for name, (a, b) in (("face", face_textured()), ("skull", skull_textured())):
        plt.imsave(os.path.join(FIG, f"relief_tex_{name}3d_a30.png"), a)
        plt.imsave(os.path.join(FIG, f"relief_tex_{name}dom_a30.png"), b)
    print("done")
