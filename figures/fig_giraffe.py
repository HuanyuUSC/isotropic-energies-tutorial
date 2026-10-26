"""The 3D deformation cell: a textured giraffe lowers its head. Parameterization + deformation together:

1. UV atlas of the giraffe surface with the talk's own pipeline (distortion/atlas.py): charts grown and split
   until each is a disk with low distortion, flattened by Symmetric Dirichlet (projected Newton), packed.
2. A reticulated giraffe coat painted procedurally in 2D texture space (Voronoi patches, cream lines).
   The atlas is nearly isometric, so the patches come out the same size all over the body.
3. The textured surface (the original mesh, 9k vertices) rides inside the simulated tets (barycentric
   embedding in the rest shape) from scratch/demo3d/giraffe_nh.npz.

    python figures/fig_giraffe.py       -> assets/fig/vid/vid_giraffe.mp4 (+ .jpg), assets/fig/giraffe_atlas.png,
                                           assets/fig/demo3d_giraffe_cell.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.collections import PolyCollection
from PIL import Image
from scipy.spatial import cKDTree

from distortion import style
from distortion.atlas import flatten, grow_charts, is_disk, pack
from distortion.meshes import read_off

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "scratch", "demo3d")
FIG = os.path.join(HERE, "..", "..", "assets", "fig")
VID = os.path.join(FIG, "vid")
SRC = "F:/Dropbox/3Dmodels/100 selected meshes/100 selected meshes/giraffe.off"
style.use()
sys.path.insert(0, HERE)
from fig_demo3d import Writer  # noqa: E402


def sim_frame(V):
    """The same normalization as scratch/demo3d/giraffe.py, computed from the raw tet mesh."""
    raw = np.load(os.path.join(DATA, "giraffe_tet.npz"))["V"].astype(float)
    lo, s = raw.min(0), 2.0 / np.ptp(raw[:, 1])
    c = ((raw - lo) * s)[:, [0, 2]].mean(0)
    W = (V - lo) * s
    W[:, [0, 2]] -= c
    return W


# ---------------------------------------------------------------- 1. atlas
def atlas(V, F, K=10):
    def chart_of(fs):
        vs, Fl = np.unique(F[fs], return_inverse=True)
        return vs, Fl.reshape(-1, 3)

    def components(Fl):
        import scipy.sparse as sp
        from scipy.sparse.csgraph import connected_components
        n = Fl.max() + 1
        e = np.concatenate([Fl[:, [0, 1]], Fl[:, [1, 2]]])
        _, lab_v = connected_components(sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)))
        return lab_v[Fl[:, 0]]

    def build(fs, depth=0):
        vs, Fl = chart_of(fs)
        comp = components(Fl)
        if comp.max() > 0:                                      # disconnected: one chart per component
            return sum((build(fs[comp == c], depth) for c in range(comp.max() + 1)), [])
        disk, X, q = is_disk(Fl), None, np.inf
        if disk:
            X, E = flatten(V[vs], Fl)
            s = np.linalg.svd(E.deformation(X), compute_uv=False)
            q = (E.A / E.A.sum()) @ (s[:, 0] / s[:, 1])
            if q < 1.6 or depth >= 6 or len(fs) < 150:          # good enough (or small: accept)
                return [(vs, Fl, X, q)]
        assert depth < 12, "could not split into disks"
        sub = grow_charts(V[vs], Fl, 2)                          # not a disk, or too distorted: split in two
        return build(fs[sub == 0], depth + 1) + build(fs[sub == 1], depth + 1)

    lab = grow_charts(V, F, K)
    charts = []
    for c in range(K):
        charts += build(np.nonzero(lab == c)[0])
    UV = pack([c[2] for c in charts], gap=0.02)
    print(f"{len(charts)} charts, area-weighted mean sigma1/sigma2 per chart: "
          f"{min(c[3] for c in charts):.3f} .. {max(c[3] for c in charts):.3f}")
    return charts, UV


# ---------------------------------------------------------------- 2. the coat, painted in 2D
def coat(extent, res=4096, patch=0.11, line=0.012, seed=5):
    """Reticulated pattern on [0, extent]^2 (texture units = world units): jittered-grid Voronoi cells,
    each a slightly different brown, separated by cream lines of width `line`."""
    rng = np.random.default_rng(seed)
    n = int(np.ceil(extent / patch)) + 2
    gx, gy = np.meshgrid(np.arange(n), np.arange(n))
    P = (np.c_[gx.ravel(), gy.ravel()] + rng.uniform(0.15, 0.85, (n * n, 2)) - 1) * patch
    tree = cKDTree(P)
    u = (np.arange(res) + 0.5) / res * extent
    U, Vv = np.meshgrid(u, u)
    q = np.c_[U.ravel(), Vv.ravel()]
    d, i = tree.query(q, k=2)
    edge = (d[:, 1] - d[:, 0]) < line                           # near the bisector of the two nearest cells
    browns = np.array([[166, 92, 40], [178, 104, 46], [152, 84, 38], [184, 112, 52], [160, 96, 44]])
    col = browns[rng.integers(0, len(browns), len(P))][i[:, 0]]
    shade = 1 + 0.06 * rng.standard_normal(len(P))[i[:, 0], None]
    col = np.clip(col * shade, 0, 255)
    cream = np.array([238, 222, 190])
    img = np.where(edge[:, None], cream, col).reshape(res, res, 3).astype(np.uint8)
    return img[::-1]                                            # image row 0 is v = extent


# ---------------------------------------------------------------- 2b. details, baked into the texture
def landmarks(V):
    """Eye centers (outermost head points a third of the way from the ears to the muzzle), nostrils, mouth height."""
    head = V[:, 1] > 1.76
    eyes = []
    for sgn in (1, -1):
        band = head & (np.abs(V[:, 2] - 0.525) < 0.03) & (np.abs(V[:, 1] - 1.91) < 0.035) & (sgn * V[:, 0] > 0)
        eyes.append(V[band][np.argmax(sgn * V[band, 0])])
    tip = V[head][np.argmax(V[head, 2])]
    nostrils = [tip + [sgn * 0.024, 0.016, -0.012] for sgn in (1, -1)]
    return np.array(eyes), np.array(nostrils), tip


def detail_color(P, base, marks):
    """Paint over the coat at rest-pose points P (n, 3): returns new RGB (n, 3)."""
    eyes, nostrils, tip = marks
    out = base.astype(float)

    def blend(mask_alpha, rgb):
        a = np.clip(mask_alpha, 0, 1)[:, None]
        out[:] = out * (1 - a) + np.asarray(rgb, float) * a

    y, z, ax = P[:, 1], P[:, 2], np.abs(P[:, 0])
    blend((0.30 - y) / 0.22 * 0.85 * (y > 0.02) * (z > -0.5), [236, 222, 196])          # pale lower legs
    blend((0.06 - y) / 0.02, [42, 32, 26])                                                 # dark hooves
    blend(((z > 0.62) & (y > 1.74)) * np.clip((z - 0.62) / 0.03, 0, 1), [196, 150, 104])  # plain muzzle
    blend((ax > 0.072) * (y > 1.885) * (z > 0.33) * (z < 0.47), [236, 224, 200])          # ears, cream
    blend((y - 1.955) / 0.02, [66, 48, 36])                                                # ossicone tips
    blend((z < -0.46) * (0.66 - y) / 0.06, [40, 30, 24])                                   # tail tuft
    for e in eyes:                                                                         # eyes: lid, iris, glint
        d = np.linalg.norm(P - e, axis=1)
        blend((0.028 - d) / 0.004, [92, 56, 32])
        blend((0.020 - d) / 0.003, [22, 18, 16])
        glint = e + [0.0, 0.007, 0.006]
        blend((0.0055 - np.linalg.norm(P - glint, axis=1)) / 0.002, [250, 250, 245])
    for n in nostrils:
        blend((0.010 - np.linalg.norm(P - n, axis=1)) / 0.003, [60, 40, 30])
    mouth = (z > tip[2] - 0.075) & (np.abs(y - (tip[1] - 0.022)) < 0.004) & (ax < 0.04)    # the mouth line
    blend(mouth.astype(float), [70, 44, 32])
    return out.astype(np.uint8)


def bake_details(tex, Vr, Fr, TC, marks):
    """For every texel covered by a triangle: its rest-pose 3D point by barycentric interpolation, then
    detail_color. The texture's row 0 is v = 1."""
    res = tex.shape[0]
    tex = tex.copy()
    uv = TC * res
    for f in Fr:
        a, b, c = uv[f]
        lo = np.floor(np.minimum(np.minimum(a, b), c)).astype(int)
        hi = np.ceil(np.maximum(np.maximum(a, b), c)).astype(int)
        xs, ys = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        q = np.c_[xs.ravel() + 0.5, ys.ravel() + 0.5]
        M = np.c_[b - a, c - a]
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        l = np.linalg.solve(M, (q - a).T).T
        w = np.c_[1 - l.sum(1), l]
        inside = (w > -0.02).all(1)                                # a little margin: no gaps along chart edges
        if not inside.any():
            continue
        P = w[inside] @ Vr[f]
        r = res - 1 - np.clip(ys.ravel()[inside], 0, res - 1)
        cc = np.clip(xs.ravel()[inside], 0, res - 1)
        tex[r, cc] = detail_color(P, tex[r, cc], marks)
    return tex


# ---------------------------------------------------------------- 3. embedding in the tets
def embed(P, X0, T):
    """For every point, a containing tet (or the nearest one, extrapolated) and its barycentric coords."""
    C = X0[T].mean(1)
    tree = cKDTree(C)
    _, cand = tree.query(P, k=24)
    best_t = np.zeros(len(P), int)
    best_b = np.zeros((len(P), 4))
    best_m = np.full(len(P), -np.inf)
    for j in range(cand.shape[1]):
        t = cand[:, j]
        A = np.stack([X0[T[t, 1]] - X0[T[t, 0]], X0[T[t, 2]] - X0[T[t, 0]], X0[T[t, 3]] - X0[T[t, 0]]], 2)
        l = np.linalg.solve(A, (P - X0[T[t, 0]])[..., None])[..., 0]
        b = np.c_[1 - l.sum(1), l]
        m = b.min(1)
        better = m > best_m
        best_m[better], best_t[better], best_b[better] = m[better], t[better], b[better]
    print(f"embedding: {np.mean(best_m >= -1e-9) * 100:.1f}% of surface points inside a tet, worst {best_m.min():.3f}")
    return best_t, best_b


def main():
    V, F = read_off(SRC)
    V = sim_frame(V)
    import pickle
    cache = os.path.join(DATA, "giraffe_atlas.pkl")
    if os.path.exists(cache):
        charts, UV = pickle.load(open(cache, "rb"))
    else:
        charts, UV = atlas(V, F)
        pickle.dump((charts, UV), open(cache, "wb"))
    extent = max(np.concatenate(UV).max(0)) * 1.001
    tex = coat(extent)
    Image.fromarray(tex).save(os.path.join(FIG, "giraffe_coat.png"))

    # one render mesh: chart vertices duplicated along the seams
    offs = np.cumsum([0] + [len(c[0]) for c in charts])
    Vr = np.concatenate([V[c[0]] for c in charts])
    Fr = np.concatenate([c[1] + o for c, o in zip(charts, offs[:-1])])
    TC = np.concatenate(UV) / extent
    tex = bake_details(tex, Vr, Fr, TC, landmarks(V))
    Image.fromarray(tex).save(os.path.join(FIG, "giraffe_coat.png"))
    D = np.load(os.path.join(DATA, "giraffe_nh.npz"))
    X0, T, fr = D["X0"], D["T"].astype(int), D["frames"]
    tet, bary = embed(Vr, X0, T)
    pose = lambda X: np.einsum("ij,ijk->ik", bary, X[T[tet]])

    # the 2D atlas, textured (for a slide or backup)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(tex, extent=(0, 1, 0, 1))
    for (vs, Fl, _, _), X in zip(charts, UV):
        ax.add_collection(PolyCollection(X[Fl] / extent, fc="none", ec=(1, 1, 1, 0.0), lw=0))
        b = np.unique(np.sort(np.concatenate([Fl[:, [0, 1]], Fl[:, [1, 2]], Fl[:, [2, 0]]]), 1), axis=0,
                      return_counts=True)
        edges = b[0][b[1] == 1]
        ax.plot(*(X[edges].transpose(2, 1, 0) / extent), color="black", lw=0.6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(os.path.join(FIG, "giraffe_atlas.png"), bbox_inches="tight")
    plt.close(fig)

    p = pv.Plotter(off_screen=True, window_size=(900, 760))
    p.set_background("white")
    p.add_mesh(pv.Plane(center=(0, -0.002, 0.4), direction=(0, 1, 0), i_size=3, j_size=3.5), color="#e2e1dc",
               lighting=False)
    m = pv.PolyData(pose(fr[0]), np.hstack([np.full((len(Fr), 1), 3), Fr]).ravel())
    m.active_texture_coordinates = TC.astype(np.float32)
    p.add_mesh(m, texture=pv.Texture(tex), smooth_shading=True, ambient=0.35, diffuse=0.75, specular=0.05)
    p.camera_position = [(4.6, 1.5, 2.2), (0, 0.95, 0.35), (0, 1, 0)]
    W = Writer(os.path.join(VID, "vid_giraffe"), fps=30, poster=len(fr) - 1)
    seq = list(range(len(fr))) + [len(fr) - 1] * 20 + list(range(len(fr) - 1, -1, -1)) + [0] * 10
    for k in seq:
        m.points = pose(fr[k])
        m.Modified()
        p.render()
        W.add(np.asarray(p.screenshot(return_img=True)))
    W.close()
    # master-table still: the lowered head, with a faint ghost of the rest pose
    m.points = pose(fr[-1])
    p.add_mesh(pv.PolyData(pose(fr[0]), np.hstack([np.full((len(Fr), 1), 3), Fr]).ravel()), color="#d9d6cf",
               opacity=0.22)
    p.render()
    img = np.asarray(p.screenshot(return_img=True))
    ys, xs = np.nonzero((img[..., :3] < 245).any(-1))
    Image.fromarray(img[max(ys.min() - 10, 0):ys.max() + 10, max(xs.min() - 10, 0):xs.max() + 10]).save(
        os.path.join(FIG, "demo3d_giraffe_cell.png"))
    p.close()


if __name__ == "__main__":
    main()
