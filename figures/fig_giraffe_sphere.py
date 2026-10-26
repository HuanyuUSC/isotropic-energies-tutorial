"""The 3D deformation cell, sphere-textured: the giraffe's coat is painted ON THE SPHERE and carried to the
giraffe by our own spherical map (scratch/demo3d/giraffe_sphere.py: flip-free, SymDir + volume term, so area is
nearly preserved and the patches keep one size). No seams: a sphere map of a genus-0 surface needs no cuts.

    python figures/fig_giraffe_sphere.py [nu]   -> assets/fig/vid/vid_giraffe_sphere.mp4 (+ .jpg),
                                                  assets/fig/giraffe_sphere_pair.png (sphere | giraffe)
"""
import os
import sys

import numpy as np
import pyvista as pv
from PIL import Image
from scipy.spatial import cKDTree

from distortion.meshes import read_off

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fig_demo3d import Writer  # noqa: E402
from fig_giraffe import DATA, FIG, SRC, VID, embed, sim_frame  # noqa: E402

NU = sys.argv[1] if len(sys.argv) > 1 else "0.45"


def subdivide(F, *fields):
    """One 1-to-4 midpoint subdivision, applied to every per-vertex field (positions on the giraffe and on
    the sphere alike)."""
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    uE, inv = np.unique(E, axis=0, return_inverse=True)
    n = len(fields[0])
    mid = n + inv.reshape(3, -1).T                               # midpoints of edges (01, 12, 20) of each face
    a, b, c = F.T
    m01, m12, m20 = mid.T
    F2 = np.concatenate([np.c_[a, m01, m20], np.c_[m01, b, m12], np.c_[m20, m12, c], np.c_[m01, m12, m20]])
    out = [np.vstack([X, 0.5 * (X[uE[:, 0]] + X[uE[:, 1]])]) for X in fields]
    return F2, out


def coat_on_sphere(S, n_patches=900, line=0.11, seed=5):
    """Per-vertex colors of a reticulated pattern on the unit sphere: jittered Fibonacci seeds (uniform on
    the sphere, hence ~uniform on the giraffe for an area-preserving map), cream lines along the bisectors."""
    rng = np.random.default_rng(seed)
    k = np.arange(n_patches) + 0.5
    phi = np.arccos(1 - 2 * k / n_patches)
    th = np.pi * (1 + 5 ** 0.5) * k
    P = np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]
    P += 0.35 * np.sqrt(4 * np.pi / n_patches) * rng.standard_normal(P.shape)   # jitter, ~1/3 patch
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    d, i = cKDTree(P).query(S, k=2)
    edge = (d[:, 1] - d[:, 0]) < line * np.sqrt(4 * np.pi / n_patches)   # lines ~11% of the patch spacing
    browns = np.array([[166, 92, 40], [178, 104, 46], [152, 84, 38], [184, 112, 52], [160, 96, 44]])
    col = browns[rng.integers(0, len(browns), len(P))][i[:, 0]] * (1 + 0.06 * rng.standard_normal(len(P)))[i[:, 0], None]
    return np.where(edge[:, None], [238, 222, 190], np.clip(col, 0, 255)).astype(np.uint8), P


def save_png(img, name):
    """Save into assets/fig; if the file is locked (open in a viewer, Dropbox syncing), save as name_new.png."""
    path = os.path.join(FIG, name)
    try:
        Image.fromarray(img).save(path)
    except OSError:
        path = path.replace(".png", "_new.png")
        Image.fromarray(img).save(path)
    print("saved", os.path.basename(path))


def tri_area(X, F):
    return 0.5 * np.linalg.norm(np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]), axis=1)


def coat_adaptive(V, S, F, Vf, Sf, kf, patch=0.11, line=0.012, seed=5):
    """The coat painted on the sphere, sized for the giraffe: patch seeds are spread evenly ON THE GIRAFFE
    (dart throwing, spacing ~patch) and sent to the sphere by the map; the pattern (nearest seeds, cream lines on
    the bisectors) is evaluated on the sphere, with the line width scaled by the map's local scale kf, so that
    every patch and line has the same size on the giraffe whatever the map does to area. Vf, Sf, kf: the fine
    (subdivided) vertices on the giraffe and on the sphere, and the local scale there."""
    rng = np.random.default_rng(seed)
    A = tri_area(V, F)
    m = int(12 * A.sum() / patch ** 2)
    t = rng.choice(len(F), m, p=A / A.sum())
    r1, r2 = rng.random(m), rng.random(m)
    u = 1 - np.sqrt(r1); v = np.sqrt(r1) * (1 - r2); w = np.sqrt(r1) * r2
    cand3 = u[:, None] * V[F[t, 0]] + v[:, None] * V[F[t, 1]] + w[:, None] * V[F[t, 2]]
    candS = u[:, None] * S[F[t, 0]] + v[:, None] * S[F[t, 1]] + w[:, None] * S[F[t, 2]]
    keep, tree, acc = [], None, []
    rmin = 0.8 * patch
    for i in range(m):                                          # dart throwing on the giraffe
        if tree is not None and tree.query_ball_point(cand3[i], rmin):
            continue
        if acc and np.min(np.linalg.norm(np.array(acc[-256:]) - cand3[i], axis=1)) < rmin:
            continue
        keep.append(i); acc.append(cand3[i])
        if len(acc) % 256 == 0:
            tree = cKDTree(np.array(acc))
    P = candS[keep] / np.linalg.norm(candS[keep], axis=1, keepdims=True)
    print(f"{len(P)} patches on the giraffe")
    d, i = cKDTree(P).query(Sf, k=2)
    edge = (d[:, 1] - d[:, 0]) < line * kf                      # line width on the giraffe = line
    browns = np.array([[166, 92, 40], [178, 104, 46], [152, 84, 38], [184, 112, 52], [160, 96, 44]])
    col = browns[rng.integers(0, len(browns), len(P))][i[:, 0]] * (1 + 0.06 * rng.standard_normal(len(P)))[i[:, 0], None]
    return np.where(edge[:, None], [238, 222, 190], np.clip(col, 0, 255)).astype(np.uint8)


def pullback_metric(V, S, F):
    """Per vertex, the giraffe's metric pulled back to the sphere, g = A^T A (3 x 3, acting on sphere tangent
    vectors), where per triangle A maps the sphere triangle's edges onto the giraffe triangle's edges (and the
    sphere normal to 0). Its eigen-ellipses are the map's Tissot ellipses, inverted."""
    e1s, e2s = S[F[:, 1]] - S[F[:, 0]], S[F[:, 2]] - S[F[:, 0]]
    n = np.cross(e1s, e2s)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    Bs = np.stack([e1s, e2s, n], 2)                              # columns: sphere edges and normal
    Bx = np.stack([V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]], np.zeros_like(n)], 2)
    A = Bx @ np.linalg.inv(Bs)                                   # giraffe vector = A @ sphere vector
    g = np.swapaxes(A, 1, 2) @ A
    w = tri_area(S, F)
    G = np.zeros((len(V), 3, 3))
    for k in range(3):
        np.add.at(G, F[:, k], w[:, None, None] * g)
    return G / np.bincount(F.ravel(), np.repeat(w, 3), len(V))[:, None, None]


def coat_metric(V, S, F, Sf, Gf, Vf_seedsurf=None, patch=0.11, line=0.012, seed=5, k=64):
    """Distortion-aware coat on the sphere: Voronoi cells and cream lines measured in the pulled-back metric g,
    so they are round and uniform ON THE GIRAFFE (cf. anisometric texture synthesis, Lefebvre & Hoppe 2006).
    Seeds are spread evenly on the giraffe (dart throwing) and carried to the sphere."""
    rng = np.random.default_rng(seed)
    A = tri_area(V, F)
    m = int(12 * A.sum() / patch ** 2)
    t = rng.choice(len(F), m, p=A / A.sum())
    r1, r2 = rng.random(m), rng.random(m)
    bw = np.c_[1 - np.sqrt(r1), np.sqrt(r1) * (1 - r2), np.sqrt(r1) * r2]
    cand3 = np.einsum("mi,mij->mj", bw, V[F[t]])
    acc, keep, tree = [], [], None
    for i in range(m):
        if tree is not None and tree.query_ball_point(cand3[i], 0.8 * patch):
            continue
        if acc and np.min(np.linalg.norm(np.array(acc[-256:]) - cand3[i], axis=1)) < 0.8 * patch:
            continue
        keep.append(i); acc.append(cand3[i])
        if len(acc) % 256 == 0:
            tree = cKDTree(np.array(acc))
    keep = np.array(keep)
    P = np.einsum("mi,mij->mj", bw[keep], S[F[t[keep]]])
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    G0 = pullback_metric(V, S, F)
    GP = np.einsum("mi,mijk->mjk", bw[keep], G0[F[t[keep]]])      # metric at the seeds
    print(f"{len(P)} patches; metric-aware Voronoi with {k} candidates per point")
    # candidate seeds: the nearest ones ON THE GIRAFFE (where the map is very anisotropic, the nearest seeds on
    # the sphere can miss the true neighbours); the ranking below is still done with the sphere metric
    _, cand = cKDTree(cand3[keep]).query(Vf_seedsurf, k=k) if Vf_seedsurf is not None else cKDTree(P).query(Sf, k=k)
    best = np.full((len(Sf), 2), np.inf)
    besti = np.zeros(len(Sf), int)
    for j in range(k):
        c = cand[:, j]
        if EXACT:                                              # d(s, p) = |Phi(s) - Phi(p)|: the metric, integrated
            d = np.linalg.norm(Vf_seedsurf - cand3[keep][c], axis=1)
        else:                                                  # first order: sqrt(ds^T g ds), g averaged at both ends
            dv = Sf - P[c]
            M = 0.5 * (Gf + GP[c])
            d = np.sqrt(np.maximum(np.einsum("ni,nij,nj->n", dv, M, dv), 0))
        first = d < best[:, 0]
        best[:, 1] = np.where(first, best[:, 0], np.minimum(best[:, 1], d))
        best[:, 0] = np.where(first, d, best[:, 0])
        besti = np.where(first, c, besti)
    edge = (best[:, 1] - best[:, 0]) < line
    browns = np.array([[166, 92, 40], [178, 104, 46], [152, 84, 38], [184, 112, 52], [160, 96, 44]])
    col = browns[rng.integers(0, len(browns), len(P))][besti] * (1 + 0.06 * rng.standard_normal(len(P)))[besti, None]
    return np.where(edge[:, None], [238, 222, 190], np.clip(col, 0, 255)).astype(np.uint8)


def spacing(P):
    """Patch spacing on the giraffe (reticulated giraffe, cf. photos): big on the torso, smaller up the neck,
    tiny on the face, small down the legs. P: rest-pose points (sim frame, height 2)."""
    return np.interp(P[:, 1], [0.30, 0.85, 1.00, 1.35, 1.60, 1.78, 1.90],
                     [0.035, 0.065, 0.10, 0.10, 0.075, 0.055, 0.03])


def warp(P, amp, seed=7):
    """A smooth 3D noise displacement (sum of random plane waves), amplitude ~amp (per point)."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(P)
    for _ in range(6):
        k = rng.standard_normal(3); k *= rng.uniform(12, 30) / np.linalg.norm(k)
        out += np.sin(P @ k + rng.uniform(0, 2 * np.pi))[:, None] * rng.standard_normal(3)
    return P + amp[:, None] * out / 6


def coat_real(V, S, F, Vf, seed=5):
    """The coat, closer to a reticulated giraffe: varying patch size (spacing field), thin near-white lines,
    one warm chestnut hue, irregular edges (noise warp). Still a function on the sphere carried by the map:
    distances are |Phi(s) - Phi(p)|, measured on the giraffe."""
    rng = np.random.default_rng(seed)
    A = tri_area(V, F)
    C = V[F].mean(1)
    dens = A / spacing(C) ** 2
    m = int(14 * dens.sum())
    t = rng.choice(len(F), m, p=dens / dens.sum())
    r1, r2 = rng.random(m), rng.random(m)
    bw = np.c_[1 - np.sqrt(r1), np.sqrt(r1) * (1 - r2), np.sqrt(r1) * r2]
    cand = np.einsum("mi,mij->mj", bw, V[F[t]])
    rc = spacing(cand)
    acc, accr = [], []
    tree, built = None, 0
    for i in range(m):                                           # variable-radius dart throwing
        rmin = 0.82 * rc[i]
        if tree is not None and tree.query_ball_point(cand[i], rmin):
            continue
        if len(acc) > built:
            recent = np.array(acc[built:])
            if np.min(np.linalg.norm(recent - cand[i], axis=1)) < rmin:
                continue
        acc.append(cand[i])
        if len(acc) - built >= 256:
            tree, built = cKDTree(np.array(acc)), len(acc)
    seeds = np.array(acc)
    r = spacing(Vf)
    Q = warp(Vf, 0.10 * r)                                       # irregular, notched edges
    d, i = cKDTree(seeds).query(Q, k=2)
    # toward the muzzle the network does not stop: it blurs and fades (cf. close-up photos). b: 0 at the
    # eyes, 1 at the muzzle tip; lines get soft, wider and tan, patches drift toward the muzzle tan
    from fig_giraffe import landmarks
    eyes, _, tip = landmarks(V)
    ez = eyes[:, 2].mean()
    b = np.clip((Vf[:, 2] - ez) / (tip[2] - ez), 0, 1) * (Vf[:, 1] > 1.74)
    w = 0.11 * r * (1 + 1.5 * b)                                 # line half-width grows
    soft = 0.012 * r + 0.6 * w * b                               # and its edge blurs
    line = np.clip((w - (d[:, 1] - d[:, 0])) / soft, 0, 1) * (1 - 0.75 * b ** 1.5)   # contrast fades
    base = np.array([186, 92, 34]) * (1 + 0.035 * rng.standard_normal(len(seeds)))[:, None]
    patch = base[i[:, 0]] * (1 - 0.55 * b[:, None]) + np.array([200, 132, 74]) * 0.55 * b[:, None]
    lcol = np.array([246, 241, 230]) * (1 - b[:, None]) + np.array([222, 176, 124]) * b[:, None]
    col = patch * (1 - line[:, None]) + lcol * line[:, None]
    print(f"{len(seeds)} patches (spacing field 0.03 .. 0.10)")
    return np.clip(col, 0, 255).astype(np.uint8)


def details_real(P, col, V0, N=None):
    """Head, ears, eye, mane, white lower legs, pale belly, hooves, tail tuft (rest-pose masks).
    N: rest-pose vertex normals (the belly is lightened only where the surface faces down)."""
    from fig_giraffe import landmarks
    eyes, nostrils, tip = landmarks(V0)
    out = col.astype(float)
    down = np.clip(-N[:, 1], 0, 1) if N is not None else np.zeros(len(P))

    def blend(a, rgb):
        a = np.clip(a, 0, 1)[:, None]
        out[:] = out * (1 - a) + np.asarray(rgb, float) * a

    y, z, ax = P[:, 1], P[:, 2], np.abs(P[:, 0])
    blend(0.35 * np.clip((y - 1.80) / 0.04, 0, 1) * (out[:, 0] < 240), [214, 132, 66])   # lighter face
    # lower legs: patches keep going but fade to white, gradually, from the knees/hocks down
    blend(np.clip((0.44 - y) / 0.26, 0, 1) ** 1.5 * (z > -0.5), [240, 236, 226])
    blend(0.5 * np.clip((down - 0.55) / 0.3, 0, 1) * (y > 0.7) * (y < 1.1), [232, 214, 186])   # pale underside
    blend((0.030 - y) / 0.008, [70, 64, 60])                                              # hooves, dark grey
    # mane: the back ridge of the neck (smallest z at each height, near the midline)
    neck = (V0[:, 1] > 1.30) & (V0[:, 1] < 1.95) & (V0[:, 2] > -0.25)
    hb = np.linspace(1.30, 1.95, 27)
    zmin = np.array([V0[neck & (np.abs(V0[:, 1] - h) < 0.03), 2].min() for h in hb])
    zr = np.interp(y, hb, zmin)
    mane = (y > 1.30) & (y < 1.93) & (z - zr < 0.05) & (ax < 0.06)      # wide enough to read from the side
    blend(mane * np.clip((0.05 - (z - zr)) / 0.02, 0, 1) * np.clip((0.06 - ax) / 0.02, 0, 1), [98, 40, 18])
    blend((ax > 0.072) * (y > 1.885) * (z > 0.33) * (z < 0.47), [246, 242, 234])         # ears: white
    blend((y - 1.955) / 0.02, [52, 38, 30])                                                # ossicone tufts
    blend((z < -0.46) * (0.66 - y) / 0.06, [24, 20, 18])                                   # tail tuft, black
    # (the snout's blurred network is in coat_real); a pale cheek below the eye
    for e in eyes:
        c = e + [0.0, -0.032, -0.012]
        blend(0.55 * np.clip((0.03 - np.linalg.norm(P - c, axis=1)) / 0.015, 0, 1), [240, 226, 200])
    for e in eyes:
        # an almond eye: longer along the head (z) than tall (y), in a soft patch of lighter fur, with a dark
        # upper lid / lashes and only a faint highlight (photos: no cartoon glint)
        dz, dy, dx = P[:, 2] - e[2], P[:, 1] - e[1], P[:, 0] - e[0]
        rho = np.sqrt((dz / 0.019) ** 2 + (dy / (0.0105 - 0.004 * np.clip(-dz / 0.019, -1, 1) ** 2)) ** 2
                      + (dx / 0.03) ** 2)
        near = np.sqrt(dz ** 2 + dy ** 2 + dx ** 2)
        blend(0.55 * np.clip((0.045 - near) / 0.02, 0, 1), [226, 178, 120])               # lighter fur around
        blend((1.35 - rho) / 0.12 * (dy > -0.002), [44, 26, 18])                          # upper lid and lashes
        blend((1.0 - rho) / 0.15, [52, 30, 20])                                            # the eye: dark brown
        blend((0.55 - rho) / 0.2, [30, 18, 12])                                            # darker center
        hl = np.sqrt(((dz + 0.006) / 0.004) ** 2 + ((dy - 0.004) / 0.0025) ** 2 + (dx / 0.03) ** 2)
        blend(0.45 * np.clip((1 - hl) / 0.4, 0, 1), [150, 130, 115])                     # faint, soft highlight
    # ---- muzzle (cf. a close-up photo): a rounded, lighter warm-tan block; slit nostrils with dark rims on its
    # top-front; a pale overhanging upper lip; a dark mouth line curving down at the corner; a shadowed lower lip
    head = (y > 1.74)
    mz = tip[2] - 0.075                                                   # where the muzzle block begins
    blend(0.45 * head * np.clip((z - mz) / 0.06, 0, 1), [204, 146, 92])                   # muzzle: warmer, softly
    hv = V0[V0[:, 1] > 1.74]
    for sgn in (1, -1):
        target = tip + [sgn * 0.018, 0.034, -0.026]                      # top-front of the muzzle, each side
        c = hv[np.argmin(np.linalg.norm(hv - target, axis=1))]           # anchored on the surface
        a = np.array([sgn * 0.45, 0.25, -0.86]); a /= np.linalg.norm(a)  # slit: runs back, outwards and up
        nrm = c - hv.mean(0); nrm /= np.linalg.norm(nrm)                  # rough outward direction there
        b = np.cross(nrm, a); b /= np.linalg.norm(b)                      # across the slit, on the surface
        q = P - c
        slit = np.sqrt((q @ a / 0.018) ** 2 + (q @ b / 0.0045) ** 2 + (q @ nrm / 0.015) ** 2)
        blend(0.6 * np.clip((2.4 - slit) / 0.8, 0, 1) * head, [150, 98, 62])             # darker skin around
        blend(np.clip((1.45 - slit) / 0.3, 0, 1) * head, [96, 60, 40])                    # the rim
        blend(np.clip((1.0 - slit) / 0.25, 0, 1) * head, [26, 18, 16])                    # the opening
    lip_y = tip[1] - 0.026 + 0.10 * np.clip(tip[2] - z, 0, None) - 2.5 * np.clip(tip[2] - 0.06 - z, 0, None) ** 1.0
    mouth = (z > tip[2] - 0.085) & (z < tip[2] + 0.01) & (ax < 0.05) & head
    up = mouth * (y > lip_y) * np.clip((0.02 - (y - lip_y)) / 0.012, 0, 1)
    blend(0.55 * up, [224, 178, 128])                                                      # pale upper lip
    low = mouth * (y < lip_y) * np.clip((0.022 - (lip_y - y)) / 0.012, 0, 1)
    blend(0.6 * low, [134, 82, 48])                                                        # lower lip, in shadow
    blend(mouth * np.clip((0.0065 - np.abs(y - lip_y)) / 0.0025, 0, 1), [44, 28, 20])    # the mouth line
    return out.astype(np.uint8)


MODE = sys.argv[2] if len(sys.argv) > 2 else "real"
EXACT = MODE == "exact"


def main():
    from fig_giraffe import detail_color, landmarks
    G = np.load(os.path.join(DATA, "giraffe_sphere.npz"))
    F0 = G["F"]
    S0 = G[f"nu_{NU}"]
    S0 = S0 / np.linalg.norm(S0, axis=1, keepdims=True)
    Vraw, _ = read_off(SRC)
    V0 = sim_frame(Vraw)
    # the map's local scale per vertex: sqrt(sphere area / giraffe area), averaged over the incident faces
    ratio = np.sqrt(tri_area(S0, F0) / tri_area(V0, F0))
    k0 = np.bincount(F0.ravel(), np.repeat(ratio, 3), len(V0)) / np.bincount(F0.ravel(), minlength=len(V0))
    G0 = pullback_metric(V0, S0, F0).reshape(len(V0), 9)
    F, Vs, S, k, Gf = F0, V0, S0, k0[:, None], G0
    for _ in range(3):
        F, (Vs, S, k, Gf) = subdivide(F, Vs, S, k, Gf)
    S = S / np.linalg.norm(S, axis=1, keepdims=True)
    if MODE == "real":
        Nf = pv.PolyData(Vs, np.hstack([np.full((len(F), 1), 3), F]).ravel()).compute_normals(
            cell_normals=False, auto_orient_normals=True)["Normals"]
        col = details_real(Vs, coat_real(V0, S0, F0, Vs), V0, Nf)
    else:
        if MODE in ("metric", "exact"):
            col = coat_metric(V0, S0, F0, S, Gf.reshape(-1, 3, 3), Vf_seedsurf=Vs, k=16)
        else:
            col = coat_adaptive(V0, S0, F0, Vs, S, k[:, 0])
        col = detail_color(Vs, col, landmarks(V0))                # hooves, eyes, ears, muzzle, tail tuft
    D = np.load(os.path.join(DATA, "giraffe_nh.npz"))
    X0, T, fr = D["X0"], D["T"].astype(int), D["frames"]
    tet, bary = embed(Vs, X0, T)
    pose = lambda X: np.einsum("ij,ijk->ik", bary, X[T[tet]])
    faces = np.hstack([np.full((len(F), 1), 3), F]).ravel()

    # sphere | giraffe, side by side (for the slide)
    p = pv.Plotter(off_screen=True, window_size=(1600, 760), shape=(1, 2), border=False)
    p.set_background("white")
    p.subplot(0, 0)
    ms = pv.PolyData(S, faces); ms.point_data["rgb"] = col
    p.add_mesh(ms, scalars="rgb", rgb=True, smooth_shading=True, ambient=0.35, diffuse=0.75)
    p.camera_position = [(3.2, 1.2, 2.2), (0, 0, 0), (0, 1, 0)]
    p.subplot(0, 1)
    mg = pv.PolyData(pose(fr[0]), faces); mg.point_data["rgb"] = col
    p.add_mesh(mg, scalars="rgb", rgb=True, smooth_shading=True, ambient=0.35, diffuse=0.75)
    p.camera_position = [(4.6, 1.5, 2.2), (0, 0.95, 0.35), (0, 1, 0)]
    save_png(np.asarray(p.screenshot(return_img=True)), "giraffe_sphere_pair.png")
    p.close()

    # side view like the reference photo (facing left), next to it
    p = pv.Plotter(off_screen=True, window_size=(1000, 1000))
    p.set_background("white")
    mg = pv.PolyData(pose(fr[0]), faces); mg.point_data["rgb"] = col
    p.add_mesh(mg, scalars="rgb", rgb=True, smooth_shading=True, ambient=0.4, diffuse=0.7)
    p.camera_position = [(5.5, 1.05, 0.08), (0, 1.0, 0.08), (0, 1, 0)]
    side = np.asarray(p.screenshot(return_img=True))
    p.close()
    ref = os.path.join(FIG, "giraffe_real.jpg")
    if os.path.exists(ref):
        photo = Image.open(ref).convert("RGB")
        photo = np.asarray(photo.resize((int(photo.width * 1000 / photo.height), 1000)))
        save_png(np.concatenate([side, photo], 1), "giraffe_vs_photo.png")
        # head close-up, next to the photo's head
        p = pv.Plotter(off_screen=True, window_size=(700, 560))
        p.set_background("white")
        mh = pv.PolyData(pose(fr[0]), faces); mh.point_data["rgb"] = col
        p.add_mesh(mh, scalars="rgb", rgb=True, smooth_shading=True, ambient=0.4, diffuse=0.7)
        p.camera_position = [(1.0, 1.9, 0.52), (0.0, 1.88, 0.52), (0, 1, 0)]
        head = np.asarray(p.screenshot(return_img=True))
        p.close()
        ph = Image.open(ref).convert("RGB").crop((560, 110, 830, 326)).resize((700, 560))
        save_png(np.concatenate([head, np.asarray(ph)], 1), "giraffe_head_vs_photo.png")
    ref2 = os.path.join(FIG, "giraffe_muzzle.jpg")
    if os.path.exists(ref2):                                   # muzzle close-up, three-quarter view, mirrored
        p = pv.Plotter(off_screen=True, window_size=(800, 422))
        p.set_background("white")
        mh = pv.PolyData(pose(fr[0]), faces); mh.point_data["rgb"] = col
        p.add_mesh(mh, scalars="rgb", rgb=True, smooth_shading=True, ambient=0.4, diffuse=0.7)
        p.camera_position = [(0.42, 1.93, 0.86), (0.0, 1.86, 0.58), (0, 1, 0)]
        mz = np.asarray(p.screenshot(return_img=True))[:, ::-1]
        p.close()
        ph2 = np.asarray(Image.open(ref2).convert("RGB").resize((800, 422)))
        save_png(np.concatenate([mz, ph2], 1), "giraffe_muzzle_vs_photo.png")

    p = pv.Plotter(off_screen=True, window_size=(900, 760))
    p.set_background("white")
    p.add_mesh(pv.Plane(center=(0, -0.002, 0.4), direction=(0, 1, 0), i_size=3, j_size=3.5), color="#e2e1dc",
               lighting=False)
    m = pv.PolyData(pose(fr[0]), faces)
    m.point_data["rgb"] = col
    p.add_mesh(m, scalars="rgb", rgb=True, smooth_shading=True, ambient=0.35, diffuse=0.75, specular=0.05)
    p.camera_position = [(4.6, 1.5, 2.2), (0, 0.95, 0.35), (0, 1, 0)]
    W = Writer(os.path.join(VID, "vid_giraffe_sphere"), fps=30, poster=len(fr) - 1)
    for k in list(range(len(fr))) + [len(fr) - 1] * 20 + list(range(len(fr) - 1, -1, -1)) + [0] * 10:
        m.points = pose(fr[k])
        m.Modified()
        p.render()
        W.add(np.asarray(p.screenshot(return_img=True)))
    W.close()
    p.close()


if __name__ == "__main__":
    main()
