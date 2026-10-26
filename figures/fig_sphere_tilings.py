"""Sphere tilings made of near-equal, near-regular cells, as equirectangular textures (cells filled with
the 3 talk colors + paper by greedy graph coloring, edges in dark gray), rendered on the sphere.
  hex:      Goldberg polyhedron = Voronoi cells of the geodesic-sphere vertices (hexagons + 12 pentagons)
  tri:      geodesic sphere (icosahedron, each face cut into n^2 triangles, centrally projected)
  cvt:      spherical centroidal Voronoi tessellation (Fibonacci points + Lloyd)
  healpix:  HEALPix equal-area cells (ring scheme)
  mercator: the Mercator checker, for comparison
    python figures/fig_sphere_tilings.py [names...]"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from distortion import polyhedral as ph  # noqa: E402
from distortion.render import render_textured, subdivide  # noqa: E402
from distortion.textures import COLORS, PAPER  # noqa: E402

OUT = os.path.join(HERE, "..", "..", "assets", "fig")
H = 2048
EDGE = np.array([70, 70, 75], float)
PALETTE = np.vstack([COLORS, PAPER, [200, 200, 200], [255, 0, 255], [255, 0, 255], [255, 0, 255], [255, 0, 255]])
ICO_V, ICO_F = ph.icosahedron()


def directions(h=H):
    w = 2 * h
    lon = (np.arange(w) + 0.5) / w * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi
    LON, LAT = np.meshgrid(lon, lat)
    return np.stack([np.cos(LAT) * np.cos(LON), np.cos(LAT) * np.sin(LON), np.sin(LAT)], -1).reshape(-1, 3)


# ---------------------------------------------------------------- cell-id functions: unit vectors -> int ids
def geodesic_points(n):
    """Vertices of the frequency-n geodesic sphere (central projection of each face's barycentric grid)."""
    P = []
    for a, b, c in ICO_F:
        A, B, C = ICO_V[a], ICO_V[b], ICO_V[c]
        for i in range(n + 1):
            for j in range(n + 1 - i):
                P.append((i * A + j * B + (n - i - j) * C) / n)
    P = np.array(P); P /= np.linalg.norm(P, axis=1, keepdims=True)
    _, keep = np.unique(np.round(P, 6), axis=0, return_index=True)
    return P[np.sort(keep)]


def voronoi_ids(P):
    tree = cKDTree(P)
    return lambda E: tree.query(E)[1]


def tri_ids(n):
    N = np.cross(ICO_V[ICO_F[:, 1]] - ICO_V[ICO_F[:, 0]], ICO_V[ICO_F[:, 2]] - ICO_V[ICO_F[:, 0]])
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    N *= np.sign((N * ICO_V[ICO_F].mean(1)).sum(1))[:, None]
    Minv = np.linalg.inv(np.transpose(ICO_V[ICO_F], (0, 2, 1)))   # columns A, B, C -> barycentric (unnormalized)

    def f(E):
        k = np.argmax(E @ N.T, 1)
        bc = np.einsum("nij,nj->ni", Minv[k], E)
        bc /= bc.sum(1, keepdims=True)
        g = np.clip(np.floor(bc * n), 0, n - 1).astype(int)
        up = (g.sum(1) == n - 1).astype(int)
        return ((k * (n + 1) + g[:, 0]) * (n + 1) + g[:, 1]) * 2 + up
    return f


def cvt_points(n=1450, iters=40, samples=400000, seed=0):
    k = np.arange(n) + 0.5
    z = 1 - 2 * k / n
    phi = np.pi * (1 + 5 ** 0.5) * k
    P = np.c_[np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi), z]
    rng = np.random.default_rng(seed)
    P += 0.3 * np.sqrt(4 * np.pi / n) * rng.normal(size=P.shape)    # start irregular, let Lloyd regularize
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    for _ in range(iters):
        X = rng.normal(size=(samples, 3)); X /= np.linalg.norm(X, axis=1, keepdims=True)
        idx = cKDTree(P).query(X)[1]
        S = np.zeros_like(P); np.add.at(S, idx, X)
        cnt = np.bincount(idx, minlength=n)
        P = np.where(cnt[:, None] > 0, S, P)
        P /= np.linalg.norm(P, axis=1, keepdims=True)
    return P


def healpix_ids(nside):
    def f(E):
        z = np.clip(E[:, 2], -1, 1)
        phi = np.mod(np.arctan2(E[:, 1], E[:, 0]), 2 * np.pi)
        za = np.abs(z)
        tt = np.mod(phi / (np.pi / 2), 4.0)
        pix = np.empty(len(E), np.int64)
        eq = za <= 2 / 3
        t1 = nside * (0.5 + tt[eq]); t2 = nside * z[eq] * 0.75
        jp = (t1 - t2).astype(np.int64); jm = (t1 + t2).astype(np.int64)
        ir = nside + 1 + jp - jm
        kshift = 1 - (ir & 1)
        ip = np.mod((jp + jm - nside + kshift + 1) // 2, 4 * nside)
        pix[eq] = 2 * nside * (nside - 1) + (ir - 1) * 4 * nside + ip
        pl = ~eq
        tp = tt[pl] - np.floor(tt[pl])
        tmp = nside * np.sqrt(3 * (1 - za[pl]))
        jp = (tp * tmp).astype(np.int64); jm = ((1 - tp) * tmp).astype(np.int64)
        ir = jp + jm + 1
        ip = np.mod((tt[pl] * ir).astype(np.int64), 4 * ir)
        pix[pl] = np.where(z[pl] > 0, 2 * ir * (ir - 1) + ip, 12 * nside * nside - 2 * ir * (ir + 1) + ip)
        return pix
    return f


def mercator_ids(n=64):
    def f(E):
        lat = np.arcsin(np.clip(E[:, 2], -1, 1)); lon = np.arctan2(E[:, 1], E[:, 0])
        psi = np.log(np.tan(np.pi / 4 + np.clip(lat, -1.5607, 1.5607) / 2))
        d = 2 * np.pi / n
        return np.floor((lon + np.pi) / d).astype(np.int64) * 100000 + np.floor(psi / d).astype(np.int64)
    return f


# ---------------------------------------------------------------- texture: fill + edges
def tangent_frame(E):
    a = np.where(np.abs(E[:, 2:3]) < 0.9, [[0, 0, 1.0]], [[1.0, 0, 0]])
    t1 = np.cross(E, a); t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    return t1, np.cross(E, t1)


def color_graph(pairs, ids, tries=300):
    """Randomized DSatur coloring with the 3 colors + paper; keeps the attempt that uses the fewest colors."""
    nbr = {i: set() for i in ids}
    for a, b in pairs:
        nbr[a].add(b); nbr[b].add(a)
    rng = np.random.default_rng(0)
    best = None
    for _ in range(tries):
        tie = {i: rng.random() for i in ids}
        col, sat = {}, {i: set() for i in ids}
        left = set(ids)
        while left:
            i = max(left, key=lambda v: (len(sat[v]), len(nbr[v]), tie[v]))
            left.remove(i)
            c = next(c for c in range(9) if c not in sat[i])
            col[i] = c
            for u in nbr[i]:
                sat[u].add(c)
        k = max(col.values()) + 1
        if best is None or k < best[0]:
            best = (k, col)
        if k <= 4:
            break
    return best[1]


def texture(idf, width=0.0035, chunk=1 << 21, maxdeg=None):
    E = directions()
    ids = np.empty(len(E), np.int64); edge = np.zeros(len(E), bool)
    from collections import Counter
    pairs = Counter()
    for s in range(0, len(E), chunk):
        e = E[s:s + chunk]
        c = idf(e); ids[s:s + chunk] = c
        t1, t2 = tangent_frame(e)
        for t in (t1, t2, -t1, -t2, (t1 + t2) / 1.414, (t1 - t2) / 1.414):
            q = e + width * t; q /= np.linalg.norm(q, axis=1, keepdims=True)
            o = idf(q)
            m = o != c
            edge[s:s + chunk] |= m
            pairs.update(tuple(sorted(x)) for x in zip(c[m].tolist(), o[m].tolist()))
    uniq, inv = np.unique(ids, return_inverse=True)
    med = np.median(list(pairs.values()))
    keep = {k: v for k, v in pairs.items() if v > 0.4 * med}
    if maxdeg:                                   # triangles: each cell has exactly maxdeg edge-neighbors
        rank = {}
        for (a, b), v in keep.items():
            rank.setdefault(a, []).append((v, (a, b))); rank.setdefault(b, []).append((v, (a, b)))
        top = {i: {k for _, k in sorted(r, reverse=True)[:maxdeg]} for i, r in rank.items()}
        keep = {k: v for k, v in keep.items() if k in top[k[0]] and k in top[k[1]]}
    col = color_graph(set(keep), uniq.tolist())   # edges, not corners
    cidx = np.array([col[u] for u in uniq])[inv]
    img = PALETTE[cidx]
    img[edge] = EDGE
    print(f"  {len(uniq)} cells, colors used {sorted(set(col.values()))}")
    return img.reshape(H, 2 * H, 3).astype(np.uint8), len(uniq)


def cell_stats(P):
    """Area spread of the Voronoi cells of P (Monte Carlo)."""
    X = np.random.default_rng(1).normal(size=(2_000_000, 3)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    a = np.bincount(cKDTree(P).query(X)[1], minlength=len(P)).astype(float)
    return a.max() / a.min()


TILINGS = {
    "hex": ("Hexagons + 12 pentagons", lambda: voronoi_ids(geodesic_points(12))),
    "tri": ("Geodesic triangles", lambda: tri_ids(8)),
    "cvt": ("Voronoi of spread points", lambda: voronoi_ids(cvt_points())),
    "healpix": ("HEALPix (equal area)", lambda: healpix_ids(11)),
    "mercator": ("Mercator squares", lambda: mercator_ids()),
}


def main(names):
    V, F = ph.icosahedron()
    _, S, Fs = subdivide(V, V, F, 6)
    # look at an icosahedron vertex (a pentagon / 5-valent point) slightly off center, and a HEALPix corner
    look = ICO_V[0] * 0.8 + ICO_V[5] * 0.2
    look /= np.linalg.norm(look)
    for name in names:
        title, make = TILINGS[name]
        print(name)
        tex, n = texture(make(), maxdeg=3 if name == "tri" else None)
        view = look if name != "healpix" else np.array([0.55, 0.25, 0.80]) / np.linalg.norm([0.55, 0.25, 0.8])
        up = np.array([0, 0, 1.0]) if abs(view[2]) < 0.95 else np.array([1.0, 0, 0])
        img = render_textured(S, S, Fs, tex, camera=[tuple(5.2 * view), (0, 0, 0), tuple(up)],
                              size=(900, 900), shape=True)
        plt.imsave(os.path.join(OUT, f"aux_tiling_{name}.png"), img)
        np.save(os.path.join(HERE, "..", "scratch", f"tiling_{name}.npy"), tex[::2, ::2])


if __name__ == "__main__":
    main(sys.argv[1:] or list(TILINGS))


# ---------------------------------------------------------------- hex colorings: balanced, and cube-like regions
def geodesic_points_labeled(n):
    """Geodesic vertices with (face, i, j) of the first face that owns them."""
    P, lab = [], []
    for f, (a, b, c) in enumerate(ICO_F):
        A, B, C = ICO_V[a], ICO_V[b], ICO_V[c]
        for i in range(n + 1):
            for j in range(n + 1 - i):
                P.append((i * A + j * B + (n - i - j) * C) / n); lab.append((f, i, j))
    P = np.array(P); P /= np.linalg.norm(P, axis=1, keepdims=True)
    _, keep = np.unique(np.round(P, 6), axis=0, return_index=True)
    keep = np.sort(keep)
    return P[keep], np.array(lab)[keep]


def face_colors():
    """3-coloring of the 20 icosahedron faces (adjacent faces differ; the dodecahedron graph is 3-colorable)."""
    E = {}
    for f, t in enumerate(ICO_F):
        for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            E.setdefault(tuple(sorted(e)), []).append(f)
    nbr = [set() for _ in ICO_F]
    for fs in E.values():
        nbr[fs[0]].add(fs[1]); nbr[fs[1]].add(fs[0])
    col = [-1] * len(ICO_F)

    def bt(k):
        if k == len(ICO_F):
            return True
        for c in range(3):
            if all(col[u] != c for u in nbr[k]):
                col[k] = c
                if bt(k + 1):
                    return True
        col[k] = -1
        return False
    assert bt(0)
    return np.array(col)


def edge_pairs(P):
    """Adjacent Voronoi cells = edges of the convex hull of the points."""
    from scipy.spatial import ConvexHull
    T = ConvexHull(P).simplices
    return {tuple(sorted(e)) for t in T for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}


def balanced_coloring(pairs, n, k=4, iters=200000, seed=0):
    """Proper k-coloring with near-equal class sizes: DSatur start, then single-cell moves and
    two-cell swaps (Kempe-free) from the largest class to the smallest."""
    col = color_graph(pairs, list(range(n)))
    col = np.array([col[i] for i in range(n)])
    nbr = [[] for _ in range(n)]
    for a, b in pairs:
        nbr[a].append(b); nbr[b].append(a)
    rng = np.random.default_rng(seed)
    for _ in range(iters):
        cnt = np.bincount(col, minlength=k)
        if cnt.max() - cnt.min() <= 1:
            break
        big, small = cnt.argmax(), cnt.argmin()
        i = rng.choice(np.flatnonzero(col == big))
        free = [c for c in range(k) if all(col[u] != c for u in nbr[i])]
        if small in free:
            col[i] = small
        elif free and rng.random() < 0.3:
            col[i] = rng.choice(free + [big])             # random walk inside the proper colorings
    return col


def texture_from_colors(P, img_colors, width=0.0035, chunk=1 << 21):
    tree = cKDTree(P)
    E = directions()
    out = np.empty((len(E), 3))
    for s in range(0, len(E), chunk):
        e = E[s:s + chunk]
        c = tree.query(e)[1]
        edge = np.zeros(len(e), bool)
        t1, t2 = tangent_frame(e)
        for t in (t1, t2, -t1, -t2):
            q = e + width * t; q /= np.linalg.norm(q, axis=1, keepdims=True)
            edge |= tree.query(q)[1] != c
        rgb = img_colors[c]
        rgb[edge] = EDGE
        out[s:s + chunk] = rgb
    return out.reshape(H, 2 * H, 3).astype(np.uint8)


def hex_variants(n=12):
    P, lab = geodesic_points_labeled(n)
    pairs = edge_pairs(P)
    # 1. balanced 4-coloring
    col = balanced_coloring(pairs, len(P))
    print("balanced counts", np.bincount(col))
    yield "hex_balanced", texture_from_colors(P, PALETTE[col].astype(float))
    # 2. cube-like: region hue per icosahedron face; honeycomb classes dark / light / paper
    fc = face_colors()
    cls = (lab[:, 1] - lab[:, 2]) % 3
    hue = COLORS[fc[lab[:, 0]]]
    light = hue + 0.45 * (PAPER - hue)
    rgb = np.where(cls[:, None] == 0, hue, np.where(cls[:, None] == 1, light, PAPER))
    same_paper = sum(1 for a, b in pairs if cls[a] == 2 and cls[b] == 2)
    print("cube-like: adjacent white pairs", same_paper)
    yield "hex_regions", texture_from_colors(P, rgb)
    # 3. cube-like with two classes: hue cells + paper cells (paper = one honeycomb class, 1/3)
    rgb2 = np.where(cls[:, None] == 2, PAPER, hue)
    yield "hex_regions2", texture_from_colors(P, rgb2)


def render_variants():
    V, F = ph.icosahedron()
    _, S, Fs = subdivide(V, V, F, 6)
    look = ICO_V[0] * 0.8 + ICO_V[5] * 0.2
    look /= np.linalg.norm(look)
    for name, tex in hex_variants():
        img = render_textured(S, S, Fs, tex, camera=[tuple(5.2 * look), (0, 0, 0), (0, 0, 1)], size=(900, 900), shape=True)
        plt.imsave(os.path.join(OUT, f"aux_tiling_{name}.png"), img)
        np.save(os.path.join(HERE, "..", "scratch", f"tiling_{name}.npy"), tex[::2, ::2])
        print("wrote", name)


def global_three_classes(pairs, n, steps=3_000_000, seed=0):
    """3-coloring of the hexagon cells with as few same-class neighbors as possible (simulated annealing).
    A perfect one cannot exist (each pentagon's 5 neighbors form an odd ring), so the leftovers sit at pentagons."""
    nbr = [[] for _ in range(n)]
    for a, b in pairs:
        nbr[a].append(b); nbr[b].append(a)
    rng = np.random.default_rng(seed)
    col = rng.integers(0, 3, n)
    I = rng.integers(0, n, steps); C = rng.integers(0, 3, steps); U = rng.random(steps)
    T = np.geomspace(2.0, 0.02, steps)
    for s in range(steps):
        i, c = I[s], C[s]
        if c == col[i]:
            continue
        d = sum(col[u] == c for u in nbr[i]) - sum(col[u] == col[i] for u in nbr[i])
        if d <= 0 or U[s] < np.exp(-d / T[s]):
            col[i] = c
    bad = sum(col[a] == col[b] for a, b in pairs)
    return col, bad


def hex_regions_global(n=12):
    P, lab = geodesic_points_labeled(n)
    pairs = edge_pairs(P)
    cls, bad = global_three_classes(pairs, len(P))
    print("same-class neighbor pairs:", bad, " class sizes", np.bincount(cls))
    # white = class with fewest internal conflicts; break remaining white-white contacts by turning one to hue
    wc = min(range(3), key=lambda c: sum(cls[a] == c and cls[b] == c for a, b in pairs))
    white = cls == wc
    for a, b in pairs:
        if white[a] and white[b]:
            white[b] = False
    fc = face_colors()
    hue = COLORS[fc[lab[:, 0]]]
    others = [c for c in range(3) if c != wc]
    light = hue + 0.45 * (PAPER - hue)
    rgb1 = np.where(white[:, None], PAPER, np.where((cls == others[1])[:, None], light, hue))
    rgb2 = np.where(white[:, None], PAPER, hue)
    return P, rgb1, rgb2


def render_regions_global():
    V, F = ph.icosahedron()
    _, S, Fs = subdivide(V, V, F, 6)
    look = ICO_V[0] * 0.8 + ICO_V[5] * 0.2
    look /= np.linalg.norm(look)
    P, rgb1, rgb2 = hex_regions_global()
    for name, rgb in (("hex_regions", rgb1), ("hex_regions2", rgb2)):
        tex = texture_from_colors(P, rgb)
        img = render_textured(S, S, Fs, tex, camera=[tuple(5.2 * look), (0, 0, 0), (0, 0, 1)], size=(900, 900), shape=True)
        plt.imsave(os.path.join(OUT, f"aux_tiling_{name}.png"), img)
        np.save(os.path.join(HERE, "..", "scratch", f"tiling_{name}.npy"), tex[::2, ::2])
