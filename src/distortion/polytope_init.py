"""Provably flip-free initial spherical map via a polytope layout (thesis Ch. 3, §3.4.1).

1. Cut the closed genus-0 mesh into triangular patches with the face structure of a
   polytope inscribed in the sphere (tetrahedron: 4 patches, octahedron: 8):
   pick corner vertices far apart, connect them by vertex-disjoint shortest paths.
2. Map each patch onto its polytope face with Tutte's embedding (boundary paths
   placed on the face's edges by arc length, mean-value weights inside).
3. Centrally project the polytope onto the sphere.  Central projection preserves
   orientation, so a flip-free Tutte layout gives det(s0, s1, s2) > 0 for every triangle.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components, dijkstra


# ---------------------------------------------------------------- graph helpers
def edge_graph(V, F):
    E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1), axis=0)
    w = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    n = len(V)
    return sp.coo_matrix((np.r_[w, w], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])),
                         shape=(n, n)).tocsr()


def shortest_path(G, a, b, allowed):
    """Shortest path a -> b using only vertices in `allowed` (a boolean mask); None if blocked."""
    idx = np.nonzero(allowed)[0]
    pos = -np.ones(G.shape[0], int)
    pos[idx] = np.arange(len(idx))
    sub = G[idx][:, idx]
    dist, pred = dijkstra(sub, indices=pos[a], return_predecessors=True)
    if not np.isfinite(dist[pos[b]]):
        return None
    path, v = [], pos[b]
    while v != pos[a]:
        path.append(idx[v])
        v = pred[v]
    path.append(a)
    return path[::-1]


def farthest_points(G, k, start=0, candidates=None, first=None):
    """Greedy farthest-point sampling by graph geodesic distance (from `first` if given)."""
    mask = np.ones(G.shape[0], bool) if candidates is None else candidates
    if first is None:
        d = dijkstra(G, indices=start)
        first = int(np.argmax(np.where(mask, d, -1)))
    pts = [first]
    dmin = dijkstra(G, indices=pts[0])
    while len(pts) < k:
        pts.append(int(np.argmax(np.where(mask, dmin, -1))))
        dmin = np.minimum(dmin, dijkstra(G, indices=pts[-1]))
    return pts


def face_components(F, cut_edges):
    """Connected components of faces when the mesh is cut along `cut_edges` (set of sorted pairs)."""
    T = len(F)
    he = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    key = np.sort(he, 1)
    fid = np.tile(np.arange(T), 3)
    order = np.lexsort((key[:, 1], key[:, 0]))
    k, f = key[order], fid[order]
    same = np.all(k[1:] == k[:-1], axis=1)
    pairs = np.c_[f[:-1][same], f[1:][same]]
    keep = np.array([tuple(e) not in cut_edges for e in k[:-1][same]], bool)
    pairs = pairs[keep]
    A = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(T, T))
    return connected_components(A, directed=False)[1]


def path_edges(path):
    return {tuple(sorted(e)) for e in zip(path[:-1], path[1:])}


# ---------------------------------------------------------------- layout: cycle + apexes
def _buffer(G, mask):
    """mask plus its 1-ring."""
    return mask | (G @ mask.astype(float) > 0)


def _cycle(G, corners, forbidden):
    """Paths corners[i] -> corners[i+1] (cyclically), kept one vertex apart from each other
    (except at shared corners) so the cycle never pinches off pockets of either side."""
    used = forbidden.copy()
    paths = []
    k = len(corners)
    for i in range(k):
        a, b = corners[i], corners[(i + 1) % k]
        near_ends = np.zeros(G.shape[0], bool)
        near_ends[[a, b]] = True
        allowed = ~_buffer(G, used) | _buffer(G, near_ends)
        allowed &= ~used
        allowed[[c for c in corners if c not in (a, b)]] = False
        allowed[[a, b]] = True
        p = shortest_path(G, a, b, allowed)
        if p is None:
            return None
        paths.append(p)
        used[p] = True
    return paths


def _ring(G, verts, k=1):
    m = np.zeros(G.shape[0], bool)
    m[verts] = True
    for _ in range(k):
        m = _buffer(G, m)
    return m


def _spokes(G, apex, targets, region):
    """Paths apex -> each target inside `region`, vertex-disjoint except at the apex and kept
    one vertex apart from each other, so no spoke can wall off a later target."""
    used = np.zeros(G.shape[0], bool)
    paths = []
    apex_ring = _ring(G, [apex])
    for m, t in enumerate(targets):
        allowed = region & ~used & (~_buffer(G, used) | apex_ring | _ring(G, [t]))
        if m + 1 < len(targets):   # keep later targets reachable: reserve their 2-rings
            allowed &= ~_ring(G, targets[m + 1:], 2)
        allowed[[apex, t]] = True
        p = shortest_path(G, apex, t, allowed)
        if p is None:
            return None
        paths.append(p)
        used[p[1:]] = True
    return paths


def _side_masks(F, n, cycle_paths):
    """Vertex masks of the two sides of a closed cycle (cycle vertices excluded)."""
    cut = set().union(*[path_edges(p) for p in cycle_paths])
    comp = face_components(F, cut)
    if comp.max() != 1:
        return None
    on_cycle = np.zeros(n, bool)
    for p in cycle_paths:
        on_cycle[p] = True
    sides = []
    for c in (0, 1):
        m = np.zeros(n, bool)
        m[F[comp == c].ravel()] = True
        sides.append(m & ~on_cycle)
    return sides


POLYTOPES = {
    # vertices on the unit sphere; the first `k` vertices form the separating cycle
    "tetrahedron": dict(
        X=np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3), cycle=3, apexes=1),
    "octahedron": dict(
        X=np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float),
        cycle=4, apexes=2),
}


def polytope_layout(V, F, polytope="tetrahedron", seed=0, tries=20):
    """Corner vertices and boundary paths realizing the polytope's edge graph on the mesh.
    Returns (corners, paths) where paths is a list of (i, j, vertex_list) in polytope indices."""
    n = len(V)
    G = edge_graph(V, F)
    spec = POLYTOPES[polytope]
    k, n_apex = spec["cycle"], spec["apexes"]
    rng = np.random.default_rng(seed)
    for attempt in range(tries):
        start = int(rng.integers(n))
        if polytope == "tetrahedron":
            cyc = farthest_points(G, 3, start)
            apexes = farthest_points(G, 1, start, candidates=~np.isin(np.arange(n), cyc))
            apexes = [a for a in apexes]
            orders = [cyc]
        else:
            N, S = farthest_points(G, 2, start)
            dN, dS = dijkstra(G, indices=N), dijkstra(G, indices=S)
            band = np.abs(dN - dS) < 0.15 * (dN + dS).mean()
            first = int(rng.choice(np.nonzero(band)[0])) if attempt else None
            eq = farthest_points(G, 4, start, candidates=band, first=first)
            apexes = [N, S]
            a, b, c, d = eq
            orders = [[a, b, c, d], [a, b, d, c], [a, c, b, d]]
        for cyc in orders:
            forbidden = np.zeros(n, bool)
            forbidden[apexes] = True
            cpaths = _cycle(G, cyc, forbidden)
            if cpaths is None:
                continue
            sides = _side_masks(F, n, cpaths)
            if sides is None:
                continue
            in_side = [[s[a] for s in sides] for a in apexes]
            if n_apex == 2 and not (in_side[0][0] != in_side[1][0]):
                continue      # the two apexes must lie on opposite sides of the cycle
            spokes = []
            ok = True
            for a, flags in zip(apexes, in_side):
                region = sides[0] if flags[0] else sides[1]
                sp_ = _spokes(G, a, cyc, region)
                if sp_ is None:
                    ok = False
                    break
                spokes.append(sp_)
            if not ok:
                continue
            corners = list(cyc) + list(apexes)
            paths = [(i, (i + 1) % k, p) for i, p in enumerate(cpaths)]
            for ai, sp_ in enumerate(spokes):
                paths += [(k + ai, i, p) for i, p in enumerate(sp_)]
            return corners, paths
    raise RuntimeError("could not realize the polytope layout; try another seed")


# ---------------------------------------------------------------- Tutte per patch + projection
def mean_value_weights(V, F):
    n = len(V)
    rows, cols, vals = [], [], []
    for k in range(3):
        o, i, j = F[:, k], F[:, (k + 1) % 3], F[:, (k + 2) % 3]
        a, b = V[i] - V[o], V[j] - V[o]
        la, lb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        t = np.tan(0.5 * np.arccos(np.clip(np.einsum("ij,ij->i", a, b) / (la * lb), -1, 1)))
        rows += [o, o]
        cols += [i, j]
        vals += [t / la, t / lb]
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))


def polytope_map(V, F, polytope="tetrahedron", seed=0):
    """Flip-free map of the mesh onto the unit sphere through the polytope layout."""
    n = len(V)
    corners, paths = polytope_layout(V, F, polytope, seed)
    X = POLYTOPES[polytope]["X"].copy()
    P = np.zeros((n, 3))
    fixed = np.zeros(n, bool)
    for i, j, p in paths:   # place path vertices on the polytope edge by arc length
        L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(V[p], axis=0), axis=1))]
        t = (L / L[-1])[:, None]
        P[p] = (1 - t) * X[i] + t * X[j]
        fixed[p] = True
    cut = set().union(*[path_edges(p) for _, _, p in paths])
    comp = face_components(F, cut)
    W = mean_value_weights(V, F)
    # Tutte inside each patch (weights only from that patch's faces)
    for c in range(comp.max() + 1):
        Fc = F[comp == c]
        verts = np.unique(Fc)
        inner = verts[~fixed[verts]]
        if len(inner) == 0:
            continue
        Wc = mean_value_weights(V, Fc)
        Lc = sp.diags(np.asarray(Wc.sum(1)).ravel()) - Wc
        bnd = verts[fixed[verts]]
        P[inner] = spla.spsolve(Lc[inner][:, inner].tocsc(), -(Lc[inner][:, bnd] @ P[bnd]))
    S = P / np.linalg.norm(P, axis=1, keepdims=True)
    det = np.einsum("ij,ij->i", S[F[:, 0]], np.cross(S[F[:, 1]], S[F[:, 2]]))
    if (det < 0).sum() > len(F) / 2:   # layout came out mirrored w.r.t. the mesh orientation
        S[:, 0] *= -1
    info = dict(corners=corners, paths=paths, patches=comp, n_patches=comp.max() + 1)
    return S, info


def one_hole_map(V, F, v=None, R=2.0):
    """The simplest flip-free spherical map (no polytope): remove one vertex, Tutte-embed the
    rest into a convex polygon on the plane z = -1, project centrally, put the vertex at the
    north pole.  Provably flip-free, but it crushes almost everything into one region, and on
    shapes with long protrusions it underflows double precision."""
    n = len(V)
    v = int(np.argmin(V[:, 1])) if v is None else v
    T = F[(F == v).any(1)]
    nxt = {}
    for t in T:
        k = list(t).index(v)
        nxt[t[(k + 1) % 3]] = t[(k + 2) % 3]
    ring = [next(iter(nxt))]
    while len(ring) < len(nxt):
        ring.append(nxt[ring[-1]])
    W = mean_value_weights(V, F)
    L = sp.diags(np.asarray(W.sum(1)).ravel()) - W
    fixed = np.zeros(n, bool)
    fixed[ring + [v]] = True
    P = np.zeros((n, 2))
    th = 2 * np.pi * np.arange(len(ring)) / len(ring)
    P[ring] = R * np.c_[np.cos(th), np.sin(th)]
    free = ~fixed
    P[free] = spla.spsolve(L[free][:, free].tocsc(), -(L[free][:, fixed] @ P[fixed]))
    S = np.c_[P, -np.ones(n)]
    S[v] = [0, 0, 1]
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    det = np.einsum("ij,ij->i", S[F[:, 0]], np.cross(S[F[:, 1]], S[F[:, 2]]))
    if (det < 0).sum() > len(F) / 2:
        S[:, 0] *= -1
    return S
