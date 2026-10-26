"""Polyhedral world maps: project the globe centrally (gnomonic) onto a regular polyhedron's faces,
cut along some edges, and unfold the faces into a flat net (cf. Fuller's Dymaxion map, which uses an
icosahedron with its own, nearly equal-area face projection).

Everything is raster-based: a pixel of the net -> its face -> barycentric point on the face -> the
sphere, so no polygon clipping across faces is needed.
"""
import numpy as np


def icosahedron():
    t = (1 + 5 ** 0.5) / 2
    V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                  [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    F = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
                  [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
                  [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    return V, F


def octahedron():
    V = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    F = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    return V, F


def lonlat_to_xyz(lon, lat):
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)


def xyz_to_lonlat(p):
    return np.arctan2(p[..., 1], p[..., 0]), np.arcsin(np.clip(p[..., 2], -1, 1))


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def edges_of(F):
    E = {}
    for f, tri in enumerate(F):
        for k in range(3):
            e = tuple(sorted((tri[k], tri[(k + 1) % 3])))
            E.setdefault(e, []).append(f)
    return E


def arc_weight(V, e, pts, delta):
    """How many sample points lie within angular distance delta of the great-circle arc e."""
    a, b = V[e[0]], V[e[1]]
    n = np.cross(a, b)
    n /= np.linalg.norm(n)
    near_plane = np.abs(pts @ n) < np.sin(delta)
    # between a and b: on the arc side of both endpoints
    mid = (a + b) / np.linalg.norm(a + b)
    within = pts @ mid > a @ mid - 1e-9
    return int((near_plane & within).sum())


def unfold(V, F, tree):
    """Lay out faces in the plane along a spanning tree {child: (parent, shared edge)}; returns
    {face: 3x2 corners in face-vertex order} or None if two faces overlap."""
    s = 1.0
    root = next(f for f in range(len(F)) if f not in tree)
    pos = {root: np.array([[0, 0], [s, 0], [s / 2, s * 3 ** 0.5 / 2]])}
    order = [root]
    children = {}
    for c, (p, e) in tree.items():
        children.setdefault(p, []).append((c, e))
    while order:
        p = order.pop()
        for c, e in children.get(p, []):
            P = pos[p]
            i0, i1 = [list(F[p]).index(v) for v in e]
            A, B = P[i0], P[i1]
            C_par = P[3 - i0 - i1]
            mid = (A + B) / 2
            apex = 2 * mid - C_par                       # reflect the parent's third corner
            Q = np.zeros((3, 2))
            for k, v in enumerate(F[c]):
                Q[k] = A if v == e[0] else B if v == e[1] else apex
            pos[c] = Q
            order.append(c)
    cents = np.array([pos[f].mean(0) for f in range(len(F))])
    d = np.linalg.norm(cents[:, None] - cents[None], axis=-1) + np.eye(len(F)) * 9
    return pos if d.min() > 0.3 * s else None


def best_net(V, F, land_pts, rng, n_rot=1500, delta=np.radians(1.5)):
    """Search orientations; for each, cut the edges carrying the least land (minimum-land
    complement of a maximum spanning tree), keeping only nets that unfold without overlap."""
    E = edges_of(F)
    keys = list(E)
    best = None
    for it in range(n_rot):
        R = np.eye(3) if it == 0 else random_rotation(rng)
        VR = V @ R.T
        w = np.array([arc_weight(VR, e, land_pts, delta) for e in keys], float)
        for trial in range(6):
            ww = w + (rng.random(len(w)) * (w.max() + 1) * 0.3 if trial else 0)
            # Kruskal, heaviest edges first = kept (folds); the rest are cut
            parent = list(range(len(F)))
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            kept = []
            for k in np.argsort(-ww):
                f0, f1 = E[keys[k]]
                r0, r1 = find(f0), find(f1)
                if r0 != r1:
                    parent[r0] = r1
                    kept.append(k)
            cut = w.sum() - w[kept].sum()
            if best is not None and cut >= best[0]:
                continue
            adj = {}
            for k in kept:
                f0, f1 = E[keys[k]]
                adj.setdefault(f0, []).append((f1, keys[k]))
                adj.setdefault(f1, []).append((f0, keys[k]))
            tree, seen, stack = {}, {0}, [0]
            while stack:
                p = stack.pop()
                for c, e in adj.get(p, []):
                    if c not in seen:
                        seen.add(c)
                        tree[c] = (p, e)
                        stack.append(c)
            pos = unfold(VR, F, tree)
            if pos is not None:
                best = (cut, R, tree, pos, [keys[k] for k in range(len(keys)) if k not in kept])
                break
    return best


def render(V, F, pos, sample, res=1600, pad=0.03):
    """Rasterize the net. sample(p) maps unit vectors (n, 3) to RGB (n, 3) floats in [0, 1].
    Returns the image (H, W, 3), a mask of covered pixels, and for each pixel its sphere point."""
    allp = np.concatenate(list(pos.values()))
    lo, hi = allp.min(0) - pad, allp.max(0) + pad
    W = res
    H = int(res * (hi[1] - lo[1]) / (hi[0] - lo[0]))
    xs = lo[0] + (np.arange(W) + 0.5) / W * (hi[0] - lo[0])
    ys = hi[1] - (np.arange(H) + 0.5) / H * (hi[1] - lo[1])
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X.ravel(), Y.ravel()], 1)
    img = np.ones((H * W, 3))
    pts = np.full((H * W, 3), np.nan)
    for f, T in pos.items():
        M = np.c_[T[1] - T[0], T[2] - T[0]]
        bc = np.linalg.solve(M, (P - T[0]).T).T
        l = np.c_[1 - bc.sum(1), bc]
        inside = (l > -1e-9).all(1) & np.isnan(pts[:, 0])
        q = l[inside] @ V[F[f]]                           # gnomonic: a point on the flat face
        pts[inside] = q / np.linalg.norm(q, axis=1, keepdims=True)
    ok = ~np.isnan(pts[:, 0])
    img[ok] = sample(pts[ok])
    return img.reshape(H, W, 3), ok.reshape(H, W), pts.reshape(H, W, 3), (lo, hi)


def gnomonic_sigmas(V, F, p):
    """Principal stretches (sphere -> face plane) of the per-face gnomonic map at unit vectors p,
    scaled so the net's total area equals the sphere's (4 pi). Returns (n, 2), sorted descending."""
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    h = np.einsum("ij,ij->i", n, V[F[:, 0]])              # distance of each face plane from the center
    f = np.argmax(p @ n.T, axis=1)
    c = np.einsum("ij,ij->i", p, n[f])                     # cos(angle from the face center)
    s = np.stack([h[f] / c**2, h[f] / c], 1)               # radial, tangential
    face_area = 0.5 * np.linalg.norm(np.cross(V[F[0, 1]] - V[F[0, 0]], V[F[0, 2]] - V[F[0, 0]]))
    k = np.sqrt(4 * np.pi / (len(F) * face_area))
    return s * k


def to_net(V, F, pos, p):
    """Net coordinates (n, 2) of unit vectors p (n, 3), for faces laid out by `unfold`."""
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    f = np.argmax(p @ n.T, axis=1)
    out = np.empty((len(p), 2))
    for k, (q, fk) in enumerate(zip(p, f)):
        T = V[F[fk]]
        x = q * (n[fk] @ T[0]) / (q @ n[fk])            # central projection onto the face plane
        l = np.linalg.lstsq(T.T, x, rcond=None)[0]       # barycentric (sums to 1 on the plane)
        out[k] = l @ pos[fk]
    return out


def rotate_net(pos, angle):
    c, s = np.cos(angle), np.sin(angle)
    Rm = np.array([[c, -s], [s, c]])
    return {f: T @ Rm.T for f, T in pos.items()}
