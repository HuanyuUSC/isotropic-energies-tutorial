"""Texture atlases: cut a closed surface into disk-like charts, flatten each chart with the planar
Symmetric Dirichlet energy (projected Newton, flip-free steps), and pack the charts into one atlas.

The solver is the shared core (core.Elastic + core.minimize); only the rest shapes come from the 3D
triangles, each laid out in its own 2D frame.
"""
import heapq

import numpy as np
import scipy.sparse.linalg as spla

from .eigensystem import SymmetricDirichlet
from .core import LinearSolver, Problem, minimize, rest_frames
from .planar import MeshEnergy, dets


# ---------------------------------------------------------------- charts
def face_adjacency(F):
    E = {}
    for f, tri in enumerate(F):
        for k in range(3):
            E.setdefault(tuple(sorted((tri[k], tri[(k + 1) % 3]))), []).append(f)
    adj = [[] for _ in F]
    for fs in E.values():
        if len(fs) == 2:
            adj[fs[0]].append(fs[1])
            adj[fs[1]].append(fs[0])
    return adj


def grow_charts(V, F, k, seed=0):
    """k charts by farthest-point seeds on face centroids and multi-source Dijkstra over faces,
    with a penalty on normal changes so cuts follow creases."""
    C = V[F].mean(1)
    N = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    adj = face_adjacency(F)
    seeds = [seed]
    d = np.linalg.norm(C - C[seed], axis=1)
    for _ in range(k - 1):
        seeds.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(C - C[seeds[-1]], axis=1))
    lab = -np.ones(len(F), int)
    dist = np.full(len(F), np.inf)
    pq = [(0.0, s, i) for i, s in enumerate(seeds)]
    for _, s, i in pq:
        dist[s] = 0
    heapq.heapify(pq)
    while pq:
        dd, f, i = heapq.heappop(pq)
        if lab[f] >= 0:
            continue
        lab[f] = i
        for g in adj[f]:
            if lab[g] < 0:
                w = np.linalg.norm(C[f] - C[g]) * (1 + 4 * (1 - N[f] @ N[g]))
                if dd + w < dist[g]:
                    dist[g] = dd + w
                    heapq.heappush(pq, (dd + w, g, i))
    return lab


def is_disk(F):
    E = set()
    for tri in F:
        for k in range(3):
            E.add(tuple(sorted((tri[k], tri[(k + 1) % 3]))))
    nv = len(np.unique(F))
    import igl
    loops = igl.all_boundary_loop(np.ascontiguousarray(F, dtype=np.int64)) if hasattr(igl, "all_boundary_loop") else None
    one_loop = loops is None or len(loops) == 1
    return nv - len(E) + len(F) == 1 and one_loop


# ---------------------------------------------------------------- flattening
def local_energy(V3, F, energy):
    """The planar energy whose rest shapes are the 3D triangles, each in its own 2D frame."""
    return MeshEnergy(None, F, energy, Dm=rest_frames(V3, F))


def tutte(V3, F):
    """Tutte embedding (uniform weights, boundary on a circle; flip-free for a convex boundary), with the
    3D area. Returns X (n, 2)."""
    import igl
    import scipy.sparse as sp
    Fi = np.ascontiguousarray(F, dtype=np.int64)
    b = igl.boundary_loop(Fi)
    bc = igl.map_vertices_to_circle(np.ascontiguousarray(V3), b)
    n = len(V3)
    I = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    A = sp.coo_matrix((np.ones(len(I)), (I[:, 0], I[:, 1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(float)
    L = (sp.diags(np.asarray(A.sum(1)).ravel()) - A).tocsr()
    free = np.setdiff1d(np.arange(n), b)
    X = np.zeros((n, 2))
    X[b] = bc
    X[free] = spla.spsolve(L[free][:, free].tocsc(), -(L[free][:, b] @ bc))
    if dets(X, F).mean() < 0:
        X[:, 1] *= -1
    area = 0.5 * np.linalg.norm(np.cross(V3[F[:, 1]] - V3[F[:, 0]], V3[F[:, 2]] - V3[F[:, 0]]), axis=1).sum()
    X *= np.sqrt(area / (0.5 * np.abs(dets(X, F)).sum()))
    assert (dets(X, F) > 0).all(), "Tutte map flipped"
    return X


def flatten(V3, F, iters=60, energy=None, method="pnewton", X0=None, tol=1e-9, history=None):
    """Tutte -> minimize a planar distortion energy (default Symmetric Dirichlet) with the core solver.
    A free boundary leaves a rigid null space: a tiny Levenberg shift (1e-8 max|diag H|) removes it.
    Returns UV (n, 2) and the energy."""
    E = local_energy(V3, F, energy or SymmetricDirichlet())
    X = tutte(V3, F) if X0 is None else X0.copy()
    n = len(X)
    P = Problem(n, 2, [E], solver=LinearSolver(), damping=1e-8)
    X = minimize(P, X, method, iters=iters, tol_rel_f=tol, keep_going_when_capped=False, f_floor=0.0,
                 history=history)
    return X, E


# ---------------------------------------------------------------- packing
def pack(uvs, gap=0.04):
    """Rotate each chart to its principal axes, then shelf-pack by decreasing height."""
    out, boxes = [], []
    for X in uvs:
        X = X - X.mean(0)
        _, _, W = np.linalg.svd(X, full_matrices=False)
        if np.linalg.det(W) < 0:
            W[1] *= -1
        X = X @ W.T
        X -= X.min(0)
        out.append(X)
        boxes.append(X.max(0))
    total = sum(b[0] * b[1] for b in boxes)
    width = 1.6 * np.sqrt(total)
    order = np.argsort([-b[1] for b in boxes])
    x = y = shelf = 0.0
    off = [None] * len(uvs)
    for i in order:
        w, h = boxes[i]
        if x + w > width:
            x, y, shelf = 0.0, y + shelf + gap * width, 0.0
        off[i] = np.array([x, y])
        x += w + gap * width
        shelf = max(shelf, h)
    return [X + o for X, o in zip(out, off)]
