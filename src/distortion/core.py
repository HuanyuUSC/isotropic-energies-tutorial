"""One machine for every demo: per-element distortion energies, sparse assembly, inversion-safe steps,
and one projected-Newton loop.

    Elastic(T, energy, Dm)          sum_t w_t psi(F_t),  F_t = D_s D_m^{-1}: triangles in 2D, tetrahedra in 3D,
                                    3D triangles flattened in their own frames, sphere tets (apex at the origin)
    Springs, Inertia                quadratic terms (handles; implicit Euler's inertia)
    Problem(n, d, terms, pinned)    the sum of terms; any term may also add a sparse matrix (contact, planes)
    minimize(problem, X)            gradient descent / Newton / projected Newton, with the line search capped by
                                    every term's safe step (no inversion, no crossing a plane or a contact)

Speed: the per-element work (SVD, closed-form eigensystem, B^T H B) runs as batched NumPy on thread chunks
(NumPy releases the GIL); the sparsity pattern is analysed once and every iteration only scatters values;
the linear solve uses MKL PARDISO (multithreaded, symbolic factorization reused) when available.
"""
import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .eigensystem import projected_hessian, psi_and_P, signed_singular_values, signed_svd

# ------------------------------------------------------------------ parallel per-element work
# The per-element work is independent across elements, like tbb::parallel_for. DISTORTION_THREADS=1 turns it off.
# Measured on a 24-core machine (15k tets): 1.8x at 8 threads, slower beyond (memory-bound batched matmuls).
THREADS = int(os.environ.get("DISTORTION_THREADS", min(8, os.cpu_count() or 1)))
_POOL = ThreadPoolExecutor(THREADS) if THREADS > 1 else None
if _POOL is not None:
    try:                                   # one level of parallelism: our threads, not OpenBLAS's inside them
        from threadpoolctl import threadpool_limits
        threadpool_limits(1, user_api="blas")
    except ImportError:
        pass
MIN_CHUNK = 256


def parallel_chunks(n, fn):
    """Run fn(slice) over ~THREADS contiguous chunks of range(n); results in chunk order."""
    k = min(THREADS, max(1, n // MIN_CHUNK))
    if _POOL is None or k == 1:
        return [fn(slice(0, n))]
    edges = np.linspace(0, n, k + 1).astype(int)
    return list(_POOL.map(fn, [slice(a, b) for a, b in zip(edges[:-1], edges[1:])]))


# ------------------------------------------------------------------ the elastic term
def rest_frames(V3, F):
    """D_m of 3D triangles, each written in its own 2D frame (for flattening a surface): (t, 2, 2)."""
    e1, e2 = V3[F[:, 1]] - V3[F[:, 0]], V3[F[:, 2]] - V3[F[:, 0]]
    x = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
    n = np.cross(e1, e2)
    y = np.cross(n / np.linalg.norm(n, axis=1, keepdims=True), x)
    dot = lambda a, b: np.einsum("ij,ij->i", a, b)
    return np.stack([np.stack([dot(e1, x), dot(e1, y)], 1), np.stack([dot(e2, x), dot(e2, y)], 1)], 2)


class Elastic:
    """sum_t w_t psi(F_t) over simplices T (t, d+1) of the d-dimensional unknowns X (n, d).

    Dm: rest edge matrices (t, d, d); default from rest positions V. weights: default the rest volumes
    |det D_m| / d!. Every element is independent: batched SVD -> closed-form eigensystem -> projected
    d^2 x d^2 Hessian -> B^T H B, in parallel chunks."""

    def __init__(self, T, energy, V=None, Dm=None, weights=None, projection="clamp", offset=0):
        T = np.asarray(T) + offset
        if Dm is None:
            Dm = np.stack([V[T[:, i] - offset] - V[T[:, 0] - offset] for i in range(1, T.shape[1])], 2)
        self.T, self.energy, self.projection, self.offset = T, energy, projection, offset
        self.tmax = int(T.max())
        d = self.d = Dm.shape[1]
        self.vol = np.abs(np.linalg.det(Dm)) / math.factorial(d)
        self.w = self.vol if weights is None else np.asarray(weights, float)
        self.Dinv = np.linalg.inv(Dm)
        G = np.zeros((len(T), d + 1, d))
        G[:, 1:] = self.Dinv                       # dF_ab/dx_{k,c} = delta_ac Dinv[k-1, b]: rows of D_m^{-1}
        G[:, 0] = -G[:, 1:].sum(1)
        self.B = np.einsum("ac,tkb->tabkc", np.eye(d), G).reshape(len(T), d * d, d * (d + 1))
        self.dof = (d * T[:, :, None] + np.arange(d)).reshape(len(T), d * (d + 1))
        self.barrier = getattr(energy, "barrier", False)

    def _F(self, X, c=slice(None)):
        T = self.T[c]
        if len(X) <= self.tmax:                    # X is this body's own vertices (multi-body scenes)
            T = T - self.offset
        Ds = np.stack([X[T[:, i]] - X[T[:, 0]] for i in range(1, self.d + 1)], 2)
        return Ds @ self.Dinv[c]

    deformation = _F

    def dets(self, X):
        return np.linalg.det(self._F(X))

    def value(self, X):
        def chunk(c):
            Fd = self._F(X, c)
            if hasattr(self.energy, "density"):    # an energy of F itself (e.g. |F - I|^2, not rotation invariant)
                return float(self.w[c] @ self.energy.density(Fd)[0])
            sv = signed_singular_values(Fd)
            if self.barrier and (sv[:, -1] <= 0).any():
                return np.inf                      # inverted: rejected by the line search
            return float(self.w[c] @ self.energy.psi(sv))
        return float(sum(parallel_chunks(len(self.T), chunk)))

    def terms(self, X, mode="clamp"):
        """Energy, gradient (n d), per-element Hessians (t, d(d+1), d(d+1)) or None (mode=None, gradient only).
        mode: 'clamp' / 'abs' projection of the eigensystem, or 'none' for the true (indefinite) Hessian."""
        mode = self.projection if mode == "default" else mode
        dd = self.d * self.d

        def chunk(c):
            Fd = self._F(X, c)
            B, w = self.B[c], self.w[c]
            if hasattr(self.energy, "density"):
                psi, P = self.energy.density(Fd)
                HF = None if mode is None else self.energy.hessF(Fd)
            else:
                usv = signed_svd(Fd)               # one SVD per element serves the stress and the Hessian
                psi, P = psi_and_P(self.energy, Fd, usv)
                HF = None if mode is None else projected_hessian(self.energy, Fd, mode, usv)
            gt = (P.reshape(len(Fd), 1, dd) @ B)[:, 0] * w[:, None]
            Ht = None if HF is None else (np.swapaxes(B, 1, 2) @ (HF @ B)) * w[:, None, None]   # batched BLAS
            return float(w @ psi), gt, Ht
        parts = parallel_chunks(len(self.T), chunk)
        gt = np.concatenate([q[1] for q in parts])
        Ht = None if mode is None else np.concatenate([q[2] for q in parts])
        return sum(q[0] for q in parts), gt, Ht

    def max_step(self, X, D):
        """0.9 x the largest a with det(D_s(X + a D)) keeping its sign in every element (degree-d polynomial in a);
        infinite for energies without a barrier (they may invert, e.g. ARAP)."""
        if not self.barrier:
            return np.inf
        T, d = self.T, self.d
        E = [X[T[:, i]] - X[T[:, 0]] for i in range(1, d + 1)]
        Q = [D[T[:, i]] - D[T[:, 0]] for i in range(1, d + 1)]
        if d == 2:
            cr = lambda a, b: a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
            c = [cr(*E), cr(E[0], Q[1]) + cr(Q[0], E[1]), cr(*Q)]
        else:
            det3 = lambda a, b, c: np.einsum("ij,ij->i", a, np.cross(b, c))
            c = [det3(*E), det3(Q[0], E[1], E[2]) + det3(E[0], Q[1], E[2]) + det3(E[0], E[1], Q[2]),
                 det3(E[0], Q[1], Q[2]) + det3(Q[0], E[1], Q[2]) + det3(Q[0], Q[1], E[2]), det3(*Q)]
        return 0.9 * _smallest_positive_root(c, a_max=1.2)   # stay 10% short of the first flip (only a <= 1 matters)


def _smallest_positive_root(c, a_max=None):
    """Smallest positive real root over elements of c[0] + c[1] a + ... (batched companion matrices).
    With a_max, elements that provably keep their sign on [0, a_max] (|c1| a + |c2| a^2 + ... < |c0|)
    are skipped first: in a typical Newton step almost every element is far from flipping."""
    c = np.stack(c, 1)
    if a_max is not None:
        powers = a_max ** np.arange(c.shape[1])
        risky = np.abs(c[:, 1:]) @ powers[1:] >= np.abs(c[:, 0])
        c = c[risky]
        if not len(c):
            return np.inf
    deg = c.shape[1] - 1
    lead = c[:, -1]
    lead = np.where(np.abs(lead) < 1e-300, 1e-300, lead)
    comp = np.zeros((len(c), deg, deg))
    comp[:, 0, :] = -c[:, -2::-1] / lead[:, None]
    for k in range(1, deg):
        comp[:, k, k - 1] = 1
    r = np.linalg.eigvals(comp)
    real = np.abs(r.imag) < 1e-9 * (1 + np.abs(r.real))
    pos = np.where(real & (r.real > 0), r.real, np.inf)
    return float(pos.min()) if pos.size else np.inf


# ------------------------------------------------------------------ simple terms
class Springs:
    """k/2 sum |x_h - target_h|^2 over handle vertices."""

    def __init__(self, idx, target, k=1e3):
        self.idx, self.target, self.k = np.asarray(idx), np.asarray(target, float), k

    def value(self, X):
        return 0.5 * self.k * np.sum((X[self.idx] - self.target) ** 2)

    def terms(self, X, n, d):
        g = np.zeros((n, d))
        g[self.idx] = self.k * (X[self.idx] - self.target)
        diag = np.zeros((n, d))
        diag[self.idx] = self.k
        return self.value(X), g.ravel(), diag.ravel()


class Inertia:
    """w/2 |X - Xt|_M^2 (lumped mass m per vertex): implicit Euler's inertia, or (tiny w) a quasistatic
    regularizer that removes rigid null spaces."""

    def __init__(self, m, w, Xt):
        self.m, self.w, self.Xt = m, w, Xt

    def value(self, X):
        return 0.5 * self.w * np.sum(self.m[:, None] * (X - self.Xt) ** 2)

    def terms(self, X, n, d):
        M = np.repeat(self.m, d) * self.w
        return self.value(X), M * (X - self.Xt).ravel(), M


# ------------------------------------------------------------------ assembly and linear solves
class SparsePattern:
    """The fixed sparsity pattern of the element Hessians (like Eigen's analyzePattern): built once,
    then every iteration only scatters the element values into a preallocated CSC data array."""

    def __init__(self, dofs, n, diagonal=True):
        rows = np.concatenate([np.repeat(d, d.shape[1], axis=1).ravel() for d in dofs])
        cols = np.concatenate([np.tile(d, (1, d.shape[1])).ravel() for d in dofs])
        if diagonal:                                         # keep every diagonal entry (mass, pins, planes)
            rows, cols = np.r_[rows, np.arange(n)], np.r_[cols, np.arange(n)]
        key, self.inv = np.unique(cols.astype(np.int64) * n + rows, return_inverse=True)   # CSC order
        self.n, self.nnz = n, len(key)
        self.indices = (key % n).astype(np.int32)
        self.indptr = np.searchsorted(key // n, np.arange(n + 1)).astype(np.int32)
        self.diag = self.inv[-n:] if diagonal else None      # slot of each diagonal entry
        self.ne = len(rows) - (n if diagonal else 0)

    def matrix(self, Hts, diag=None):
        vals = np.concatenate([H.ravel() for H in Hts]) if Hts else np.zeros(0)
        data = np.bincount(self.inv[:self.ne], weights=vals, minlength=self.nnz)
        if diag is not None:
            data[self.diag] += diag
        H = sp.csc_matrix((data, self.indices, self.indptr), shape=(self.n, self.n))
        H.has_sorted_indices = True
        return H


def superlu_spd(A, b):
    """SPD solve with symmetric-mode SuperLU and a symmetric ordering (the fallback; no symbolic reuse)."""
    return spla.splu(A.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                     options=dict(SymmetricMode=True)).solve(b)


class LinearSolver:
    """Sparse SPD solves that reuse PARDISO's symbolic analysis (Eigen's analyzePattern + factorize).

    The element pattern is fixed, but contact couples new vertex pairs as the active set changes; the
    pattern then grows to the union seen so far (optionally seeded with upcoming couplings), and shrinks
    back if it outgrows the current matrix by `max_ratio`. Analysis is redone only when the pattern
    changes; every other iteration is numeric factorization + solve. SuperLU if PARDISO is unavailable."""

    def __init__(self, seed=None, max_ratio=None, backend="auto"):
        self.seed, self.max_ratio, self.keys, self.n, self.ps = seed, max_ratio, None, -1, None
        self.analyses = self.solves = 0
        self.pp, self.backend = None, backend
        if backend in ("auto", "pardiso"):
            try:
                import pypardiso
                self.pp = pypardiso
            except ImportError:
                pass

    def _analyze(self, keys, n):
        if self.ps is not None:
            try:
                self.ps.free_memory(everything=True)
            except Exception:
                pass
        self.keys, self.n = keys, n
        rows, cols = keys // n, keys % n
        self.indices = cols.astype(np.int32)
        self.indptr = np.searchsorted(rows, np.arange(n + 1)).astype(np.int32)
        self.ps = self.pp.PyPardisoSolver(mtype=2)                  # real symmetric positive definite
        A = sp.csr_matrix((np.where(rows == cols, 1.0, 0.0), self.indices, self.indptr), shape=(n, n))
        self.ps.set_phase(11)
        self.ps._call_pardiso(A, np.zeros(n))
        self.analyses += 1

    MIN_PARDISO_NNZ = 150000      # below this many nonzeros SuperLU wins (PARDISO's setup + re-analysis overhead);
                                  # 3D meshes pass it at a few thousand vertices, 2D ones much later

    def solve(self, A, b):
        n = A.shape[0]
        if self.pp is None or (self.backend == "auto" and A.nnz < self.MIN_PARDISO_NNZ):
            return superlu_spd(A, b)
        U = sp.triu(A, format="coo")                                  # PARDISO's SPD mode reads the upper triangle
        keys = U.row.astype(np.int64) * n + U.col
        if self.keys is None or n != self.n:
            new = np.unique(keys)
        else:
            pos = np.searchsorted(self.keys, keys)
            inside = (pos < len(self.keys)) & (self.keys[np.minimum(pos, len(self.keys) - 1)] == keys)
            new = None if inside.all() else np.unique(np.concatenate([self.keys, keys]))
        if new is not None and self.max_ratio is not None and len(new) > self.max_ratio * len(np.unique(keys)):
            new = np.unique(keys)
        if new is not None:
            if self.seed is not None:
                new = np.union1d(new, self.seed(n))
            self._analyze(new, n)
        pos = np.searchsorted(self.keys, keys)
        data = np.bincount(pos, weights=U.data, minlength=len(self.keys))
        M = sp.csr_matrix((data, self.indices, self.indptr), shape=(n, n))
        try:
            self.ps.set_phase(23)                                    # numeric factorization + solve only
            x = self.ps._call_pardiso(M, b)
        except Exception:
            self.keys = None
            return superlu_spd(A, b)
        self.solves += 1
        return x


# ------------------------------------------------------------------ the problem and the solver
class Problem:
    """min_X sum of terms over X (n, d), pinned vertices fixed.

    terms: Elastic (element Hessians, assembled through one fixed pattern), Springs / Inertia (diagonal),
    and any object with value(X), terms(X, n, d) -> (f, g, diag or None[, sparse H]) and optionally
    max_step(X, D) and admissible(X) (e.g. planes and IPC contact)."""

    def __init__(self, n, d, terms, pinned=None, solver=None, damping=0.0):
        self.n, self.d, self.terms, self.damping = n, d, list(terms), damping   # damping: Levenberg shift, relative
        self.pinned = np.zeros(n, bool) if pinned is None else np.asarray(pinned, bool)
        self.free = np.repeat(~self.pinned, d)
        self.elastic = [t for t in self.terms if isinstance(t, Elastic)]
        self.pattern = SparsePattern([e.dof for e in self.elastic], d * n)
        self.solver = solver if solver is not None else LinearSolver()

    def value(self, X):
        tot = 0.0
        for t in self.terms:
            if hasattr(t, "admissible") and not t.admissible(X):
                return np.inf
            v = t.value(X)
            if not np.isfinite(v):
                return np.inf
            tot += v
        return tot

    def evaluate(self, X, mode="clamp"):
        """f, g (n d), H (sparse, n d x n d; None for mode=None)."""
        f, g = 0.0, np.zeros(self.d * self.n)
        Hts, diag, extra = [], np.zeros(self.d * self.n), []
        for t in self.terms:
            if isinstance(t, Elastic):
                fe, gt, Ht = t.terms(X, mode)
                f += fe
                g += np.bincount(t.dof.ravel(), weights=gt.ravel(), minlength=self.d * self.n)
                Hts.append(Ht)
            else:
                r = t.terms(X, self.n, self.d)
                f += r[0]; g += r[1]
                if r[2] is not None:
                    diag += r[2]
                if len(r) > 3 and r[3] is not None and r[3].nnz:
                    extra.append(r[3])
        if mode is None:
            return f, g, None
        H = self.pattern.matrix(Hts, diag)
        if self.damping:                    # + damping * max|diag H| * I: removes null spaces (free boundary)
            H.data[self.pattern.diag] += self.damping * np.abs(H.diagonal()).max()
        for E in extra:                     # contact / planes: their pattern changes with the active set
            H = H + E
        return f, g, H

    def max_step(self, X, D):
        return min([1.0] + [t.max_step(X, D) for t in self.terms if hasattr(t, "max_step")])

    def direction(self, X, g, H, method):
        fr = self.free
        D = np.zeros(self.d * self.n)
        if method == "gd":
            D[fr] = -g[fr]
        else:
            Hf = H[fr][:, fr].tocsc()
            if method == "newton":          # the true Hessian may be indefinite: no SPD solver
                D[fr] = spla.spsolve(Hf, -g[fr])
            else:
                D[fr] = self.solver.solve(Hf, -g[fr])
        return D.reshape(-1, self.d)


def minimize(P, X, method="pnewton", iters=100, tol_dir=None, tol_step=None, tol_rel_f=1e-9,
             keep_going_when_capped=True, f_floor=1.0, history=None, callback=None):
    """Minimize problem P from X. method: 'gd', 'newton' (true Hessian), 'pnewton' (projected).

    Each iteration: direction D, then a <= every term's safe step (no inversion, no crossing a plane, CCD
    for contact), then Armijo backtracking. Stops when the Newton direction is invisible (|D| < tol_dir,
    after taking it), the step is invisible (|a D| < tol_step) or the energy stops changing
    (|f - f_new| < tol_rel_f max(f_floor, |f|)). With keep_going_when_capped, a step cut short by a constraint is never taken as
    convergence (stopping next to a contact would freeze the bodies). history gets one dict per iteration."""
    mode = {"gd": None, "newton": "none", "pnewton": "default"}[method]
    it = -1
    for it in range(iters):
        f, g, H = P.evaluate(X, mode)
        D = P.direction(X, g, H, method)
        slope = float(g @ D.ravel())
        info = dict(f=f, slope=slope, uphill=slope >= 0)
        if slope >= 0:                     # not a descent direction (indefinite Hessian): no step possible
            info["alpha"] = 0.0
            if history is not None:
                history.append(info)
            break
        a = P.max_step(X, D)
        capped = keep_going_when_capped and a < 1.0
        if method == "gd":
            a = min(a, 1.0 / max(1e-12, np.abs(D).max()))   # a sane first guess for gradient descent
        while True:
            fn = P.value(X + a * D)
            if fn <= f + 1e-4 * a * slope or a <= 1e-14:
                break
            a *= 0.5
        X = X + a * D
        info.update(alpha=a, f_next=fn)
        if history is not None:
            history.append(info)
        if callback is not None:
            callback(it, X, info)
        Dmax = np.abs(D).max()
        if tol_dir is not None and Dmax < tol_dir:
            break
        if capped:
            continue
        if tol_step is not None and a * Dmax < tol_step:
            break
        if tol_rel_f is not None and abs(f - fn) < tol_rel_f * max(f_floor, abs(f)):
            break
    P.iterations = it + 1
    return X
