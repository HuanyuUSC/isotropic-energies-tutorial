"""Bijective spherical parameterization of a closed genus-0 mesh.

Re-implementation of H. Chen, "Efficient Bijective Spherical Parameterizations",
B.S. thesis, USTC 2021 (advisor Ligang Liu), Ch. 3:
  * each triangle + the sphere's center is a tetrahedron, so the map's deformation gradient
    is 3x3 and det F is proportional to that tetrahedron's volume; a barrier energy
    (Symmetric Dirichlet) then forbids spherical triangles from flipping, and a locally
    flip-free map of a closed sphere is globally bijective;
  * the sphere constraint is handled by optimizing E(P(x)), P = radial projection:
    tangent-projected Newton directions, and updates x <- P(x + a d);
  * the largest flip-free step is the smallest positive root of a cubic per triangle.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .core import Elastic, SparsePattern
from .eigensystem import SymmetricDirichlet, signed_svd


# ---------------------------------------------------------------- initialization
def _adjacency(F, n):
    E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1), axis=0)
    return sp.coo_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])),
                         shape=(n, n)).tocsr()


def triple(S, F):
    """det(s0, s1, s2) per triangle: 6x the signed volume of (center, s0, s1, s2)."""
    return np.einsum("ij,ij->i", S[F[:, 0]], np.cross(S[F[:, 1]], S[F[:, 2]]))


def cotan_laplacian(V, F):
    n = len(V)
    rows, cols, vals = [], [], []
    for k in range(3):
        o, i, j = F[:, k], F[:, (k + 1) % 3], F[:, (k + 2) % 3]
        a, b = V[i] - V[o], V[j] - V[o]
        cot = np.einsum("ij,ij->i", a, b) / np.linalg.norm(np.cross(a, b), axis=1)
        rows += [i, j, i, j]
        cols += [j, i, i, j]
        vals += [cot / 2, cot / 2, -cot / 2, -cot / 2]
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))


def lumped_mass(V, F):
    A = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    m = np.zeros(len(V))
    for k in range(3):
        np.add.at(m, F[:, k], A / 3)
    return m


def conformal_flow_init(V, F, steps=1000, dt=1e-2, tol=5e-3, repair_iters=500):
    """Conformalized mean-curvature flow [Kazhdan et al. 2012] -> radial projection,
    then local relaxation of any (rare) flipped triangles.  Returns flip-free points on S^2.
    The flow runs until the surface is round (max | |x| - 1 | < tol): radially projecting a
    surface that is not yet a sphere is not conformal (30 steps left the hand at sigma1/sigma2 ~ 1.7)."""
    L = cotan_laplacian(V, F)
    X = V.copy()
    for _ in range(steps):
        m = lumped_mass(X, F)
        X = spla.spsolve((sp.diags(m) - dt * L).tocsc(), m[:, None] * X)
        X -= (m[:, None] * X).sum(0) / m.sum()
        X /= np.sqrt(m.sum() / (4 * np.pi))
        if np.abs(np.linalg.norm(X, axis=1) - 1).max() < tol:
            break
    S = X / np.linalg.norm(X, axis=1, keepdims=True)
    if (triple(S, F) <= 0).sum() > len(F) / 2:
        S[:, 0] *= -1
    return repair_flips(S, F, repair_iters)


def repair_flips(S, F, max_iter=500):
    """Make a nearly-flip-free spherical map flip-free by local relaxation: vertices of flipped
    triangles (and a growing ring around them) move to the normalized average of their
    neighbors.  Raises if flips remain."""
    S = S / np.linalg.norm(S, axis=1, keepdims=True)
    A = _adjacency(F, len(S))
    grow = 1
    for it in range(max_iter):
        bad = triple(S, F) <= 0
        if not bad.any():
            return S
        mark = np.zeros(len(S), bool)
        mark[F[bad].ravel()] = True
        for _ in range(grow):
            mark |= (A @ mark.astype(float)) > 0
        idx = np.nonzero(mark)[0]
        for _ in range(10):
            M = A[idx] @ S
            S[idx] = M / np.linalg.norm(M, axis=1, keepdims=True)
        if it % 50 == 49:
            grow += 1
    raise RuntimeError("map still has flipped triangles")


# ---------------------------------------------------------------- the energy
def source_tets(V, F):
    """Per triangle, a tetrahedron (center, r0, r1, r2) with r_k on the unit sphere and the
    face (r0, r1, r2) congruent to the mesh triangle.  Returns inv(R) with R = [r0 r1 r2]."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e1 = b - a
    lab = np.linalg.norm(e1, axis=1)
    x = e1 / lab[:, None]
    n = np.cross(e1, c - a)
    y = np.cross(n / np.linalg.norm(n, axis=1)[:, None], x)
    P = np.stack([np.zeros(len(F)), lab, np.einsum("ij,ij->i", c - a, x)], 1)   # local 2D x-coords
    Q = np.stack([np.zeros(len(F)), np.zeros(len(F)), np.einsum("ij,ij->i", c - a, y)], 1)
    # circumcenter in the local frame
    bx, cx, cy = P[:, 1], P[:, 2], Q[:, 2]
    ux = bx / 2
    uy = (cx**2 + cy**2 - bx * cx) / (2 * cy)
    rho2 = ux**2 + uy**2
    # Slivers (circumradius near/above 1) cannot sit on the unit sphere: put the apex above
    # the centroid instead, at a height set by the mean squared vertex distance.
    sliver = rho2 > 0.81
    gx, gy = (P.sum(1) / 3), (Q.sum(1) / 3)
    r2 = ((P - gx[:, None])**2 + (Q - gy[:, None])**2).mean(1)
    ux, uy = np.where(sliver, gx, ux), np.where(sliver, gy, uy)
    h = np.sqrt(np.clip(np.where(sliver, 1 - r2, 1 - rho2), 0.01, None))
    R = np.stack([np.stack([P[:, k] - ux, Q[:, k] - uy, h], 1) for k in range(3)], 2)   # columns r_k
    return np.linalg.inv(R)


class SphericalEnergy(Elastic):
    """The spherical distortion energy as an ordinary core.Elastic term: each mesh triangle plus the sphere's
    center (an extra vertex, index n, fixed at the origin) is a tetrahedron, with the source tet of
    source_tets() as its rest shape and the triangle's area fraction as its weight."""

    def __init__(self, V, F, energy=None, projection="clamp"):
        self.F, self.n = F, len(V)
        A = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
        T = np.c_[np.full(len(F), len(V)), F]
        super().__init__(T, energy or SymmetricDirichlet(), Dm=np.linalg.inv(source_tets(V, F)),
                         weights=A / A.sum(), projection=projection)
        self.barrier = False                       # flips are prevented by the max step, not by E = inf
        self._pattern = SparsePattern([self.dof], 3 * (self.n + 1))

    def _ext(self, S):
        return np.vstack([S, np.zeros((1, 3))])

    def deformation(self, S):
        return self._F(self._ext(S))

    def value(self, S):
        return super().value(self._ext(S))

    def value_grad(self, S):
        f, gt, _ = self.terms(self._ext(S), None)
        g = np.bincount(self.dof.ravel(), weights=gt.ravel(), minlength=3 * (self.n + 1))[:3 * self.n]
        self._S = S
        return f, g.reshape(-1, 3), self.deformation(S)

    def _assemble(self, H_F):
        """Global matrix from per-tet 9x9 blocks w.r.t. vec(F) (the origin's rows dropped)."""
        Ht = (np.swapaxes(self.B, 1, 2) @ (H_F @ self.B)) * self.w[:, None, None]
        m = 3 * self.n
        return self._pattern.matrix([Ht])[:m, :m]

    def newton_hessian(self, Fd):
        """Second order: analytic eigensystem, projected to PSD (the same per-element code as every demo)."""
        _, _, Ht = self.terms(self._ext(self._S), "default")
        m = 3 * self.n
        return self._pattern.matrix([Ht])[:m, :m]

    def slim_hessian(self, Fd):
        """First order, SLIM [Rabinovich et al. 2017], tetrahedral form: the Hessian of the
        reweighted proxy |W (F - R)|^2 with W^T W = U diag(w^2) U^T, w_i^2 = psi_i / (2 (s_i - 1)),
        whose gradient matches the true energy's gradient at the current iterate."""
        U, s, _ = signed_svd(Fd)
        g = self.energy.grad(s)
        near = np.abs(s - 1) < 1e-6
        w2 = np.where(near, np.diagonal(self.energy.hess(s), axis1=-2, axis2=-1) / 2,
                      g / (2 * np.where(near, 1, s - 1)))
        w2 = np.maximum(w2, 1e-3 * np.abs(w2).max(axis=1, keepdims=True) + 1e-12)   # keep the proxy SPD
        W2 = np.einsum("tij,tj,tkj->tik", U, w2, U)
        return self._assemble(2 * np.einsum("tac,bd->tabcd", W2, np.eye(3)).reshape(len(Fd), 9, 9))

    def sobolev_hessian(self, V):
        """First order, Sobolev gradient: a fixed cotan Laplacian (+ a little mass) of the input mesh."""
        L = -cotan_laplacian(V, self.F)
        M = sp.diags(lumped_mass(V, self.F))
        A = (L + 1e-3 * M) / M.diagonal().sum()
        return sp.kron(A, sp.eye(3), format="csc")

    def value_grad_hess(self, S):
        f, g, Fd = self.value_grad(S)
        return f, g, self.newton_hessian(Fd)

    def safe_step(self, S, D):
        """0.9 x the largest step before any spherical triangle flips (core.Elastic.max_step, apex fixed)."""
        self.barrier = True
        try:
            return Elastic.max_step(self, self._ext(S), self._ext(D))
        finally:
            self.barrier = False


# ---------------------------------------------------------------- the solver
def max_step(S, D, F):
    """Largest a such that det(s_k + a d_k) > 0 for every triangle (cubic in a)."""
    s = [S[F[:, k]] for k in range(3)]
    d = [D[F[:, k]] for k in range(3)]
    det = lambda a, b, c: np.einsum("ij,ij->i", a, np.cross(b, c))
    c0 = det(*s)
    c1 = det(d[0], s[1], s[2]) + det(s[0], d[1], s[2]) + det(s[0], s[1], d[2])
    c2 = det(s[0], d[1], d[2]) + det(d[0], s[1], d[2]) + det(d[0], d[1], s[2])
    c3 = det(*d)
    # roots of c3 a^3 + c2 a^2 + c1 a + c0 via the batched companion matrix (scaled by c0 > 0)
    comp = np.zeros((len(F), 3, 3))
    comp[:, 1, 0] = comp[:, 2, 1] = 1
    lead = np.where(np.abs(c3) < 1e-300, 1e-300, c3)
    comp[:, 0, :] = -np.stack([c2, c1, c0], 1) / lead[:, None]
    r = np.linalg.eigvals(comp)
    real = np.abs(r.imag) < 1e-9 * (1 + np.abs(r.real))
    pos = np.where(real & (r.real > 0), r.real, np.inf).min(1)
    return pos.min()


def _block_diag(blocks):
    n = len(blocks)
    r = (3 * np.arange(n)[:, None, None] + np.arange(3)[None, :, None]).repeat(3, 2)
    c = np.swapaxes(r, 1, 2)
    return sp.csc_matrix((blocks.ravel(), (r.ravel(), c.ravel())), shape=(3 * n, 3 * n))


def tangent_newton_direction(S, g, H, reg=1e-4, curvature=False):
    """Newton step restricted to the sphere's tangent spaces: (T H T + C + mu T + N) d = -T g.
    C = -(s_v . g_v) T_v per vertex is the sphere's curvature term in the Hessian of E(P(x))
    (kept only where it is PSD).  The energy is invariant to rotating the whole sphere, so
    T H T is singular; the small mu T term (Levenberg-style) removes that null space."""
    n = len(S)
    SS = np.einsum("vi,vj->vij", S, S)
    N = _block_diag(SS)
    T = _block_diag(np.eye(3) - SS)
    mu = reg * np.median(H.diagonal())
    K = T @ H @ T + mu * T + N
    if curvature:
        c = np.maximum(-np.einsum("vi,vi->v", S, g), 0)
        K = K + _block_diag(c[:, None, None] * (np.eye(3) - SS))
    K = K.tocsc()
    rhs = -(T @ g.ravel())
    return spla.spsolve(K, rhs).reshape(n, 3)


def parameterize(V, F, S0=None, energy=None, schedule=(("slim", 0.05), ("newton", 1e-6)),
                 projection="clamp", curvature=True, iters=300, callback=None, verbose=True):
    """Minimize the spherical distortion energy from a flip-free initial map.

    schedule: sequence of (method, stop) with method in {"slim", "sobolev", "newton"}; each
    phase runs until the relative energy decrease of one iteration falls below `stop`
    (thesis strategy: a first-order preconditioner for the first steps, then Newton).
    Every phase uses the same tangent projection, flip-free max step and line search.
    Returns S and a history list of (seconds, energy, method)."""
    import time
    if S0 is None:
        from .polytope_init import polytope_map
        S0, _ = polytope_map(V, F, "tetrahedron")
    S = S0.copy()
    assert (triple(S, F) > 0).all(), "initial map must be flip-free"
    E = SphericalEnergy(V, F, energy, projection)
    t0 = time.perf_counter()
    f, g, Fd = E.value_grad(S)
    history = [(0.0, f, schedule[0][0])]
    H_sob = None
    it = 0
    for method, stop in schedule:
        reg = 1e-4 if method == "newton" else 1e-8   # LM damping (adapted for Newton)
        while it < iters:
            if method == "newton":
                H = E.newton_hessian(Fd)
            elif method == "slim":
                H = E.slim_hessian(Fd)
            else:
                H_sob = E.sobolev_hessian(V) if H_sob is None else H_sob
                H = H_sob
            D = tangent_newton_direction(S, g, H, reg, curvature and method == "newton")
            a = min(1.0, E.safe_step(S, D))
            slope = float(np.sum(g * D))
            while True:   # backtracking Armijo line search along the great circles
                Sn = S + a * D
                Sn /= np.linalg.norm(Sn, axis=1, keepdims=True)
                fn = E.value(Sn)
                if fn <= f + 1e-4 * a * slope or a < 1e-12:
                    break
                a *= 0.5
            if not fn < f:   # no decrease even for a tiny step: this phase is done
                break
            if method == "newton":
                reg = reg * 10 if a < 0.1 else max(reg / 3, 1e-6) if a == 1.0 else reg
            S = Sn
            f_old = f
            f, g, Fd = E.value_grad(S)
            it += 1
            history.append((time.perf_counter() - t0, f, method))
            if callback:
                callback(it, S, f, method)
            if verbose:
                print(f"{method:7s} iter {it:3d}  E = {f:.6f}  step = {a:.3g}  flips = {(triple(S, F) <= 0).sum()}")
            if 0 <= (f_old - f) / f < stop:
                break
    return S, history


def untangle(S, F, rings=2, rel_eps=0.1, max_rounds=8):
    """Fix (nearly) degenerate flipped spherical triangles with minimal motion: on the vertices of
    flipped triangles plus `rings` rings, minimize sum_t max(0, eps_t - det_t)^2 / eps_t^2 over
    points on the sphere (L-BFGS on x, evaluated at x / |x|).  eps_t = rel_eps x the median |det|
    of the triangles around t, so tiny triangles get tiny targets.  Raises if flips remain."""
    from scipy.optimize import minimize
    S = S / np.linalg.norm(S, axis=1, keepdims=True)
    A = _adjacency(F, len(S))
    for _ in range(max_rounds):
        d = triple(S, F)
        bad = d <= 0
        if not bad.any():
            return S
        mark = np.zeros(len(S), bool)
        mark[F[bad].ravel()] = True
        for _ in range(rings):
            mark |= (A @ mark.astype(float)) > 0
        idx = np.nonzero(mark)[0]
        T = np.nonzero(mark[F].any(1))[0]                 # triangles touching the region
        Ft = F[T]
        # local scale: median |det| of the triangles sharing a vertex with t
        vmed = np.zeros(len(S))
        absd = np.abs(d)
        for k in range(3):
            np.maximum.at(vmed, F[:, k], 0)
        inc = sp.coo_matrix((np.ones(3 * len(F)), (F.ravel(), np.repeat(np.arange(len(F)), 3))),
                            shape=(len(S), len(F))).tocsr()
        loc = np.array([np.median(absd[inc[Ft[i]].indices]) for i in range(len(Ft))])
        eps = rel_eps * loc
        pos = -np.ones(len(S), int)
        pos[idx] = np.arange(len(idx))

        def f(z):
            X = S.copy()
            Z = z.reshape(-1, 3)
            nz = np.linalg.norm(Z, axis=1, keepdims=True)
            X[idx] = Z / nz
            s0, s1, s2 = X[Ft[:, 0]], X[Ft[:, 1]], X[Ft[:, 2]]
            det = np.einsum("ij,ij->i", s0, np.cross(s1, s2))
            r = np.maximum(0, eps - det) / eps
            val = np.sum(r**2)
            w = (-2 * r / eps)[:, None]
            G = np.zeros_like(X)
            np.add.at(G, Ft[:, 0], w * np.cross(s1, s2))
            np.add.at(G, Ft[:, 1], w * np.cross(s2, s0))
            np.add.at(G, Ft[:, 2], w * np.cross(s0, s1))
            Gi = G[idx]
            U = Z / nz
            Gz = (Gi - np.einsum("vi,vi->v", Gi, U)[:, None] * U) / nz   # chain rule through x / |x|
            return val, Gz.ravel()

        res = minimize(f, S[idx].ravel(), jac=True, method="L-BFGS-B", options=dict(maxiter=2000, gtol=1e-14, ftol=1e-16))
        Z = res.x.reshape(-1, 3)
        S[idx] = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        rings += 1
    if (triple(S, F) <= 0).any():
        raise RuntimeError("untangling left flipped triangles")
    return S
