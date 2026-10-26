"""Global injectivity for 2D deformations: a boundary barrier in the spirit of Smith & Schaefer 2015
("Bijective parameterization with free boundaries").

Symmetric Dirichlet keeps every triangle un-inverted (local injectivity), but two distant parts of the
boundary can still pass through each other. Here every boundary vertex p and every boundary edge (a, b)
not containing it get a barrier on their distance d:

    b(d) = kappa * (eps / d - 1)^2   for d < eps,   0 otherwise         (infinite as d -> 0)

and every step is limited by a conservative continuous-collision bound, so no iterate can cross.
Gradients are exact (envelope theorem on the closest point); the Hessian is the Gauss-Newton part
b''(d) grad d grad d^T, which is PSD because b is convex in d.
"""
import numpy as np
import scipy.sparse as sp

from .planar import DragProblem


def boundary_edges(F):
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    E, c = np.unique(E, axis=0, return_counts=True)
    return E[c == 1]


class BoundaryBarrier:
    """A core.Problem term: barrier between distant boundary vertices and edges, with a conservative CCD step."""

    def __init__(self, V, F, eps=None, kappa=1e-2):
        BE = boundary_edges(F)
        bv = np.unique(BE)
        P, Ei = np.meshgrid(bv, np.arange(len(BE)), indexing="ij")
        P, Ei = P.ravel(), Ei.ravel()
        keep = (BE[Ei, 0] != P) & (BE[Ei, 1] != P)
        self.p, self.a, self.b = P[keep], BE[Ei[keep], 0], BE[Ei[keep], 1]
        L = np.linalg.norm(V[BE[:, 1]] - V[BE[:, 0]], axis=1)
        self.eps = eps if eps is not None else 0.5 * L.mean()
        self.kappa = kappa
        # skip pairs that are within eps at rest (neighbours along the boundary): only distant parts collide
        d0, _, _ = self._dist(V)
        far = d0 > 1.5 * self.eps
        self.p, self.a, self.b = self.p[far], self.a[far], self.b[far]

    def _dist(self, X):
        p, a, b = X[self.p], X[self.a], X[self.b]
        ab = b - a
        t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.einsum("ij,ij->i", ab, ab), 0, 1)
        r = p - (a + t[:, None] * ab)
        return np.linalg.norm(r, axis=1), r, t

    def value(self, X):
        d, _, _ = self._dist(X)
        d = d[d < self.eps]
        return self.kappa * np.sum((self.eps / d - 1) ** 2)

    def terms(self, X, n, dim):
        d, r, t = self._dist(X)
        on = d < self.eps
        g = np.zeros((n, 2))
        if not on.any():
            return 0.0, g.ravel(), None
        d, r, t = d[on], r[on], t[on]
        q = self.eps / d - 1
        f = self.kappa * np.sum(q ** 2)
        db = -2 * self.kappa * q * self.eps / d ** 2                              # b'(d)
        ddb = 2 * self.kappa * (self.eps ** 2 / d ** 4 + 2 * q * self.eps / d ** 3)   # b''(d) > 0
        nrm = r / d[:, None]
        G = np.stack([nrm, -(1 - t)[:, None] * nrm, -t[:, None] * nrm], 1)      # grad d wrt (p, a, b)
        idx = np.stack([self.p[on], self.a[on], self.b[on]], 1)
        np.add.at(g, idx.ravel(), (db[:, None, None] * G).reshape(-1, 2))
        dof = (2 * idx[:, :, None] + np.arange(2)).reshape(len(idx), 6)
        Gf = G.reshape(len(idx), 6)
        blocks = ddb[:, None, None] * Gf[:, :, None] * Gf[:, None, :]           # Gauss-Newton: PSD
        H = sp.csc_matrix((blocks.ravel(), (np.repeat(dof, 6, axis=1).ravel(), np.tile(dof, (1, 6)).ravel())),
                          shape=(2 * n, 2 * n))
        return f, g.ravel(), None, H

    def max_step(self, X, D):
        """Largest safe step: relative motion of p against segment ab can shrink d by at most
        |dp| + max(|da|, |db|) per unit step (conservative advancement)."""
        d, _, _ = self._dist(X)
        m = np.linalg.norm(D[self.p], axis=1) + np.maximum(np.linalg.norm(D[self.a], axis=1),
                                                          np.linalg.norm(D[self.b], axis=1))
        with np.errstate(divide="ignore"):
            a = np.where(m > 0, 0.9 * d / m, np.inf)
        return float(a.min())


class InjectiveDrag(DragProblem):
    """DragProblem + BoundaryBarrier: the same core solver, one more term."""

    def __init__(self, V, F, energy, pinned, handles, k=1e3, eps=None, kappa=1e-2):
        self.barrier_term = BoundaryBarrier(V, F, eps, kappa)
        super().__init__(V, F, energy, pinned, handles, k, extra=(self.barrier_term,))


def boundary_crossings(X, F):
    """Number of pairs of non-adjacent boundary edges that intersect (0 = globally injective)."""
    BE = boundary_edges(F)
    A, B = X[BE[:, 0]], X[BE[:, 1]]
    cr = lambda u, v: u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    n = 0
    for i in range(len(BE)):
        o = (BE[i + 1:, 0] != BE[i, 0]) & (BE[i + 1:, 0] != BE[i, 1]) & (BE[i + 1:, 1] != BE[i, 0]) & (BE[i + 1:, 1] != BE[i, 1])
        a, b, c, d = A[i], B[i], A[i + 1:][o], B[i + 1:][o]
        s1 = cr(b - a, c - a) * cr(b - a, d - a)
        s2 = cr(d - c, a - c) * cr(d - c, b - c)
        n += int(np.sum((s1 < 0) & (s2 < 0)))
    return n
