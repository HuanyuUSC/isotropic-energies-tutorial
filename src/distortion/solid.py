"""3D solids: the planar machinery with d = 3. A tetrahedral mesh, F_t = D_s D_m^{-1} (3 x 3), the same
isotropic energies, the same analytic eigensystem (any d), and projected Newton.

    extruded_gingerbread()   the 2D gingerbread, extruded into a thick cookie of tetrahedra
    TetEnergy                sum_t vol_t psi(F_t), its gradient and projected Hessian
    SolidDrag                pinned vertices + handles pulled by stiff springs, as planar.DragProblem
    SolidJelly               implicit Euler = minimization (inertia + elastic energy + floor barrier)
"""
import numpy as np
from .planar import gingerbread


def extruded_gingerbread(max_area=0.01, thickness=0.5, layers=3):
    """Prisms over the 2D triangulation, each split into 3 tets by the lowest-index rule (conforming)."""
    V2, F2, G = gingerbread(max_area)
    n = len(V2)
    z = np.linspace(-thickness / 2, thickness / 2, layers + 1)
    V = np.vstack([np.c_[V2, np.full(n, zk)] for zk in z])
    T = []
    for k in range(layers):
        for a, b, c in F2:
            bot = [a + k * n, b + k * n, c + k * n]
            top = [i + n for i in bot]
            T += _split_prism(bot, top)
    T = np.array(T)
    # positive orientation
    D = np.stack([V[T[:, i]] - V[T[:, 0]] for i in (1, 2, 3)], 2)
    neg = np.linalg.det(D) < 0
    T[neg] = T[neg][:, [0, 2, 1, 3]]
    groups = {g: np.concatenate([idx + k * n for k in range(layers + 1)]) for g, idx in G.items()}
    return V, T, groups


def _split_prism(bot, top):
    """Split prism (bot a b c, top a' b' c') into 3 tets; each quad face is cut through its
    lowest-index vertex, so neighbouring prisms agree."""
    # rotate so the global minimum vertex is bot[0] or top[0]
    verts = bot + top
    m = int(np.argmin(verts))
    r = m % 3
    bot, top = bot[r:] + bot[:r], top[r:] + top[:r]
    if m >= 3:                                      # minimum on top: flip the prism
        bot, top = top, bot
    a, b, c = bot
    A, B, C = top
    # quad (b, c, C, B): diagonal through its lowest vertex
    if min(b, C) < min(c, B):
        return [[a, b, c, C], [a, b, C, B], [a, B, C, A]]
    return [[a, b, c, B], [a, B, c, C], [a, B, C, A]]


from .core import THREADS, Elastic, Inertia, Problem, SparsePattern, Springs, minimize, parallel_chunks  # noqa: F401


class TetEnergy(Elastic):
    """core.Elastic built from rest positions (tetrahedra in 3D, triangles in 2D), with the older interface:
    element_terms(X) -> (f, assembled gradient, per-element Hessians)."""

    def __init__(self, V, T, energy, projection="clamp"):
        super().__init__(T, energy, V=V, projection=projection)
        self.n = len(V)

    def element_terms(self, X):
        f, gt, Ht = self.terms(X, "default")
        g = np.bincount(self.dof.ravel(), weights=gt.ravel(), minlength=self.d * self.n)
        return f, g, Ht

    def grad_hess(self, X):
        f, g, Ht = self.element_terms(X)
        if not hasattr(self, "_pattern"):
            self._pattern = SparsePattern([self.dof], self.d * self.n)
        return f, g, self._pattern.matrix([Ht])


class SolidDrag:
    """Pinned vertices + handles pulled by stiff springs: core.Problem([Elastic, Springs])."""

    def __init__(self, V, T, energy, pinned, handles, k=1e3):
        self.V, self.T, self.k = V, T, k
        self.E = TetEnergy(V, T, energy)
        self.springs = Springs(handles, V[np.asarray(handles)].copy(), k)
        pin = np.zeros(len(V), bool)
        pin[pinned] = True
        self.P = Problem(len(V), 3, [self.E, self.springs], pinned=pin)
        self.n = len(V)

    @property
    def target(self):
        return self.springs.target

    @target.setter
    def target(self, t):
        self.springs.target = np.asarray(t, float)

    def dets(self, X):
        return self.E.dets(X)

    def value(self, X):
        return self.P.value(X)

    def solve(self, X, iters=60, tol=1e-9):
        return minimize(self.P, X, "pnewton", iters=iters, tol_rel_f=tol, keep_going_when_capped=False)


class SolidJelly:
    """x_{n+1} = argmin_x 1/(2h^2) |x - x~|_M^2 + E(x) + B(x),  x~ = x_n + h v_n + h^2 g:
    core.Problem([Elastic, Inertia, floor plane])."""

    def __init__(self, V, T, energy, h=1 / 120, floor=-3.0, dhat=0.06, kappa=2e3, g=(0.0, -9.8, 0.0)):
        from .contact import Plane
        self.E = TetEnergy(V, T, energy)
        m = np.zeros(len(V))
        np.add.at(m, T.ravel(), np.repeat(self.E.vol / 4, 4))
        self.m, self.h, self.g = m, h, np.asarray(g)
        self.floor = Plane((0, 1, 0), floor, dhat=dhat, kappa=kappa)
        self.inertia = Inertia(m, 1 / h ** 2, None)
        self.P = Problem(len(V), 3, [self.E, self.inertia, self.floor.term()])

    def step(self, X, Vel, iters=30):
        h = self.h
        self.inertia.Xt = X + h * Vel + h * h * self.g
        Y = minimize(self.P, X.copy(), "pnewton", iters=iters, tol_step=1e-7, tol_rel_f=None)
        return Y, (Y - X) / h
