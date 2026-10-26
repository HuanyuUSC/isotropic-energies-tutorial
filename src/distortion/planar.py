"""2D shape deformation: a triangle mesh, distortion energies, and three solvers.

    minimize_x  sum_t A_t psi(F_t(x)) + k/2 sum_h |x_h - target_h|^2,   pinned vertices fixed,

with F_t = D_s(x) D_m^{-1} per triangle.  Solvers: gradient descent, Newton with the true
(indefinite) Hessian, and projected Newton (analytic eigensystem, clamped).  Barrier energies
get a flip-free max step (det F_t(x + a d) > 0 is a quadratic in a).
"""
import numpy as np

from .core import Elastic, LinearSolver, Problem, SparsePattern, Springs, _smallest_positive_root, minimize


# ---------------------------------------------------------------- the gingerbread man
def gingerbread(max_area=0.004):
    """Procedural cookie-shaped triangle mesh. Returns V (n,2), F (t,3), and named vertex groups."""
    import triangle
    from shapely.geometry import LineString, Point
    from shapely.ops import unary_union
    cap = lambda a, b, r: LineString([a, b]).buffer(r, quad_segs=16)
    shape = unary_union([
        Point(0, 1.62).buffer(0.55, quad_segs=24),                      # head
        cap((0, 0.95), (0, -0.25), 0.62),                               # body
        cap((-0.4, 0.75), (-1.45, 0.55), 0.26), cap((0.4, 0.75), (1.45, 0.55), 0.26),   # arms
        cap((-0.3, -0.45), (-0.55, -1.7), 0.3), cap((0.3, -0.45), (0.55, -1.7), 0.3),    # legs
    ]).simplify(0.01)
    B = np.asarray(shape.exterior.coords)[:-1]
    # resample the outline evenly
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.vstack([B, B[:1]]), axis=0), axis=1))]
    t = np.linspace(0, L[-1], int(L[-1] / np.sqrt(2.2 * max_area)), endpoint=False)
    B = np.c_[np.interp(t, L, np.r_[B[:, 0], B[0, 0]]), np.interp(t, L, np.r_[B[:, 1], B[0, 1]])]
    seg = np.c_[np.arange(len(B)), (np.arange(len(B)) + 1) % len(B)]
    m = triangle.triangulate(dict(vertices=B, segments=seg), f"pq30a{max_area}")
    V, F = m["vertices"], m["triangles"]
    groups = dict(
        feet=np.nonzero(V[:, 1] < -1.85)[0],
        right_hand=np.nonzero(np.linalg.norm(V - [1.45, 0.55], axis=1) < 0.2)[0],
        left_hand=np.nonzero(np.linalg.norm(V - [-1.45, 0.55], axis=1) < 0.2)[0],
        head=np.nonzero(np.linalg.norm(V - [0, 1.75], axis=1) < 0.2)[0],
    )
    return V, F, groups


# ---------------------------------------------------------------- energy on a mesh
class LinearElasticLike:
    """psi = |F - I|^2: quadratic, *not* rotation invariant (the naive first try). An energy of F itself,
    so it provides density(F) and hessF(F) instead of a function of the singular values."""
    name = "|F − I|²"
    barrier = False

    def density(self, F):
        D = F - np.eye(F.shape[-1])
        return np.sum(D ** 2, axis=(1, 2)), 2 * D

    def hessF(self, F):
        d2 = F.shape[-1] ** 2
        return np.broadcast_to(2 * np.eye(d2), (len(F), d2, d2))


class MeshEnergy(Elastic):
    """sum_t A_t psi(F_t) on a 2D triangle mesh: core.Elastic, with the older planar interface
    (grad returns (f, g as (n, 2), F); hessian(F, mode) returns the assembled sparse matrix)."""

    def __init__(self, V, F, energy, Dm=None):
        super().__init__(F, energy, V=V, Dm=Dm)
        self.F, self.n, self.A = F, (len(V) if V is not None else int(F.max()) + 1), self.vol
        self._pattern = SparsePattern([self.dof], 2 * self.n, diagonal=False)

    def grad(self, X):
        f, gt, _ = self.terms(X, None)
        g = np.bincount(self.dof.ravel(), weights=gt.ravel(), minlength=2 * self.n)
        self._X = X
        return f, g.reshape(-1, 2), self.deformation(X)

    def hessian(self, Fd, mode="clamp"):
        _, _, Ht = self.terms(self._X, mode)       # Fd is the deformation of the last grad(X) call
        return self._pattern.matrix([Ht])


def dets(X, F):
    e1, e2 = X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]
    return e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]


def max_step(X, D, F):
    """Largest a with det(e1 + a d1, e2 + a d2) > 0 for all triangles (a quadratic in a)."""
    e1, e2 = X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]
    d1, d2 = D[F[:, 1]] - D[F[:, 0]], D[F[:, 2]] - D[F[:, 0]]
    cr = lambda a, b: a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return _smallest_positive_root([cr(e1, e2), cr(e1, d2) + cr(d1, e2), cr(d1, d2)])


# ---------------------------------------------------------------- the drag problem
class DragProblem:
    """Pinned vertices fixed; handle vertices pulled toward targets by stiff springs:
    core.Problem([Elastic, Springs]) solved by core.minimize."""

    def __init__(self, V, F, energy, pinned, handles, k=1e3, extra=()):
        self.V, self.F, self.E = V, F, MeshEnergy(V, F, energy)
        self.pinned, self.handles, self.k = np.asarray(pinned), np.asarray(handles), k
        self.springs = Springs(self.handles, V[self.handles].copy(), k)
        pin = np.zeros(len(V), bool)
        pin[self.pinned] = True
        self.P = Problem(len(V), 2, [self.E, self.springs, *extra], pinned=pin,
                         solver=LinearSolver(backend="superlu" if len(V) < 3000 else "auto"))
        self.free_dof = self.P.free

    @property
    def target(self):
        return self.springs.target

    @target.setter
    def target(self, t):
        self.springs.target = np.asarray(t, float)

    def value(self, X):
        return self.P.value(X)

    def grad_hess(self, X, mode):
        f, g, H = self.P.evaluate(X, None if mode is None else mode)
        return f, g.reshape(-1, 2), H

    def step(self, X, method="pnewton", verbose=False):
        """One iteration. method: 'gd', 'newton' (true Hessian), 'pnewton' (clamped)."""
        hist = []
        Xn = minimize(self.P, X, method, iters=1, tol_rel_f=None, keep_going_when_capped=False, history=hist)
        return Xn, hist[0]

    def solve(self, X, method="pnewton", iters=100, tol=1e-9, history=None):
        return minimize(self.P, X, method, iters=iters, tol_rel_f=tol, keep_going_when_capped=False,
                        history=history)
