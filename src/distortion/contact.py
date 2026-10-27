"""Several elastic solids with contact: the 3D demos for the deformation and dynamics cells.

Every step is one minimization, as in the jelly (solid.SolidJelly):

    Y = argmin  1/(2h^2) |Y - X~|_M^2  +  sum_b E_b(Y)  +  planes(Y)  +  kappa * IPC(Y)

* E_b: one isotropic energy per body (tetrahedra, the same TetEnergy and analytic eigensystem as before).
* planes: rigid half-spaces (floor, squeeze plates) with the same log barrier as the jelly's floor.
* IPC: the barrier between surfaces (body-body and self contact) from ipctk [Li et al. 2020].
  The line search never steps past the first collision (continuous collision detection), never
  through a plane, and never inverts a tet (E = inf), so every frame is intersection- and inversion-free.
* Quasistatic mode (deformation cell): no gravity and a tiny inertia weight that only removes the
  null space of a frictionless body between plates.

    Scene(bodies, planes, pinned).step(X, Vel)        one implicit Euler step
    Scene(..., h=None).solve(X)                       one quasistatic solve
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .core import Elastic, Inertia, LinearSolver, Problem, SparsePattern, minimize, superlu_spd  # noqa: F401
from .solid import TetEnergy  # noqa: F401


def box_tets(size, n, origin=(0.0, 0.0, 0.0)):
    """A box [origin, origin + size] cut into n = (nx, ny, nz) cubes, each split into 6 tets (Kuhn)."""
    nx, ny, nz = n
    g = [np.linspace(0, s, k + 1) for s, k in zip(size, n)]
    I, J, K = np.meshgrid(*[np.arange(k + 1) for k in n], indexing="ij")
    V = np.c_[g[0][I.ravel()], g[1][J.ravel()], g[2][K.ravel()]] + origin
    vid = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k
    paths = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    T = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for p in paths:                      # walk from corner 000 to 111 along the 3 axes
                    c = [0, 0, 0]
                    tet = [vid(i, j, k)]
                    for ax in p:
                        c[ax] += 1
                        tet.append(vid(i + c[0], j + c[1], k + c[2]))
                    T.append(tet)
    T = np.array(T)
    D = np.stack([V[T[:, a]] - V[T[:, 0]] for a in (1, 2, 3)], 2)
    neg = np.linalg.det(D) < 0
    T[neg] = T[neg][:, [0, 2, 1, 3]]
    return V, T


def surface(T):
    """Boundary of a simplex mesh: outward triangles of a tet mesh, or the boundary edges of a triangle
    mesh (2D), with the owning element of each."""
    if T.shape[1] == 3:
        e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
        owner = np.tile(np.arange(len(T)), 3)
        _, idx, cnt = np.unique(np.sort(e, 1), axis=0, return_index=True, return_counts=True)
        return e[idx[cnt == 1]], owner[idx[cnt == 1]]
    tri = np.concatenate([T[:, [0, 2, 1]], T[:, [0, 1, 3]], T[:, [1, 2, 3]], T[:, [0, 3, 2]]])
    owner = np.tile(np.arange(len(T)), 4)
    _, idx, cnt = np.unique(np.sort(tri, 1), axis=0, return_index=True, return_counts=True)
    return tri[idx[cnt == 1]], owner[idx[cnt == 1]]


class Plane:
    """The half-space n . x >= offset(t), with the jelly's log barrier (zero beyond distance dhat)."""

    def __init__(self, normal, offset, dhat=0.02, kappa=2e3, mu=0.0, eps_v=1e-2):
        self.n = np.asarray(normal, float) / np.linalg.norm(normal)
        self.offset = offset if callable(offset) else (lambda t, o=offset: o)
        self.dhat, self.kappa = dhat, kappa
        self.mu, self.eps_v = mu, eps_v                 # Coulomb friction (IPC's lagged, smoothed form)
        self.lam = None

    # ---- friction: D(Y) = sum_i mu lam_i f0(|P (Y_i - X_i)|), lam_i = the normal force at the step's start
    def lag(self, X, t, h):
        """Freeze the normal forces and the start positions for one time step (IPC's lagged friction)."""
        if self.mu <= 0:
            return
        _, g1, _ = self.energy(X, t, True)
        self.lam, self.Xs, self.eps = np.abs(g1), X.copy(), self.eps_v * h

    def friction(self, Y, deriv=False):
        n = len(Y)
        if self.mu <= 0 or self.lam is None or not self.lam.any():
            return (0.0, np.zeros(Y.size), None) if deriv else 0.0
        u = Y - self.Xs
        u = u - (u @ self.n)[:, None] * self.n                # tangential displacement over the step
        y = np.linalg.norm(u, axis=1)
        e = self.eps
        f0 = np.where(y < e, -y ** 3 / (3 * e * e) + y * y / e + e / 3, y)    # C1 smoothing of |u| near 0
        w = self.mu * self.lam
        if not deriv:
            return float(w @ f0)
        f1_y = np.where(y < e, -y / (e * e) + 2 / e, 1 / np.maximum(y, 1e-300))   # f0'(y) / y > 0
        g = (w * f1_y)[:, None] * u
        # PSD (lagged-IPC) Hessian: w f0'(y)/y on the tangent plane, per active vertex
        on = np.nonzero(w > 0)[0]
        d = Y.shape[1]
        P = np.eye(d) - np.outer(self.n, self.n)
        rows, cols, vals = [], [], []
        for a in range(d):
            for c in range(d):
                if P[a, c] != 0:
                    rows.append(d * on + a); cols.append(d * on + c); vals.append(w[on] * f1_y[on] * P[a, c])
        H = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(d * n,) * 2)
        return float(w @ f0), g.ravel(), H

    def dist(self, Y, t):
        return Y @ self.n - self.offset(t)

    def energy(self, Y, t, deriv=False):
        d = self.dist(Y, t)
        on = d < self.dhat
        dd = np.where(on, np.maximum(d, 1e-300), self.dhat)
        with np.errstate(divide="ignore", invalid="ignore"):
            L = np.log(dd / self.dhat)
        b = np.where(on, -(dd - self.dhat) ** 2 * L, 0)
        if not deriv:
            return self.kappa * b.sum()
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            g1 = np.where(on, -2 * (dd - self.dhat) * L - (dd - self.dhat) ** 2 / dd, 0)
            g2 = np.where(on, -2 * L - 4 * (dd - self.dhat) / dd + (dd - self.dhat) ** 2 / dd ** 2, 0)
        return self.kappa * b.sum(), self.kappa * g1, self.kappa * np.maximum(g2, 0)

    def max_step(self, Y, D, t):
        """Largest alpha <= 1 that keeps every vertex at least 10% of its distance inside."""
        d, v = self.dist(Y, t), D @ self.n
        m = v < 0
        return 1.0 if not m.any() else min(1.0, 0.9 * np.min(d[m] / -v[m]))

    def term(self, time=lambda: 0.0):
        """This plane (barrier + friction) as a core.Problem term; time() gives the current t."""
        return PlaneTerm(self, time)


class PlaneTerm:
    def __init__(self, plane, time):
        self.p, self.time = plane, time

    def admissible(self, Y):
        return not (self.p.dist(Y, self.time()) <= 0).any()

    def value(self, Y):
        return self.p.energy(Y, self.time()) + self.p.friction(Y)

    def terms(self, Y, n, d):
        p, t = self.p, self.time()
        b, g1, g2 = p.energy(Y, t, True)
        g = (g1[:, None] * p.n).ravel()
        on = np.nonzero(g2)[0]
        nn = np.outer(p.n, p.n)                               # Hessian of b(n . x) = b'' n n^T per vertex
        r, c = np.nonzero(nn)
        rows = (d * on[:, None] + r).ravel()
        cols = (d * on[:, None] + c).ravel()
        H = sp.csc_matrix(((g2[on][:, None] * nn[r, c]).ravel(), (rows, cols)), shape=(d * n,) * 2)
        if p.mu > 0:
            ff, gf, Hf = p.friction(Y, True)
            if Hf is not None:
                b += ff
                g = g + gf
                H = H + Hf
        return b, g, None, H

    def max_step(self, Y, D):
        return self.p.max_step(Y, D, self.time())


class IPCTerm:
    """kappa * IPC barrier between all surfaces (body-body and self contact), from ipctk [Li et al. 2020];
    its safe step is ipctk's continuous collision detection.
    ccd="additive": Additive CCD [Li et al. 2021], conservative advancement per candidate pair; on the letters demo
    it is 17x faster than Tight Inclusion and gives the same steps (mean 1.06x). ccd="tight": Tight Inclusion."""

    def __init__(self, cmesh, dhat, kappa, ccd="additive"):
        import ipctk
        self.ipc, self.cmesh, self.dhat = ipctk, cmesh, dhat
        self.barrier = ipctk.BarrierPotential(dhat, kappa)
        self.broad = ipctk.SweepAndPrune() if hasattr(ipctk, "SweepAndPrune") else None
        self.ccd = {"additive": ipctk.AdditiveCCD, "tight": ipctk.TightInclusionCCD}[ccd]()
        self.ncol = 0
        self._cache = (None, None)          # (positions, collision set) from the last terms() call

    def _col(self, C):
        col = self.ipc.NormalCollisions()
        col.build(self.cmesh, C, self.dhat)
        return col

    def value(self, Y):
        C = self.cmesh.vertices(Y)
        return self.barrier(self._col(C), self.cmesh, C)

    def terms(self, Y, n, d):
        C = self.cmesh.vertices(Y)
        col = self._col(C)
        self._cache = (C.copy(), col)       # max_step is called next at the same positions: reuse the set
        self.ncol = len(col)
        b = self.barrier(col, self.cmesh, C)
        g = self.cmesh.to_full_dof(self.barrier.gradient(col, self.cmesh, C))
        H = self.cmesh.to_full_dof(self.barrier.hessian(col, self.cmesh, C,
                                                        project_hessian_to_psd=self.ipc.PSDProjectionMethod.CLAMP))
        return b, np.asarray(g).ravel(), None, H

    def max_step(self, Y, D):
        """Conservative advancement first: two surface points approach by at most 2 max|D| per unit step, so a
        full step is safe if that is below the current minimum surface distance; otherwise run the CCD on the
        swept broad phase's candidates."""
        C0 = self.cmesh.vertices(Y)
        m = np.sqrt((self.cmesh.vertices(D) ** 2).sum(1).max())
        Cc, col = self._cache
        if Cc is None or Cc.shape != C0.shape or not np.array_equal(Cc, C0):
            col = self._col(C0)
        dmin = np.sqrt(col.compute_minimum_distance(self.cmesh, C0)) if len(col) else self.dhat
        if 2 * m < 0.9 * dmin:
            return 1.0
        C1 = self.cmesh.vertices(Y + D)
        C0, C1 = np.asfortranarray(C0), np.asfortranarray(C1)
        cand = self.ipc.Candidates()
        cand.build(self.cmesh, C0, C1, 0.0, self.broad)
        return cand.compute_collision_free_stepsize(self.cmesh, C0, C1, 0.0, self.ccd)


GrowingPatternSolver = LinearSolver          # the old name


class Scene:
    """bodies: list of (V, T, energy, density); planes: list of Plane; pinned: bool mask over all vertices
    (pinned vertices follow `pin_path(t, X0)` if given, else stay put). ccd: "additive" or "tight" (see IPCTerm).
    A core.Problem: one Elastic term per body + Inertia + planes + IPC contact, solved by core.minimize."""

    def __init__(self, bodies, planes=(), pinned=None, pin_path=None, h=1 / 120, gravity=(0, -9.8, 0),
                 ipc=True, dhat=5e-3, kappa=100.0, quasistatic_weight=1e-3, solver="auto", pattern_margin=0.005,
                 ccd="additive"):
        self.problem = None
        self.parts, Vs, off, m = [], [], 0, []
        for V, T, energy, rho in bodies:
            E = Elastic(T, energy, V=V, offset=off)                  # triangles in 2D, tets in 3D
            self.parts.append((E, off, len(V), T + off))
            mb = np.zeros(len(V))
            k = T.shape[1]
            np.add.at(mb, T.ravel(), np.repeat(rho * E.vol / k, k))
            m.append(mb)
            Vs.append(V)
            off += len(V)
        self.X0 = np.vstack(Vs)
        self.n = off
        self.d = d = self.X0.shape[1]
        if len(gravity) != d:                                    # the default (0, -9.8, 0) in 2D: (0, -9.8)
            gravity = tuple(gravity)[:d]
        self.m = np.concatenate(m)
        self.T = np.vstack([p[3] for p in self.parts])
        self.planes = list(planes)
        self.problem = None
        self.pinned = np.zeros(self.n, bool) if pinned is None else np.asarray(pinned)
        self.pin_path = pin_path
        self.h, self.g = h, np.asarray(gravity, float)
        self.qw = quasistatic_weight
        self.pattern_margin = pattern_margin
        self.t = 0.0
        self.ipc = None
        self.inertia = Inertia(self.m, 1.0, self.X0)
        terms = [self.inertia] + [p.term(lambda: self.t) for p in self.planes]      # + the elastic parts, below
        if ipc:
            import ipctk
            F, _ = surface(self.T)
            if d == 3:
                E = ipctk.edges(F.astype(np.int32))
                self.cmesh = ipctk.CollisionMesh.build_from_full_mesh(self.X0, E, F.astype(np.int32))
            else:                                                # 2D: the boundary edges collide
                self.cmesh = ipctk.CollisionMesh.build_from_full_mesh(self.X0, F.astype(np.int32))
            self.ipc = ipctk
            self.dhat, self.kappa = dhat, kappa
            self.contact_term = IPCTerm(self.cmesh, dhat, kappa, ccd)
            terms.append(self.contact_term)
        self.stats = []
        self.solver = LinearSolver(seed=self._seed_keys if self.ipc is not None else None,
                                   backend="superlu" if solver == "superlu" else "auto")
        self._extra = terms
        self._rebuild()
        self._Y = None

    def _rebuild(self):
        """(Re)assemble the problem from the elastic parts, the extra terms and the pins."""
        terms = [E for E, _, _, _ in self._parts] + self._extra
        self.problem = Problem(self.n, self.d, terms, pinned=self._pinned, solver=self.solver)

    @property
    def parts(self):
        return self._parts

    @parts.setter
    def parts(self, parts):
        """Material groups may be swapped after construction (e.g. a softer joint region): keep the problem in sync."""
        self._parts = list(parts)
        if getattr(self, "problem", None) is not None:
            self._rebuild()

    @property
    def pinned(self):
        return self._pinned

    @pinned.setter
    def pinned(self, mask):
        """Pins can change during a run (e.g. letters released one batch at a time): keep the problem in sync."""
        self._pinned = np.asarray(mask, bool)
        self.free = np.repeat(~self._pinned, self.d)
        if getattr(self, "problem", None) is not None:
            self._rebuild()

    def _seed_keys(self, n_free):
        """Upper-triangle couplings (free-DOF indices) of every surface vertex pair the broad phase finds
        within dhat + pattern_margin: contacts that may switch on in the next steps."""
        if self._Y is None:
            return np.zeros(0, np.int64)
        ipctk = self.ipc
        C = ipctk.Candidates()
        C.build(self.cmesh, self.cmesh.vertices(self._Y), self.dhat + self.pattern_margin)
        E, F = self.cmesh.edges, self.cmesh.faces
        ids = [c.vertex_ids(E, F) for lst in (C.vv_candidates, C.ev_candidates, C.ee_candidates, C.fv_candidates)
               for c in lst]
        if not ids:
            return np.zeros(0, np.int64)
        ids = np.array(ids)
        full = np.asarray(self.cmesh.to_full_vertex_id()).ravel()
        ids = np.where(ids >= 0, full[np.maximum(ids, 0)], -1)
        d = self.d
        fidx = np.full(d * self.n, -1)
        fidx[self.free] = np.arange(n_free)
        a, b = ids[:, :, None], ids[:, None, :]                      # all vertex pairs within a stencil
        a, b = np.broadcast_arrays(a, b)
        ok = (a >= 0) & (b >= 0)
        a, b = a[ok], b[ok]
        ra = fidx[(d * a[:, None] + np.arange(d))[:, :, None].repeat(d, 2)].ravel()
        rb = fidx[(d * b[:, None] + np.arange(d))[:, None, :].repeat(d, 1)].ravel()
        good = (ra >= 0) & (rb >= 0)
        r, c = np.minimum(ra[good], rb[good]), np.maximum(ra[good], rb[good])
        return np.unique(r.astype(np.int64) * n_free + c)

    def elastic(self, Y):
        return sum(E.value(Y) for E, _, _, _ in self.parts)

    def total(self, Y, Xt, w, t):
        self.t, self.inertia.Xt, self.inertia.w = t, Xt, w
        return self.problem.value(Y)

    # ---- one minimization
    def _minimize(self, X, Xt, w, t, iters=60, tol=None):
        Y = X.copy()
        if self.pin_path is not None:
            Y[self.pinned] = self.pin_path(t, self.X0)[self.pinned]
        self.t, self.inertia.Xt, self.inertia.w = t, Xt, w
        # stop when the Newton step is invisible: 1e-4 of the scene size (in dynamics also a velocity error
        # below 1e-4 * size / h). In contact the last iterations only shuffle the active set.
        size = getattr(self, "size", None) or np.ptp(self.X0, axis=0).max()   # set S.size if bodies start parked far away
        tol_abs = tol if tol is not None else getattr(self, "tol_abs", None) or 1e-4 * size
        self._Y = Y
        Y = minimize(self.problem, Y, "pnewton", iters=iters, tol_dir=tol_abs, tol_step=0.1 * tol_abs,
                     tol_rel_f=getattr(self, "rel_energy_tol", None), keep_going_when_capped=True,
                     callback=lambda it, Yi, info: setattr(self, "_Y", Yi))
        self.stats.append((t, self.problem.iterations))
        return Y

    def step(self, X, Vel):
        """One implicit Euler step (dynamics)."""
        h = self.h
        for p in self.planes:
            p.lag(X, self.t, h)
        self.t += h
        Xt = X + h * Vel + h * h * self.g
        Y = self._minimize(X, Xt, 1 / h ** 2, self.t)
        return Y, (Y - X) / h

    def solve(self, X, t, accel=None):
        """Quasistatic: minimize the elastic + contact energy at time t (plates/pins at their t positions),
        optionally under a body force m * accel (gravity ramp). The tiny inertia weight w only removes
        rigid null spaces: Xt = X + accel / w turns w/2 |Y - Xt|_M^2 into that weight's potential."""
        self.t = t
        Xt = X if accel is None else X + np.asarray(accel, float) / self.qw
        return self._minimize(X, Xt, self.qw, t)
