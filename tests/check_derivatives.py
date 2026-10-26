"""Finite-difference checks of every derivative the 3D demos use (gradient and unprojected Hessian).

    PYTHONPATH=src python tests/check_derivatives.py      (expect relative errors ~1e-10 .. 1e-7)
"""
import numpy as np
import scipy.sparse as sp
from distortion.contact import Plane, Scene, box_tets
from distortion.eigensystem import NeoHookean
from distortion.eigensystem import StVK, SymmetricDirichlet, WithBarrier, WithVolume
from distortion.solid import TetEnergy

rng = np.random.default_rng(0)
V, T = box_tets((1.0, 0.6, 0.4), (3, 2, 2))
X = V @ np.array([[1.3, 0.2, 0], [0, 0.8, 0.1], [0.1, 0, 1.1]]).T + 0.02 * rng.standard_normal(V.shape)   # large, non-inverted
assert (TetEnergy(V, T, NeoHookean()).dets(X) > 0).all()
h = 1e-6


def fd_grad(f, X):
    g = np.zeros(X.size)
    for i in range(X.size):
        e = np.zeros(X.size); e[i] = h
        g[i] = (f(X + e.reshape(X.shape)) - f(X - e.reshape(X.shape))) / (2 * h)
    return g


def fd_hess(gf, X):
    H = np.zeros((X.size, X.size))
    for i in range(X.size):
        e = np.zeros(X.size); e[i] = h
        H[:, i] = (gf(X + e.reshape(X.shape)) - gf(X - e.reshape(X.shape))) / (2 * h)
    return H


def rel(a, b):
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-12)


for name, psi in [("Neo-Hookean", NeoHookean(1.0, 2.0)), ("SymDir + volume", WithVolume(SymmetricDirichlet(), 1.0)),
                  ("StVK + barrier", WithBarrier(StVK()))]:
    E = TetEnergy(V, T, psi, projection="none")
    f, g, H = E.grad_hess(X)
    print(f"{name:14s} TetEnergy  grad {rel(g, fd_grad(E.value, X)):.1e}   hess(unprojected) "
          f"{rel(H.toarray(), fd_hess(lambda Y: E.grad_hess(Y)[1], X)):.1e}")

# plane barrier: put a few vertices inside the activation distance of the floor and of a tilted plane
S = Scene([(V, T, NeoHookean(1.0, 2.0), 1.0)],
          planes=[Plane((0, 1, 0), X[:, 1].min() - 0.01, dhat=0.05, kappa=3.0),
                  Plane((0.3, -1, 0.2), -(X @ np.array([0.3, -1, 0.2]) / np.linalg.norm([0.3, -1, 0.2])).max() + 0.02,
                        dhat=0.05, kappa=3.0)], ipc=False)
# every non-elastic term has the same interface: value(X), terms(X, n, d) -> (f, g, diag, H)
def term_grad(t, n, d):
    return lambda Y: t.terms(Y, n, d)[1]
PT = [t for t in S.problem.terms if type(t).__name__ == "PlaneTerm"]
fP = lambda Y: sum(t.value(Y) for t in PT)
gP = sum(term_grad(t, S.n, 3)(X) for t in PT)
HP = sum(t.terms(X, S.n, 3)[3] for t in PT)
print(f"planes         grad {rel(gP, fd_grad(fP, X)):.1e}   hess {rel(HP.toarray(), fd_hess(lambda Y: sum(term_grad(t, S.n, 3)(Y) for t in PT), X)):.1e}"
      f"   (active vertices: {sum((p.dist(X, 0) < p.dhat).sum() for p in S.planes)})")

# IPC barrier between two bodies brought within dhat (gradient through the full-mesh mapping)
VB, TB = box_tets((1.0, 0.2, 1.0), (2, 1, 2))
VA, TA = box_tets((0.4, 0.4, 0.4), (1, 1, 1), origin=(0.3, 0.2 + 0.004, 0.3))
S2 = Scene([(VB, TB, NeoHookean(1.0, 2.0), 1.0), (VA, TA, NeoHookean(1.0, 2.0), 1.0)], dhat=0.01, kappa=50.0)
Y = S2.X0 + 1e-3 * rng.standard_normal(S2.X0.shape)
C2 = S2.contact_term
gC = term_grad(C2, S2.n, 3)(Y)
print(f"IPC contact    grad {rel(gC, fd_grad(C2.value, Y)):.1e}   (collisions: {C2.ncol}; Hessian is PSD-projected by ipctk, not checked)")

# the whole dynamic objective of one implicit Euler step: the core Problem's gradient vs FD of its value
S3 = Scene([(VB, TB, NeoHookean(1.0, 2.0), 1.0), (VA, TA, WithVolume(SymmetricDirichlet(), 1.0), 1.0)],
           planes=[Plane((0, 1, 0), -0.003, dhat=0.01, kappa=5.0)], dhat=0.01, kappa=50.0, h=1 / 100)
Xt = Y + 0.01 * rng.standard_normal(Y.shape); w = 1 / S3.h ** 2
tot = lambda Z: S3.total(Z, Xt, w, 0.0)
tot(Y)                                            # sets the step's inertia target and weight
print(f"step objective grad {rel(S3.problem.evaluate(Y, None)[1], fd_grad(tot, Y)):.1e}")

# floor friction (lagged): gradient vs FD; the Hessian is the PSD lagged-IPC approximation (not checked)
Pf = Plane((0, 1, 0), X[:, 1].min() - 0.01, dhat=0.05, kappa=3.0, mu=0.5, eps_v=0.5)
Pf.lag(X, 0.0, 0.1)
Yf = X + 0.02 * rng.standard_normal(X.shape)
_, gF, _ = Pf.friction(Yf, True)
print(f"floor friction grad {rel(gF, fd_grad(Pf.friction, Yf)):.1e}   (sliding vertices: {(Pf.lam > 0).sum()})")

# ---- 2D: triangles, edge-edge IPC contact between two bodies, floor
from distortion.planar import gingerbread
V2, F2, _ = gingerbread(max_area=0.05)
X2 = V2 @ np.array([[1.1, 0.2], [-0.1, 0.9]]).T + 0.01 * rng.standard_normal(V2.shape)
E2 = TetEnergy(V2, F2, NeoHookean(1.0, 2.0), projection="none")
f, g, H = E2.grad_hess(X2)
print(f"Neo-Hookean 2D triangles  grad {rel(g, fd_grad(E2.value, X2)):.1e}   hess(unprojected) "
      f"{rel(H.toarray(), fd_hess(lambda Y: E2.grad_hess(Y)[1], X2)):.1e}")
th = np.linspace(0, 2 * np.pi, 13)[:-1]
VD = np.c_[np.r_[0, 0.3 * np.cos(th)], np.r_[0, 0.3 * np.sin(th)]] + [0.0, V2[:, 1].max() + 0.3 + 0.004]
FD = np.array([[0, 1 + i, 1 + (i + 1) % 12] for i in range(12)])
S4 = Scene([(V2, F2, NeoHookean(1.0, 2.0), 1.0), (VD, FD, NeoHookean(1.0, 2.0), 1.0)], dhat=0.05, kappa=50.0,
           planes=[Plane((0, 1), V2[:, 1].min() - 0.005, dhat=0.02, kappa=5.0)], h=1 / 100)
Y4 = S4.X0.copy()
C4 = S4.contact_term
print(f"2D IPC contact grad {rel(term_grad(C4, S4.n, 2)(Y4), fd_grad(C4.value, Y4)):.1e}   (collisions: {C4.ncol})")
Xt4 = Y4 + 0.01 * rng.standard_normal(Y4.shape); w4 = 1 / S4.h ** 2
print(f"2D step objective grad {rel((S4.total(Y4, Xt4, w4, 0.0), S4.problem.evaluate(Y4, None)[1])[1], fd_grad(lambda Z: S4.total(Z, Xt4, w4, 0.0), Y4)):.1e}")
