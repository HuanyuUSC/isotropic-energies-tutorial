"""Regression + timing for the six demos (small versions), to compare the solver core before/after refactors.
Each case prints energy / iterations / flips / time and the whole run is saved as JSON.

    python tests/regression.py out.json [case ...]
cases: drag2d flatten_arap flatten_newton_gd sphere scene2d cantilever giraffe
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "figures")]
ASSETS = os.path.join(ROOT, "..", "assets")
out = {}


def case(fn):
    name = fn.__name__
    if len(sys.argv) > 2 and name not in sys.argv[2:]:
        return fn
    t0 = time.perf_counter()
    r = fn()
    r["seconds"] = round(time.perf_counter() - t0, 3)
    out[name] = r
    print(name, r, flush=True)
    return fn


@case
def drag2d():
    from distortion.eigensystem import SymmetricDirichlet
    from distortion.planar import DragProblem, dets, gingerbread
    V, F, G = gingerbread(0.004)
    P = DragProblem(V, F, SymmetricDirichlet(), G["feet"], G["right_hand"])
    P.target = P.target + [0.9, 1.2]
    hist = []
    X = P.solve(V.copy(), "pnewton", iters=100, history=hist)
    return dict(E=float(P.value(X)), iters=len(hist), flips=int((dets(X, F) <= 0).sum()))


def _face():
    import pickle
    d = pickle.load(open(os.path.join(ROOT, "scratch", "flatten_face_arap.pkl"), "rb"))
    return d["V3"], d["F"], d["X0"]


@case
def flatten_arap():
    from distortion.atlas import local_energy
    from distortion.eigensystem import ARAP
    from distortion.planar import dets
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    V3, F, X = _face()
    E = local_energy(V3, F, ARAP())
    A = E.A.sum()
    it = 0
    for it in range(200):
        f, g, Fd = E.grad(X)
        H = E.hessian(Fd, "clamp")
        H = H + 1e-8 * abs(H.diagonal()).max() * sp.eye(H.shape[0])
        d = spla.spsolve(H.tocsc(), -g.ravel()).reshape(-1, 2)
        slope = float(np.sum(g * d))
        a = 1.0
        while E.value(X + a * d) > f + 1e-4 * a * slope and a > 1e-14:
            a *= 0.5
        Xn = X + a * d
        fn = E.value(Xn)
        X = Xn
        if abs(f - fn) < 1e-10 * f:
            break
    return dict(E_per_area=float(E.value(X) / A), iters=it + 1, flips=int((dets(X, F) <= 0).sum()))


@case
def sphere():
    from distortion.sphere_param import parameterize, triple
    d = np.load(os.path.join(ASSETS, "viewer", "cache100", "bearing.npz"))
    V, F = d["V"], d["F"]
    S, hist = parameterize(V, F, verbose=False, iters=200)
    return dict(E=float(hist[-1][1]), iters=len(hist) - 1, flips=int((triple(S, F) <= 0).sum()))


@case
def scene2d():
    from distortion.contact import Plane, Scene
    from distortion.eigensystem import NeoHookean
    from distortion.shapes2d import letter
    bodies = []
    for k, ch in enumerate("UCLA"):
        V, F = letter(ch, 0.02)
        bodies.append((V + [1.3 * k - 2, 1.5 + 0.4 * k], F, NeoHookean(mu=40.0, lam=160.0), 1.0))
    planes = [Plane((0, 1), 0.0, dhat=0.02, kappa=2e3, mu=0.4), Plane((1, 0), -3.0, dhat=0.02, kappa=2e3),
              Plane((-1, 0), -3.0, dhat=0.02, kappa=2e3)]
    S = Scene(bodies, planes=planes, h=1 / 120, dhat=1e-2, kappa=1e3)
    X, Vel = S.X0.copy(), np.zeros_like(S.X0)
    for _ in range(90):
        X, Vel = S.step(X, Vel)
    it = np.array([s[1] for s in S.stats])
    return dict(lowest=float(X[:, 1].min()), com=X.mean(0).round(5).tolist(), newton=int(it.sum()), steps=len(it))


@case
def cantilever():
    from distortion.contact import Plane, Scene, box_tets
    from distortion.eigensystem import NeoHookean
    VB, TB = box_tets((1.6, 0.2, 0.5), (16, 2, 5), origin=(0.0, 1.0, -0.25))
    VA, TA = box_tets((0.4, 0.4, 0.4), (4, 4, 4), origin=(0.9, 1.6, -0.2))
    pinned = np.r_[VB[:, 0] < 1e-9, np.zeros(len(VA), bool)]
    S = Scene([(VB, TB, NeoHookean(mu=7.7e3, lam=1.15e4), 1.0), (VA, TA, NeoHookean(mu=30.0, lam=120.0), 1.0)],
              planes=[Plane((0, 1, 0), 0.0, dhat=0.02, kappa=2e3, mu=0.5)], pinned=pinned, h=1 / 120, dhat=4e-3,
              kappa=1e5, solver="pardiso")
    X, Vel = S.X0.copy(), np.zeros_like(S.X0)
    for _ in range(60):
        X, Vel = S.step(X, Vel)
    it = np.array([s[1] for s in S.stats])
    nB = len(VB)
    return dict(tip=float(X[:nB, 1].min()), box_low=float(X[nB:, 1].min()), newton=int(it.sum()), steps=len(it))


@case
def giraffe():
    from distortion.contact import Scene
    from distortion.eigensystem import NeoHookean
    D = np.load(os.path.join(ROOT, "scratch", "demo3d", "giraffe_tet.npz"))
    V, T = D["V"].astype(float), D["T"].astype(np.int64)
    V = (V - V.min(0)) * (2.0 / np.ptp(V[:, 1])); V[:, [0, 2]] -= V[:, [0, 2]].mean(0)
    H = np.ptp(V[:, 1])
    feet = V[:, 1] < 0.36 * H
    S = Scene([(V, T, NeoHookean(mu=1 / 2.6, lam=0.3 / (1.3 * 0.4)), 1.0)], pinned=feet, h=None,
              gravity=(0, 0, 0), ipc=False)
    X = V.copy()
    for k in range(1, 5):                     # sag under a ramped body force (quasistatic)
        X = S.solve(X, 0.0, accel=(0.0, -0.05 * k, 0.03 * k))
    it = np.array([s[1] for s in S.stats])
    return dict(neck=X[(V[:, 1] > 0.75 * H) & (V[:, 1] < 0.8 * H)].mean(0).round(5).tolist(), newton=int(it.sum()))


json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "latest.json"), "w"), indent=1)
