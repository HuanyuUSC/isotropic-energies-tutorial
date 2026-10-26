"""Regression + timing for the six demos (small versions), checked against tests/baseline.json.
Each case prints energy / iterations / flips / time; the whole run is saved as JSON and compared with the baseline.
Every input is generated in code or downloaded on first use (the Stanford bunny), so this runs from a fresh clone.

    python tests/regression.py [out.json] [case ...]
cases: drag2d flatten_arap sphere scene2d cantilever drag3d
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path[:0] = [os.path.join(ROOT, "src")]
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


def _patch():
    """The largest disk-like chart of the bunny (as in demos/flatten.py), with its Tutte start."""
    from distortion.atlas import grow_charts, is_disk, tutte
    from distortion.meshes import genus0_mesh
    V, F = genus0_mesh("bunny", 6000)
    lab = grow_charts(V, F, 8)
    charts = [np.flatnonzero(lab == k) for k in np.unique(lab)]
    fs = max((c for c in charts if is_disk(np.unique(F[c], return_inverse=True)[1].reshape(-1, 3))), key=len)
    vs, Fl = np.unique(F[fs], return_inverse=True)
    Fl, V3 = Fl.reshape(-1, 3), V[vs]
    return V3, Fl, tutte(V3, Fl)


@case
def flatten_arap():
    from distortion.atlas import local_energy
    from distortion.eigensystem import ARAP
    from distortion.planar import dets
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    V3, F, X = _patch()
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
    from distortion.meshes import genus0_mesh
    V, F = genus0_mesh("bunny", 3000)
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
def drag3d():
    """The extruded cookie (demos/drag3d.py, coarser): feet and right hand pinned, the hand raised in 40 steps."""
    from distortion.contact import Scene
    from distortion.eigensystem import NeoHookean
    from distortion.solid import extruded_gingerbread
    V, T, G = extruded_gingerbread(0.02)
    hand, feet = G["right_hand"], G["feet"]
    pinned = np.zeros(len(V), bool)
    pinned[hand] = pinned[feet] = True

    def path(t, X0):
        X = X0.copy()
        X[hand] = X0[hand] + t * np.array([0.3, 1.2, 0.6])
        return X

    S = Scene([(V, T, NeoHookean(mu=1.0, lam=4.0), 1.0)], pinned=pinned, pin_path=path, h=None, gravity=(0, 0, 0),
              ipc=False)
    X = V.copy()
    for t in np.linspace(1 / 40, 1.0, 40):
        X = S.solve(X, t)
    it = np.array([s[1] for s in S.stats])
    return dict(com=X.mean(0).round(5).tolist(), min_det=float(S.parts[0][0].dets(X).min()), newton=int(it.sum()))


# ------------------------------------------------------------------------------------------- compare
def compare(new, base, rtol=1e-4):
    """Numbers within rtol of the baseline (integer counts within 10%); timings are reported, not checked."""
    bad = []
    for name, r in new.items():
        b = base.get(name)
        if b is None:
            print(f"  {name:13s} no baseline"); continue
        for k, v in r.items():
            if k == "seconds" or k not in b:
                continue
            x, y = np.asarray(v, float), np.asarray(b[k], float)
            tol = 0.1 * np.abs(y) + 1 if isinstance(v, int) else rtol * np.maximum(np.abs(y), 1e-3)
            if not np.all(np.abs(x - y) <= tol):
                bad.append(f"{name}.{k}: {v} vs {b[k]}")
        print(f"  {name:13s} {r['seconds']:7.2f} s   (baseline {b.get('seconds', float('nan')):.2f} s)")
    print("PASS" if not bad else "FAIL\n  " + "\n  ".join(bad))
    return not bad


dst = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "latest.json")
json.dump(out, open(dst, "w"), indent=1)
base = os.path.join(HERE, "baseline.json")
if os.path.exists(base) and os.path.abspath(dst) != os.path.abspath(base):
    sys.exit(0 if compare(out, json.load(open(base))) else 1)
