"""2D parameterization: cut a disk-shaped patch out of the Stanford bunny and flatten it
(Tutte start -> projected Newton on ARAP or Symmetric Dirichlet).  python demos/flatten.py [arap|symdir]"""
import sys

import numpy as np
from _common import OUT, timer

from distortion.atlas import grow_charts, is_disk, tutte
from distortion.core import Elastic, Problem, minimize, rest_frames
from distortion.eigensystem import ARAP, SymmetricDirichlet
from distortion.meshes import genus0_mesh
from distortion.planar import dets

V, F = genus0_mesh("bunny", 20000)
lab = grow_charts(V, F, 12)                             # a chart label per face; flatten the largest disk-like chart
charts = [np.flatnonzero(lab == k) for k in np.unique(lab)]
fs = max((c for c in charts if is_disk(np.unique(F[c], return_inverse=True)[1].reshape(-1, 3))), key=len)
vs, Fl = np.unique(F[fs], return_inverse=True)
Fl, V3 = Fl.reshape(-1, 3), V[vs]
energy = ARAP() if (sys.argv[1:] or ["arap"])[0] == "arap" else SymmetricDirichlet()

E = Elastic(Fl, energy, Dm=rest_frames(V3, Fl))      # each 3D triangle, in its own 2D frame, is the rest shape
P = Problem(len(V3), 2, [E], damping=1e-8)             # free boundary: a tiny Levenberg shift removes rigid motion
X0 = tutte(V3, Fl)
with timer(f"{len(Fl)} triangles, {energy.name}"):
    hist = []
    X = minimize(P, X0, "pnewton", tol_rel_f=1e-10, f_floor=0.0, keep_going_when_capped=False, history=hist)
print(f"Tutte energy/area {E.value(X0) / E.w.sum():.4f} -> {E.value(X) / E.w.sum():.4f} in {len(hist)} iterations, "
      f"flipped triangles {(dets(X, Fl) <= 0).sum()}")
np.savez(f"{OUT}/flatten.npz", V=V3, F=Fl, X=X)
