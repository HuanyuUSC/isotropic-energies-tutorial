"""2D deformation: pin the gingerbread man's feet, drag his right hand (projected Newton, flip-free steps)."""
import numpy as np
from _common import OUT, timer

from distortion.core import Elastic, Problem, Springs, minimize
from distortion.eigensystem import SymmetricDirichlet
from distortion.planar import dets, gingerbread

V, F, G = gingerbread(0.004)
pinned = np.zeros(len(V), bool)
pinned[G["feet"]] = True
hand = Springs(G["right_hand"], V[G["right_hand"]] + [0.9, 1.2], k=1e3)
P = Problem(len(V), 2, [Elastic(F, SymmetricDirichlet(), V=V), hand], pinned=pinned)
with timer(f"{len(F)} triangles"):
    hist = []
    X = minimize(P, V.copy(), "pnewton", history=hist, keep_going_when_capped=False)
print(f"{len(hist)} Newton iterations, energy {P.value(X):.4f}, flipped triangles {(dets(X, F) <= 0).sum()}")
np.savez(f"{OUT}/drag2d.npz", V=V, F=F, X=X)
