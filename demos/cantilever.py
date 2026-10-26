"""3D dynamics: a soft block falls on a cantilever beam clamped at one end, bounces off and lands on the floor."""
import numpy as np
from _common import OUT, timer

from distortion.contact import Plane, Scene, box_tets
from distortion.eigensystem import NeoHookean

VB, TB = box_tets((1.6, 0.2, 0.5), (32, 4, 10), origin=(0.0, 1.0, -0.25))
VA, TA = box_tets((0.4, 0.4, 0.4), (8, 8, 8), origin=(0.95, 1.7, -0.2))
pinned = np.r_[VB[:, 0] < 1e-9, np.zeros(len(VA), bool)]
S = Scene([(VB, TB, NeoHookean(mu=7.7e3, lam=1.15e4), 1.0), (VA, TA, NeoHookean(mu=60.0, lam=240.0), 1.0)],
          planes=[Plane((0, 1, 0), 0.0, dhat=0.02, kappa=1e5, mu=0.5)], pinned=pinned, h=1 / 120, dhat=4e-3, kappa=1e5)
X, Vel = S.X0.copy(), np.zeros_like(S.X0)
Vel[len(VB):] = [0.3, 0.0, 0.0]
frames = [X.copy()]
with timer(f"{len(S.T)} tetrahedra, 2 s at 120 steps/s"):
    for k in range(int(__import__("os").environ.get("STEPS", 240))):
        X, Vel = S.step(X, Vel)
        if k % 2:
            frames.append(X.copy())
print(f"mean Newton iterations per step {np.mean([s[1] for s in S.stats]):.1f}")
np.savez(f"{OUT}/cantilever.npz", T=S.T, frames=np.array(frames))
