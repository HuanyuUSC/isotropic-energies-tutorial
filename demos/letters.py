"""2D dynamics: soft letters U C L A fall into a box (implicit Euler as minimization, IPC contact, friction)."""
import numpy as np
from _common import OUT, timer

from distortion.contact import Plane, Scene
from distortion.eigensystem import NeoHookean
from distortion.shapes2d import letter

bodies = []
for k, ch in enumerate("UCLA"):
    V, F = letter(ch, 1.0, 0.002)
    V = V - V.mean(0) + [0.9 * np.sin(1.75 * k - np.pi / 2), 2.0 + 1.3 * k]
    bodies.append((V, F, NeoHookean(mu=40.0, lam=160.0), 1.0))
W = 1.75
planes = [Plane((0, 1), 0.0, dhat=0.01, kappa=1e3, mu=0.4), Plane((1, 0), -W, dhat=0.01, kappa=1e3, mu=0.2),
          Plane((-1, 0), -W, dhat=0.01, kappa=1e3, mu=0.2)]
S = Scene(bodies, planes=planes, h=1 / 120, dhat=1e-2, kappa=1e3)
X, Vel, frames = S.X0.copy(), np.zeros_like(S.X0), [S.X0.copy()]
with timer(f"{len(S.T)} triangles, 3 s at 120 steps/s"):
    for k in range(int(__import__("os").environ.get("STEPS", 360))):
        X, Vel = S.step(X, Vel)
        if k % 2:
            frames.append(X.copy())
print(f"mean Newton iterations per step {np.mean([s[1] for s in S.stats]):.1f}")
np.savez(f"{OUT}/letters.npz", T=S.T, frames=np.array(frames))
