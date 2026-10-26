"""3D deformation: the extruded gingerbread cookie, feet pinned, its right hand raised step by step (quasistatic)."""
import numpy as np
from _common import OUT, timer

from distortion.contact import Scene
from distortion.eigensystem import NeoHookean
from distortion.solid import extruded_gingerbread

V, T, G = extruded_gingerbread(0.01)
hand, feet = G["right_hand"], G["feet"]
pinned = np.zeros(len(V), bool)
pinned[hand] = pinned[feet] = True


def path(t, X0):                                   # the hand moves up and toward the viewer
    X = X0.copy()
    X[hand] = X0[hand] + t * np.array([0.3, 1.2, 0.6])
    return X


S = Scene([(V, T, NeoHookean(mu=1.0, lam=4.0), 1.0)], pinned=pinned, pin_path=path, h=None, gravity=(0, 0, 0),
          ipc=False)
frames = [V.copy()]
STEPS = 40                                           # small hand moves: a big jump of the pinned vertices inverts tets
with timer(f"{len(T)} tetrahedra, {STEPS} quasistatic solves"):
    X = V.copy()
    for k, t in enumerate(np.linspace(1 / STEPS, 1.0, STEPS)):
        X = S.solve(X, t)
        if (k + 1) % (STEPS // 10) == 0:
            frames.append(X.copy())
print(f"Newton iterations per solve: {[s[1] for s in S.stats]}, min det F {S.parts[0][0].dets(X).min():.3f}")
np.savez(f"{OUT}/drag3d.npz", V=V, T=T, frames=np.array(frames))
