"""Spherical maps for the meshes from the 2021 thesis website (F:\Dropbox\3Dmodels\parameterization):
<name>_INPUT.obj (mesh), _INIT.obj (initial spherical map), _RESULT.obj (the 2016/2021 result).
Our pipeline starts from INIT: SLIM -> projected Newton on Symmetric Dirichlet.

    python figures/param_meshes.py <cache-dir> name=prefix ...
"""
import os
import sys
import time

import numpy as np

from distortion.meshes import normalize_area
from distortion.sphere_param import conformal_flow_init, parameterize, triple

BASE = r"F:\Dropbox\3Dmodels\parameterization"


def read_obj(path):
    V, F = [], []
    for l in open(path):
        if l.startswith("v "):
            V.append([float(x) for x in l.split()[1:4]])
        elif l.startswith("f "):
            F.append([int(x.split("/")[0]) - 1 for x in l.split()[1:4]])
    return np.array(V), np.array(F)


def iso(V, F, S):
    def frame(P):
        e1, e2 = P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]]
        x = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
        y = np.cross(np.cross(e1, e2), x)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        return np.stack([np.stack([(e1 * x).sum(1), (e1 * y).sum(1)], 1), np.stack([(e2 * x).sum(1), (e2 * y).sum(1)], 1)], 2)
    s = np.linalg.svd(frame(S) @ np.linalg.inv(frame(V)), compute_uv=False)
    A = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    w = A / A.sum()
    la = np.log2(s.prod(1))
    return dict(iso=float(w @ np.maximum(s[:, 0], 1 / s[:, 1])), shape=float(w @ (s[:, 0] / s[:, 1])),
                area_spread=float(np.sqrt(w @ (la - w @ la) ** 2)))


def find(sub, kind):
    import glob
    hits = sorted(glob.glob(os.path.join(BASE, sub + f"_{kind}*.obj")))
    assert hits, (sub, kind)
    return hits[0]


if __name__ == "__main__":
    out = sys.argv[1]
    for arg in sys.argv[2:]:
        name, sub = arg.split("=")                # e.g. gardenpig=GardenPig/Hierachical/GardenPig
        V, F = read_obj(find(sub, "INPUT"))
        V = normalize_area(V, F)
        init = read_obj(find(sub, "INIT"))[0]
        res = read_obj(find(sub, "RESULT"))[0]
        init = init - init.mean(0) if np.ptp(np.linalg.norm(init - init.mean(0), axis=1)) > 1e-3 else init
        init /= np.linalg.norm(init, axis=1, keepdims=True)
        res /= np.linalg.norm(res, axis=1, keepdims=True)
        start, used = init, "INIT"
        if (triple(init, F) <= 0).any():
            start, used = res, "RESULT"
            if (triple(res, F) <= 0).any():
                print(f"{name}: neither INIT nor RESULT is flip-free; skipped")
                continue
        t = time.time()
        S, h = parameterize(V, F, start, iters=400, verbose=False)
        print(f"{name}: V {len(V)}, from {used}, {len(h) - 1} iterations, {time.time() - t:.0f} s, flips {(triple(S, F) <= 0).sum()}")
        maps = dict(V=V, F=F, symdir=S, thesis_init=init, thesis_result=res)
        try:
            from distortion.polytope_init import polytope_map
            maps["naive"], _ = polytope_map(V, F, "tetrahedron")
        except Exception as e:
            print("  tetrahedron init failed:", str(e)[:60])
        try:
            maps["conformal"] = conformal_flow_init(V, F)
        except Exception as e:
            print("  conformal failed:", str(e)[:60])
        for k in ["thesis_result", "symdir"]:
            print(f"  {k:14s}", {a: round(b, 3) for a, b in iso(V, F, maps[k]).items()})
        np.savez_compressed(os.path.join(out, name + ".npz"), **maps)
