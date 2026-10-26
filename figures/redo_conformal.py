"""Recompute the "conformal" map in every viewer cache with the flow run to convergence.

    python figures/redo_conformal.py [workers]
"""
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "..", "assets")


def anisotropy(V, F, S):
    """Area-weighted mean of sigma1/sigma2 (1 = conformal) of mesh triangle -> chord triangle."""
    def frame(P):
        e1, e2 = P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]]
        x = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
        y = np.cross(np.cross(e1, e2), x)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        return np.stack([np.stack([(e1 * x).sum(1), (e1 * y).sum(1)], 1),
                         np.stack([(e2 * x).sum(1), (e2 * y).sum(1)], 1)], 2)
    s = np.linalg.svd(frame(S) @ np.linalg.inv(frame(V)), compute_uv=False)
    area = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    return float(np.average(s[:, 0] / s[:, 1], weights=area))


def run(npz):
    from distortion.sphere_param import conformal_flow_init
    d = dict(np.load(npz))
    V, F = d["V"].astype(float), d["F"]
    name = os.path.basename(npz)[:-4]
    t = time.time()
    old = anisotropy(V, F, d["conformal"]) if "conformal" in d else None
    try:
        S = conformal_flow_init(V, F)
    except Exception as e:                         # flow pinched or repair failed: keep the old map
        return dict(name=name, old=old, new=None, error=str(e)[:80], time=time.time() - t)
    d["conformal"] = S.astype(d["symdir"].dtype)
    np.savez_compressed(npz, **d)
    return dict(name=name, old=old, new=anisotropy(V, F, S), time=time.time() - t)


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(ASSETS, "viewer", "cache100", "*.npz")) +
                   glob.glob(os.path.join(ASSETS, "fig", "cache", "*.npz")))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as pool:
        out = []
        for r in pool.imap_unordered(run, files):
            print(json.dumps(r), flush=True)
            out.append(r)
    json.dump(out, open(os.path.join(ASSETS, "viewer", "cache100", "conformal_log.json"), "w"), indent=1)
