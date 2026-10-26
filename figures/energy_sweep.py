"""Spherical maps under different distortion energies, all with nu = 0 (lambda = 0), so they agree
to second order at rest and differ only in how they penalize large stretch and compression.
Warm start from the Symmetric Dirichlet map; projected Newton.

    python figures/energy_sweep.py <out-dir> <workers> name=path/to/cache.npz ...
"""
import json, os, sys, time
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np


def energies():
    from distortion.eigensystem import ARAP, Hencky, NeoHookean, StVK, SymmetricARAP, WithBarrier, normalized
    # all normalized to Young's modulus E = 1 (nu = 0); scaling does not change the maps
    raw = {"neohookean": NeoHookean(mu=1.0, lam=0.0), "hencky": Hencky(), "symarap": SymmetricARAP(),
           "arap_b": WithBarrier(ARAP()), "stvk_b": WithBarrier(StVK())}
    return {k: normalized(e) for k, e in raw.items()}


def one(args):
    name, src, key, out = args
    from distortion.sphere_param import parameterize, triple
    d = np.load(src)
    V, F = d["V"], d["F"]
    t = time.time()
    try:
        S, h = parameterize(V, F, d["symdir"], energy=energies()[key],
                            schedule=(("newton", 1e-7),), iters=400, verbose=False)
    except Exception as e:
        return dict(name=name, energy=key, error=str(e)[:100])
    np.save(os.path.join(out, f"{name}__{key}.npy"), S)
    return dict(name=name, energy=key, iters=len(h) - 1, time=round(time.time() - t, 1),
                E=h[-1][1], flips=int((triple(S, F) <= 0).sum()))


if __name__ == "__main__":
    from multiprocessing import Pool
    out, workers = sys.argv[1], int(sys.argv[2])
    os.makedirs(out, exist_ok=True)
    jobs = [(a.split("=")[0], a.split("=")[1], k, out) for a in sys.argv[3:] for k in energies()]
    log = []
    with Pool(workers) as pool:
        for r in pool.imap_unordered(one, jobs):
            print(json.dumps(r), flush=True)
            log.append(r)
    json.dump(log, open(os.path.join(out, "log.json"), "w"), indent=1)
