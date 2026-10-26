"""Spherical maps with volume (area) preservation: Symmetric Dirichlet + kappa (J - 1)^2,
kappa chosen for Poisson's ratio nu.  Continuation from the nu = 0 map, warm-starting each nu.

    python figures/poisson_sweep.py <out-dir> name=path/to/cache.npz ...
"""
import json, os, sys, time
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np

NUS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.49]


def one(args):
    name, src, out = args
    from distortion.eigensystem import SymmetricDirichlet, with_poisson
    from distortion.sphere_param import parameterize, triple
    d = np.load(src)
    V, F, S = d["V"], d["F"], d["symdir"]
    res, log = {"V": V, "F": F, "nu_0.00": S}, []
    for nu in NUS[1:]:
        e = with_poisson(SymmetricDirichlet(), nu)          # E = 1, Poisson's ratio nu
        k = e.kappa
        t = time.time()
        S, h = parameterize(V, F, S, energy=e,
                            schedule=(("newton", 1e-7),), iters=400, verbose=False)
        res[f"nu_{nu:.2f}"] = S
        log.append(dict(nu=nu, kappa=k, iters=len(h) - 1, time=round(time.time() - t, 1),
                        E=h[-1][1], flips=int((triple(S, F) <= 0).sum())))
    np.savez_compressed(os.path.join(out, name + ".npz"), **res)
    return name, log


if __name__ == "__main__":
    from multiprocessing import Pool
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    jobs = [(a.split("=")[0], a.split("=")[1], out) for a in sys.argv[2:]]
    with Pool(len(jobs)) as pool:
        for name, log in pool.imap_unordered(one, jobs):
            for r in log:
                print(name, json.dumps(r), flush=True)
            json.dump(log, open(os.path.join(out, name + "_log.json"), "w"), indent=1)
