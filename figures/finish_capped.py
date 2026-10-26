"""Continue the Symmetric Dirichlet optimization for meshes that hit the iteration cap."""
import json, os, sys, time
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np


def one(path):
    from distortion.sphere_param import parameterize, triple
    d = dict(np.load(path))
    t = time.time()
    S, h = parameterize(d["V"], d["F"], d["symdir"], schedule=(("newton", 1e-6),), iters=1500, verbose=False)
    d["symdir"] = S
    np.savez_compressed(path, **d)
    return dict(name=os.path.basename(path)[:-4], extra_iters=len(h) - 1, extra_time=round(time.time() - t, 1),
                E=h[-1][1], flips=int((triple(S, d["F"]) <= 0).sum()))


if __name__ == "__main__":
    from multiprocessing import Pool
    folder = sys.argv[1]
    log = json.load(open(os.path.join(folder, "batch_log.json")))
    capped = [r["name"] for r in log if r.get("symdir_iters", 0) >= 200]
    with Pool(len(capped)) as pool:
        for r in pool.imap_unordered(one, [os.path.join(folder, n + ".npz") for n in capped]):
            print(json.dumps(r), flush=True)
            for e in log:
                if e["name"] == r["name"]:
                    e["symdir_iters"] += r["extra_iters"]
                    e["symdir_time"] = round(e["symdir_time"] + r["extra_time"], 1)
                    e["symdir_E"], e["symdir_flips"] = r["E"], r["flips"]
            json.dump(log, open(os.path.join(folder, "batch_log.json"), "w"), indent=1)
