"""Benchmark summary: dataset spherical map vs. ours (Symmetric Dirichlet) on every mesh."""
import glob, json, os, sys
import numpy as np
from distortion.sphere_param import SphericalEnergy, triple, untangle

folder = sys.argv[1]


def iso(V, S, F):
    """Per-face isometric distortion max(sigma1, 1/sigma2) of mesh triangle -> chord triangle; area weights."""
    def frame(P):
        e1, e2 = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
        x = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
        n = np.cross(e1, e2)
        y = np.cross(n / np.linalg.norm(n, axis=1, keepdims=True), x)
        return np.stack([np.stack([np.einsum("ij,ij->i", e, x), np.einsum("ij,ij->i", e, y)], 1) for e in (e1, e2)], 2)
    s = np.linalg.svd(frame(S[F]) @ np.linalg.inv(frame(V[F])), compute_uv=False)
    A = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    d = np.maximum(s[:, 0], 1 / s[:, 1])
    return float(A @ d / A.sum()), float(d.max())


rows = []
for p in sorted(glob.glob(os.path.join(folder, "*.npz")), key=str.lower):
    d = np.load(p)
    if "reference" not in d:
        continue
    V, F, R, S = d["V"], d["F"], d["reference"], d["symdir"]
    E = SphericalEnergy(V, F)
    R0 = untangle(R.copy(), F)
    rows.append(dict(name=os.path.basename(p)[:-4], E_ref=E.value(R0), E_ours=E.value(S),
                     iso_ref=iso(V, R0, F), iso_ours=iso(V, S, F),
                     flips_ref_raw=int((triple(R, F) <= 0).sum()), flips_ours=int((triple(S, F) <= 0).sum())))
json.dump(rows, open(os.path.join(folder, "benchmark_stats.json"), "w"), indent=1)
r = lambda k: np.array([x[k] for x in rows])
ratio = r("E_ours") / r("E_ref")
im = lambda k: np.array([x[k][0] for x in rows])
print(f"meshes: {len(rows)}   ours flip-free: {(r('flips_ours') == 0).sum()}   dataset files with rounding flips: {(r('flips_ref_raw') > 0).sum()}")
print(f"SymDir energy ours/dataset: median {np.median(ratio):.3f}  min {ratio.min():.3f}  max {ratio.max():.3f}  (lower in {(ratio < 1).sum()}/{len(rows)})")
print(f"mean isometric distortion: dataset median {np.median(im('iso_ref')):.2f} -> ours median {np.median(im('iso_ours')):.2f};  ours lower in {(im('iso_ours') < im('iso_ref')).sum()}/{len(rows)}")
worse = [x["name"] for x in rows if x["E_ours"] > x["E_ref"]]
print("ours worse in energy:", worse)
