"""Compute spherical maps for a folder of closed genus-0 meshes (one .npz per mesh).

For each <name>.off (and its reference <name>_sphere.off, if present):
  naive (tetrahedron layout), conformal (cMCF + repair), symdir (SLIM -> projected Newton).
Meshes above MAX_FACES are decimated first (the reference map is then dropped).

    python figures/batch_sphere.py <mesh-dir> <out-dir> [workers]
"""
import glob
import json
import os
import sys
import time
import traceback

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np

MAX_FACES = 40000


def one(args):
    path, out = args
    from distortion.meshes import decimate, normalize_area, read_off
    from distortion.polytope_init import polytope_map
    from distortion.sphere_param import conformal_flow_init, parameterize, triple, untangle
    name = os.path.basename(path)[:-4]
    dst = os.path.join(out, name + ".npz")
    if os.path.exists(dst):
        return dict(name=name, status="cached")
    info = dict(name=name)
    t0 = time.time()
    try:
        V, F = read_off(path)
        maps = {}
        ref = path[:-4] + "_sphere.off"
        if os.path.exists(ref):
            S, Fs = read_off(ref)
            if Fs.shape == F.shape and (Fs == F).all():
                maps["reference"] = S / np.linalg.norm(S, axis=1, keepdims=True)
        if len(F) > MAX_FACES:
            V0, F0 = V, F
            V, F = decimate(V, F, MAX_FACES)
            info["decimated"] = True
            if "reference" in maps:   # carry the reference map over: project, interpolate, normalize
                import igl
                _, fi, C = igl.point_mesh_squared_distance(V, V0, F0.astype(np.int32))
                T = V0[F0[fi]]
                B = igl.barycentric_coordinates(C, T[:, 0].copy(), T[:, 1].copy(), T[:, 2].copy())
                R = np.einsum("vk,vkd->vd", B, maps["reference"][F0[fi]])
                maps["reference"] = R / np.linalg.norm(R, axis=1, keepdims=True)
        # consistent orientation: outward normals (positive enclosed volume)
        if np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() < 0:
            F = F[:, [0, 2, 1]]
        V = normalize_area(V, F)
        info.update(nv=len(V), nf=len(F))
        def tet_start(V, F):
            for seed in range(6):
                try:
                    S, _ = polytope_map(V, F, "tetrahedron", seed=seed)
                    if (triple(S, F) > 0).all():
                        return S
                except Exception:
                    pass
            return None
        start, start_kind = None, None
        if "reference" in maps:          # the dataset map: flip-free up to text rounding -> repair
            try:
                if (triple(maps["reference"], F) < 0).sum() > len(F) / 2:   # faces were reoriented
                    maps["reference"][:, 0] *= -1
                start, start_kind = untangle(maps["reference"].copy(), F), "reference"
            except Exception as e:
                info["reference_repair_error"] = str(e)[:120]
        naive = tet_start(V, F)
        if naive is None and start is None and len(F) < 25000:
            # thin parts too few vertices wide for disjoint paths: subdivide once and retry
            from distortion.render import subdivide
            ref = maps.get("reference")
            V2, R2, F2 = subdivide(V, ref if ref is not None else V / np.linalg.norm(V, axis=1, keepdims=True), F, 1)
            naive = tet_start(V2, F2)
            if naive is not None:
                V, F = V2, F2
                if ref is not None:
                    maps["reference"] = R2
                info.update(subdivided=True, nv=len(V), nf=len(F))
        if naive is not None:
            maps["naive"] = naive
        try:
            maps["conformal"] = conformal_flow_init(V, F)
        except Exception as e:
            info["conformal_error"] = str(e)[:120]
        if start is None and naive is not None:
            start, start_kind = naive, "tetrahedron"
        if start is None and "conformal" in maps:
            start, start_kind = maps["conformal"], "conformal"
        if start is not None:
            t = time.time()
            S, hist = parameterize(V, F, start, iters=200, verbose=False)
            info.update(symdir_time=round(time.time() - t, 1), symdir_E=hist[-1][1], symdir_iters=len(hist) - 1,
                        symdir_flips=int((triple(S, F) <= 0).sum()),
                        symdir_start=start_kind)
            maps["symdir"] = S
        for k, S in maps.items():
            info[f"{k}_flips"] = int((triple(S, F) <= 0).sum())
        np.savez_compressed(dst, V=V, F=F, **maps)
        info["status"] = "ok"
    except Exception as e:
        info.update(status="error", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-600:])
    info["seconds"] = round(time.time() - t0, 1)
    return info


if __name__ == "__main__":
    from multiprocessing import Pool
    src, out = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    os.makedirs(out, exist_ok=True)
    # a file is a reference map only if its mesh exists too (the benchmark has a mesh named "sharp_sphere")
    allp = set(glob.glob(os.path.join(src, "*.off")))
    paths = sorted(p for p in allp if not (p.endswith("_sphere.off") and p[:-len("_sphere.off")] + ".off" in allp))
    only = sys.argv[4:]
    if only:
        paths = [p for p in paths if os.path.basename(p)[:-4] in only]
    # big meshes first so the pool ends evenly
    paths.sort(key=lambda p: -os.path.getsize(p))
    logp = os.path.join(out, "batch_log.json")
    log = [r for r in json.load(open(logp)) if r["name"] not in {os.path.basename(p)[:-4] for p in paths}] if os.path.exists(logp) else []
    with Pool(workers) as pool:
        for info in pool.imap_unordered(one, [(p, out) for p in paths]):
            log.append(info)
            print(json.dumps({k: v for k, v in info.items() if k != "trace"}), flush=True)
            json.dump(log, open(os.path.join(out, "batch_log.json"), "w"), indent=1)
