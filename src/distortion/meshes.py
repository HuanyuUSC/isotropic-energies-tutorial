"""Mesh loading and cleanup: closed genus-0 test meshes for spherical parameterization."""
import gzip
import io
import os
import tarfile
import urllib.request

import numpy as np

CACHE = os.path.join(os.path.expanduser("~"), ".cache", "distortion")

STANFORD = {
    "bunny": ("http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz",
              "bunny/reconstruction/bun_zipper.ply"),
    "armadillo": ("http://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz",
                  None),
}


def _download(url):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, os.path.basename(url))
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def _read_ply(data):
    import trimesh
    m = trimesh.load(io.BytesIO(data), file_type="ply", process=True)
    return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)


def load_stanford(name):
    """Raw Stanford scan (V, F). Data: Stanford 3D Scanning Repository; please credit."""
    url, member = STANFORD[name]
    path = _download(url)
    if path.endswith(".tar.gz"):
        with tarfile.open(path) as tar:
            data = tar.extractfile(member).read()
    else:
        with gzip.open(path) as f:
            data = f.read()
    return _read_ply(data)


def boundary_loops(F):
    """Boundary loops of a manifold triangle mesh, as lists of vertex indices (oriented like F)."""
    half = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    interior = {(a, b) for a, b in half}
    nxt = {a: b for a, b in half if (b, a) not in interior}
    loops, seen = [], set()
    for start in nxt:
        if start in seen:
            continue
        loop, v = [], start
        while v not in seen:
            seen.add(v)
            loop.append(v)
            v = nxt[v]
        loops.append(loop)
    return loops


def fill_holes(V, F):
    """Close every boundary loop with a fan around its centroid (keeps the mesh manifold)."""
    V, F = list(V), [F]
    for loop in boundary_loops(F[0]):
        c = len(V)
        V.append(np.mean([V[i] for i in loop], axis=0))
        # A boundary half-edge (a, b) belongs to an existing face; the new face runs (b, a, c).
        F.append(np.array([[loop[(k + 1) % len(loop)], loop[k], c] for k in range(len(loop))]))
    return np.array(V), np.concatenate(F)


def decimate(V, F, n_faces):
    import igl
    U, G, _, _ = igl.qslim(np.asfortranarray(V, float), np.asfortranarray(F, np.int32), n_faces)
    return np.asarray(U, float), np.asarray(G, np.int64)


def normalize_area(V, F, area=4 * np.pi):
    """Center the mesh and scale it so its total surface area equals `area`."""
    a = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum()
    V = V - V.mean(axis=0)
    return V * np.sqrt(area / a)


def genus0_mesh(name, n_faces=20000):
    """A closed, genus-0, area-4π version of a Stanford model."""
    V, F = load_stanford(name)
    V, F = fill_holes(V, F)
    if len(F) > n_faces:
        V, F = decimate(V, F, n_faces)
    chi = len(V) - len(np.unique(np.sort(np.concatenate(
        [F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)) + len(F)
    assert chi == 2 and not boundary_loops(F), f"not a closed genus-0 mesh (chi={chi})"
    return normalize_area(V, F), F


def read_off(path):
    """Minimal OFF reader (triangles only). Returns V (n, 3) float, F (m, 3) int."""
    with open(path, encoding="latin1") as f:
        tok = f.read().split()
    k = 0
    if tok[0].upper().startswith("OFF"):
        if len(tok[0]) > 3:          # "OFF123 ..." glued header
            tok[0] = tok[0][3:]
        else:
            k = 1
    nv, nf = int(tok[k]), int(tok[k + 1])
    k += 3
    V = np.array(tok[k:k + 3 * nv], float).reshape(nv, 3)
    k += 3 * nv
    F = np.array(tok[k:k + 4 * nf], int).reshape(nf, 4)
    assert (F[:, 0] == 3).all(), "only triangle meshes are supported"
    return V, F[:, 1:]
