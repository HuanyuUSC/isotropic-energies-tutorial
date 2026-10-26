"""3D parameterization: the Stanford bunny onto the sphere, flip-free (Symmetric Dirichlet on the tetrahedra made of
each triangle and the sphere's center). The energy is core.Elastic; the loop adds the sphere's tangent projection."""
import numpy as np
from _common import OUT, timer

from distortion.meshes import genus0_mesh
from distortion.sphere_param import parameterize, triple

V, F = genus0_mesh("bunny", 10000)
with timer(f"{len(F)} triangles"):
    S, hist = parameterize(V, F, verbose=False)
print(f"energy {hist[0][1]:.3f} -> {hist[-1][1]:.3f} in {len(hist) - 1} iterations, flipped triangles {(triple(S, F) <= 0).sum()}")
np.savez(f"{OUT}/sphere.npz", V=V, F=F, S=S)
