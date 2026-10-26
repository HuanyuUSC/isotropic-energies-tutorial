# Six demos, one machine

Every demo builds a `core.Problem` from the same pieces and calls the same `core.minimize`:

| | 2D (triangles, F is 2×2) | 3D (tetrahedra, F is 3×3) |
|---|---|---|
| **parameterization** | `flatten.py`: a bunny patch → the plane (Tutte → ARAP or Symmetric Dirichlet) | `sphere.py`: the bunny → the sphere (flip-free Symmetric Dirichlet) |
| **deformation** | `drag2d.py`: drag the gingerbread man's hand | `drag3d.py`: raise the 3D cookie's arm (quasistatic) |
| **dynamics** | `letters.py`: soft letters fall into a box | `cantilever.py`: a soft block falls on a cantilever |

The shared parts (`src/distortion/core.py`):

* `Elastic`: per-element F = D_s D_m⁻¹ → batched SVD → ψ(σ), stress, closed-form Hessian eigensystem → projected
  element Hessians; elements processed in parallel thread chunks.
* `SparsePattern` + `LinearSolver`: the sparsity pattern is analysed once; each iteration only scatters values and
  factorizes (MKL PARDISO for large systems, SuperLU for small ones).
* Terms: `Springs`, `Inertia`, planes and IPC contact (`contact.py`), a boundary barrier (`boundary_barrier.py`).
  Every term gives value / gradient / Hessian and the largest safe step (no inversion, no crossing).
* `minimize`: gradient descent, Newton or projected Newton, with the safe step and Armijo backtracking.

Run any demo from the repository root, e.g. `python demos/drag2d.py`; results go to `demos/out/`.
