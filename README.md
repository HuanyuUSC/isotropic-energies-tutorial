# Isotropic Distortion Energies: one machine for parameterization, deformation and dynamics

Code for the talk *Isotropic Distortion Energies* (Huanyu Chen, UCLA, October 2026).

Every problem in the talk, from flattening a face to dropping soft toys onto springboards, is the same
computation: an isotropic energy ψ(σ) of each element's singular values, its closed-form Hessian eigensystem,
and projected Newton with safe steps. This repository is that machine, small and readable.

> **Fast prototyping, not production.** Everything is NumPy/SciPy; nothing is seriously optimized.
> Expect seconds for the 2D demos and minutes for the 3D dynamics.

## The six demos

| | 2D (triangles, F is 2×2) | 3D (tetrahedra, F is 3×3) |
|---|---|---|
| **parameterization** | `demos/flatten.py`: a bunny patch → the plane | `demos/sphere.py`: the bunny → the sphere, flip-free |
| **deformation** | `demos/drag2d.py`: drag the gingerbread man's hand | `demos/drag3d.py`: raise the 3D cookie's arm |
| **dynamics** | `demos/letters.py`: soft letters fall into a box | `demos/cantilever.py`: a soft block falls on a cantilever |

Run any demo from the repository root, e.g. `python demos/drag2d.py`; results go to `demos/out/`.
See `demos/README.md` for what each one shows.

## Layout

* `src/distortion/core.py`: the machine. `Elastic` (per-element F → SVD → ψ(σ), gradient, closed-form projected
  Hessian; any dimension), `Springs`, `Inertia`, a sparse `LinearSolver` (SuperLU, or MKL PARDISO when
  available), `Problem` and `minimize` (gradient descent, Newton, projected Newton, safe step + Armijo).
* `src/distortion/eigensystem.py`: the energies (ARAP, Symmetric Dirichlet, Neo-Hookean, StVK, Hencky,
  conformal, ...) and the closed-form eigensystem of any ψ(σ).
* `atlas.py` (flattening, texture atlases), `sphere_param.py` (flip-free spherical maps), `planar.py` and
  `solid.py` (2D / 3D deformation), `contact.py` (implicit Euler + IPC contact), `boundary_barrier.py`
  (globally injective 2D maps).
* `figures/`: the scripts that drew the slide figures. Several read data (meshes, photos, cached runs)
  that is not redistributed here, so treat them as examples rather than one-click reproductions.
* `tests/`: `check_derivatives.py` (every gradient and Hessian against finite differences) and
  `regression.py` (six small scenes against `baseline.json`).

## Install

```
pip install -e .
pip install ipctk libigl pyvista trimesh pillow    # contact, Tutte, rendering, mesh I/O
pip install pypardiso                             # optional: faster sparse solves
```

## Data

Meshes and textures are downloaded at runtime and are not redistributed here: Stanford 3D Scanning
Repository (bunny, armadillo; research use, please credit), NASA Blue Marble and Natural Earth (public domain).

## License

MIT (code only).
