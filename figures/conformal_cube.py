"""Conformal cubed sphere: a lookup table from central-projection (gnomonic) coordinates of one cube
face to coordinates in which the checkerboard is conformal (right angles kept everywhere except at
the 8 cube corners, where three 90-degree corners must open up to 360 degrees).

The face's conformal coordinate u is the harmonic function on the spherical face with u = +-1 on
the edges x = +-1 and zero normal derivative on y = +-1 (level lines meet those edges at right
angles).  Laplace's equation is conformally invariant, and the face is square-symmetric (conformal
modulus 1), so (u(x, y), u(y, x)) maps the face conformally onto the square [-1, 1]^2 [cf. Rancic,
Purser, Mesinger 1996].  We solve it with P1 finite elements on the sphere (cotangent Laplacian).

    python figures/conformal_cube.py       # self-check: angle distortion of both cube maps
"""
import numpy as np
import scipy.sparse.linalg as spla

from distortion.sphere_param import cotan_laplacian


def grid(N):
    t = np.linspace(-1, 1, N)
    x, y = np.meshgrid(t, t)                        # x along columns, y along rows
    P = np.stack([x.ravel(), y.ravel(), np.ones(N * N)], 1)
    i, j = np.meshgrid(np.arange(N - 1), np.arange(N - 1))
    a = (j * N + i).ravel()
    F = np.concatenate([np.stack([a, a + 1, a + N + 1], 1), np.stack([a, a + N + 1, a + N], 1)])
    return x, y, P / np.linalg.norm(P, axis=1, keepdims=True), F


def table(N=257):
    """u on an N x N grid of gnomonic coordinates (row = y, column = x), x, y in [-1, 1]."""
    x, y, S, F = grid(N)
    L = cotan_laplacian(S, F).tocsr()
    x = x.ravel()
    fixed = np.abs(np.abs(x) - 1) < 1e-12
    free = ~fixed
    u = np.where(fixed, np.sign(x), 0.0)
    u[free] = spla.spsolve(L[free][:, free].tocsc(), -L[free][:, fixed] @ u[fixed])
    U = u.reshape(N, N)
    U = (U - U[:, ::-1]) / 2                          # odd in x, even in y (the triangulation's
    return (U + U[::-1]) / 2                          # diagonals break the symmetry by ~1e-3)


def angle_ratio(fx, fy, h=1e-4):
    """sigma1/sigma2 of sphere -> (u, v) at random face points, for a map given as two functions
    of gnomonic (x, y).  Uses the sphere's tangent frame, so 1 = conformal."""
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-0.98, 0.98, (2, 20000))
    out = []
    for X, Y in zip(x, y):
        p = lambda a, b: np.array([a, b, 1.0]) / np.sqrt(1 + a * a + b * b)
        Js = np.stack([(p(X + h, Y) - p(X - h, Y)) / (2 * h), (p(X, Y + h) - p(X, Y - h)) / (2 * h)], 1)
        Ju = np.array([[fx(X + h, Y) - fx(X - h, Y), fx(X, Y + h) - fx(X, Y - h)],
                       [fy(X + h, Y) - fy(X - h, Y), fy(X, Y + h) - fy(X, Y - h)]]) / (2 * h)
        Q, R = np.linalg.qr(Js)                       # sphere tangent frame
        s = np.linalg.svd(Ju @ np.linalg.inv(R), compute_uv=False)
        out.append(s[0] / s[1])
    out = np.array(out)
    w = (1 + x * x + y * y) ** -1.5                  # sphere area weight
    return np.average(out, weights=w), np.percentile(out, [50, 99]), out.max()


if __name__ == "__main__":
    from scipy.interpolate import RectBivariateSpline
    N = 257
    U = table(N)
    t = np.linspace(-1, 1, N)
    sp = RectBivariateSpline(t, t, U)                 # (row = y, col = x)
    fu = lambda x, y: sp(y, x)[0, 0]
    fv = lambda x, y: sp(x, y)[0, 0]
    print("symmetry  max|u(-x,y)+u(x,y)| %.1e" % np.abs(U + U[:, ::-1]).max())
    print("central projection: mean %.3f  p50/p99 %s  max %.3f" % angle_ratio(lambda x, y: x, lambda x, y: y))
    print("conformal table:    mean %.3f  p50/p99 %s  max %.3f" % angle_ratio(fu, fv))
