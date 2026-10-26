"""Isotropic energies of d x d deformation gradients (d = 2 or 3) and their analytic
Hessian eigensystem.

An isotropic energy depends on F only through its (signed) singular values,
psi(F) = psi_hat(s_1, ..., s_d).  With F = U diag(s) V^T, the d^2 x d^2 Hessian w.r.t.
vec(F) has a closed-form eigensystem [Smith, De Goes, Kim 2019]:
  * d scaling modes  U diag(e) V^T,   e = eigenvectors of the d x d Hessian of psi_hat,
  * one twist mode per pair (i, j):  U (E_ij - E_ji) V^T / sqrt2,  lambda = (psi_i + psi_j) / (s_i + s_j),
  * one flip mode per pair (i, j):   U (E_ij + E_ji) V^T / sqrt2,  lambda = (psi_i - psi_j) / (s_i - s_j).
(2D: 2 + 1 + 1 = 4 modes; 3D: 3 + 3 + 3 = 9.)  Everything is batched over a leading axis.
"""
from itertools import combinations

import numpy as np


def signed_svd(F):
    """F = U diag(s) V^T with U, V proper rotations; a reflection shows up as s[..., -1] < 0."""
    U, s, Vt = np.linalg.svd(F)
    V = np.swapaxes(Vt, -1, -2)
    fixU = np.linalg.det(U) < 0
    fixV = np.linalg.det(V) < 0
    U[fixU, :, -1] *= -1
    s[fixU, -1] *= -1
    V[fixV, :, -1] *= -1
    s[fixV, -1] *= -1
    return U, s, V


def _diag_hess(h):
    """Batched diagonal Hessian from its diagonal (..., d) -> (..., d, d)."""
    return h[..., :, None] * np.eye(h.shape[-1])


class SymmetricDirichlet:
    """psi = sum_i s_i^2 + s_i^-2: an inversion barrier (infinite as det F -> 0)."""
    name = "Symmetric Dirichlet"
    barrier = True

    def psi(self, s):
        return np.sum(s**2 + s**-2, axis=-1)

    def grad(self, s):
        return 2 * s - 2 * s**-3

    def hess(self, s):
        return _diag_hess(2 + 6 * s**-4)


class ARAP:
    """psi = sum_i (s_i - 1)^2."""
    name = "ARAP"
    barrier = False

    def psi(self, s):
        return np.sum((s - 1) ** 2, axis=-1)

    def grad(self, s):
        return 2 * (s - 1)

    def hess(self, s):
        return _diag_hess(2 * np.ones_like(s))


class Hencky:
    """psi = sum_i log^2 s_i (barrier: log blows up at s -> 0)."""
    name = "Hencky"
    barrier = True

    def psi(self, s):
        return np.sum(np.log(s) ** 2, axis=-1)

    def grad(self, s):
        return 2 * np.log(s) / s

    def hess(self, s):
        return _diag_hess((2 - 2 * np.log(s)) / s**2)


class Conformal2D:
    """2D MIPS / conformal energy: s1/s2 + s2/s1 (scale-blind; barrier)."""
    name = "Conformal"
    barrier = True

    def psi(self, s):
        return s[..., 0] / s[..., 1] + s[..., 1] / s[..., 0]

    def grad(self, s):
        a, b = s[..., 0], s[..., 1]
        return np.stack([1 / b - b / a**2, 1 / a - a / b**2], -1)

    def hess(self, s):
        a, b = s[..., 0], s[..., 1]
        H = np.empty(s.shape + (2,))
        H[..., 0, 0] = 2 * b / a**3
        H[..., 1, 1] = 2 * a / b**3
        H[..., 0, 1] = H[..., 1, 0] = -1 / b**2 - 1 / a**2
        return H


class NeoHookean:
    """psi = mu/2 (sum s_i^2 - d) - mu log J + lam/2 log^2 J,  J = prod s_i (barrier)."""
    name = "Neo-Hookean"
    barrier = True

    def __init__(self, mu=1.0, lam=1.0):
        self.mu, self.lam = mu, lam

    def psi(self, s):
        logJ = np.sum(np.log(s), axis=-1)
        return 0.5 * self.mu * (np.sum(s**2, -1) - s.shape[-1]) - self.mu * logJ + 0.5 * self.lam * logJ**2

    def grad(self, s):
        logJ = np.sum(np.log(s), axis=-1)[..., None]
        return self.mu * s - self.mu / s + self.lam * logJ / s

    def hess(self, s):
        logJ = np.sum(np.log(s), axis=-1)[..., None]
        H = self.lam / (s[..., :, None] * s[..., None, :])
        return H + _diag_hess(self.mu + self.mu / s**2 - self.lam * logJ / s**2)


class SymmetricARAP:
    """psi = sum_i (s_i - 1)^2 + (1/s_i - 1)^2: ARAP made symmetric under s -> 1/s (barrier)."""
    name = "Symmetric ARAP"
    barrier = True

    def psi(self, s):
        return np.sum((s - 1) ** 2 + (1 / s - 1) ** 2, axis=-1)

    def grad(self, s):
        return 2 * (s - 1) - 2 * (1 / s - 1) / s**2

    def hess(self, s):
        return _diag_hess(2 + 6 / s**4 - 4 / s**3)


class StVK:
    """St. Venant-Kirchhoff in principal stretches, lambda = 0: psi = sum_i ((s_i^2 - 1) / 2)^2.
    Quartic in stretch, but only 1/4 at s = 0: no barrier."""
    name = "StVK"
    barrier = False

    def psi(self, s):
        return np.sum(((s**2 - 1) / 2) ** 2, axis=-1)

    def grad(self, s):
        return s * (s**2 - 1)

    def hess(self, s):
        return _diag_hess(3 * s**2 - 1)


class WithBarrier:
    """base(s) + k sum_i b(s_i), with the C2 log barrier of IPC [Li et al. 2020]
    b(s) = -(s - h)^2 log(s / h) for s < h, 0 otherwise: infinite at s = 0, but the base energy
    is untouched for s >= h (so the rest-state Lame parameters do not change)."""

    def __init__(self, base, h=0.5, k=1.0):
        self.base, self.h, self.k = base, h, k
        self.name = f"{base.name} + barrier"
        self.barrier = True

    def _b(self, s):
        on = s < self.h
        t = np.where(on, s, self.h)
        d, L = t - self.h, np.log(t / self.h)
        return (on * -(d**2) * L, on * (-2 * d * L - d**2 / t), on * (-2 * L - 4 * d / t + d**2 / t**2))

    def psi(self, s):
        return self.base.psi(s) + self.k * np.sum(self._b(s)[0], axis=-1)

    def grad(self, s):
        return self.base.grad(s) + self.k * self._b(s)[1]

    def hess(self, s):
        return self.base.hess(s) + self.k * _diag_hess(self._b(s)[2])


def signed_singular_values(F):
    """Signed singular values only (no U, V): enough for the energy, e.g. in a line search.
    The sign of det F goes on the smallest one, as in signed_svd."""
    s = np.linalg.svd(F, compute_uv=False)
    s[..., -1] *= np.sign(np.linalg.det(F))
    return s


def psi_and_P(energy, F, usv=None):
    """Energy density and first Piola stress P = dpsi/dF = U diag(psi_i) V^T.
    usv: a precomputed signed_svd(F), so one SVD serves the stress and the Hessian."""
    U, s, V = signed_svd(F) if usv is None else usv
    P = (U * energy.grad(s)[..., None, :]) @ np.swapaxes(V, -1, -2)
    return energy.psi(s), P


def hessian_eigensystem(energy, F, eps=1e-8, usv=None):
    """Eigenvalues (t, d^2) and eigenmatrices (t, d^2, d, d) of d^2 psi / dF^2, ordered as
    [scaling modes, twist modes, flip modes].
    Vectorized as a few large operations (all d^2 outer products u_i v_j^T at once), so it runs fast and
    scales across threads; the loop version is kept as _hessian_eigensystem_loop for reference/tests."""
    U, s, V = signed_svd(F) if usv is None else usv
    T, d = s.shape
    I, J = np.triu_indices(d, 1)                                    # the axis pairs (i < j)
    g = energy.grad(s)
    H = energy.hess(s)
    lam_s, e = np.linalg.eigh(H)
    W = np.einsum("tai,tbj->tijab", U, V).reshape(T, d, d, d * d)   # W[t, i, j] = vec(u_i v_j^T)
    diagW = W[:, np.arange(d), np.arange(d)]                        # (T, d, d^2): u_i v_i^T
    Wij, Wji = W[:, I, J], W[:, J, I]
    Q = np.concatenate([np.einsum("tik,tim->tkm", e, diagW),        # scaling: U diag(e_k) V^T
                        (Wij - Wji) / np.sqrt(2),                    # twist
                        (Wij + Wji) / np.sqrt(2)], axis=1)           # flip
    si, sj, gi, gj = s[:, I], s[:, J], g[:, I], g[:, J]
    ds = si - sj
    close = np.abs(ds) < eps
    flip = np.where(close, H[:, I, I] - H[:, I, J], (gi - gj) / np.where(close, 1, ds))
    lam = np.concatenate([lam_s, (gi + gj) / (si + sj), flip], axis=1)
    return lam, Q.reshape(T, d * d, d, d)


def _hessian_eigensystem_loop(energy, F, eps=1e-8, usv=None):
    """Eigenvalues (t, d^2) and eigenmatrices (t, d^2, d, d) of d^2 psi / dF^2, ordered as
    [scaling modes, twist modes, flip modes]."""
    U, s, V = signed_svd(F) if usv is None else usv
    T, d = s.shape
    pairs = list(combinations(range(d), 2))
    g = energy.grad(s)
    H = energy.hess(s)
    lam_s, e = np.linalg.eigh(H)
    lam = np.empty((T, d * d))
    Q = np.empty((T, d * d, d, d))
    lam[:, :d] = lam_s
    Q[:, :d] = np.einsum("tij,tjk,tlj->tkil", U, e, V)        # U diag(e_k) V^T
    Vt = np.swapaxes(V, -1, -2)
    for m, (i, j) in enumerate(pairs):
        A = np.zeros((d, d))
        A[i, j], A[j, i] = 1, -1
        Bm = np.abs(A)
        lam[:, d + m] = (g[:, i] + g[:, j]) / (s[:, i] + s[:, j])                     # twist
        ds = s[:, i] - s[:, j]
        close = np.abs(ds) < eps
        flip = (g[:, i] - g[:, j]) / np.where(close, 1, ds)
        lam[:, d + len(pairs) + m] = np.where(close, H[:, i, i] - H[:, i, j], flip)   # flip
        Q[:, d + m] = U @ A @ Vt / np.sqrt(2)
        Q[:, d + len(pairs) + m] = U @ Bm @ Vt / np.sqrt(2)
    return lam, Q


def projected_hessian(energy, F, mode="clamp", usv=None):
    """PSD version of d^2 psi / dF^2, (t, d^2, d^2) in row-major vec(F).
    mode="clamp": negative eigenvalues -> 0;  mode="abs": negative eigenvalues -> |lambda|;
    mode="none": the true (possibly indefinite) Hessian."""
    lam, Q = hessian_eigensystem(energy, F, usv=usv)
    q = Q.reshape(len(F), Q.shape[1], -1)
    if mode == "abs":
        lam = np.abs(lam)
    elif mode == "clamp":
        lam = np.maximum(lam, 0)
    return np.swapaxes(q, 1, 2) @ (lam[:, :, None] * q)                 # Q^T diag(lam) Q, batched BLAS


class WithVolume:
    """base(s) + kappa (J - 1)^2,  J = prod s_i.  Adds 2 kappa to lambda_Lame, keeps mu_Lame."""

    def __init__(self, base, kappa):
        self.base, self.kappa = base, kappa
        self.name = f"{base.name} + {kappa:g}(J-1)^2"
        self.barrier = getattr(base, "barrier", False)

    def psi(self, s):
        J = np.prod(s, axis=-1)
        return self.base.psi(s) + self.kappa * (J - 1) ** 2

    def grad(self, s):
        J = np.prod(s, axis=-1)[..., None]
        return self.base.grad(s) + 2 * self.kappa * (J - 1) * J / s

    def hess(self, s):
        J = np.prod(s, axis=-1)[..., None, None]
        dJ = J[..., 0] / s                                        # dJ/ds_i
        d2J = J / (s[..., :, None] * s[..., None, :])             # d2J/ds_i ds_j, i != j
        d2J = d2J * (1 - np.eye(s.shape[-1]))
        return self.base.hess(s) + 2 * self.kappa * (dJ[..., :, None] * dJ[..., None, :] + (J - 1) * d2J)


def lame_at_rest(energy, d=3, h=1e-5):
    """(lambda_Lame, mu_Lame) of any isotropic energy from its quadratic truncation at rest:
    lambda = psi_12(1), mu = (psi_11(1) - psi_12(1)) / 2  [Chen & Barbic, SCA 2026]."""
    H = energy.hess(np.ones((1, d)))[0]
    return H[0, 1], 0.5 * (H[0, 0] - H[0, 1])


def youngs_poisson(energy, d=3):
    """(E, nu) at rest from the Lame parameters of the quadratic truncation (3D formulas)."""
    lam, mu = lame_at_rest(energy, d)
    return mu * (3 * lam + 2 * mu) / (lam + mu), lam / (2 * (lam + mu))


class Scaled:
    """c * base. Same minimizers as base (no forces), so scaling only fixes units."""

    def __init__(self, base, c, name=None):
        self.base, self.c = base, c
        self.name = name or base.name
        self.barrier = getattr(base, "barrier", False)

    def psi(self, s):
        return self.c * self.base.psi(s)

    def grad(self, s):
        return self.c * self.base.grad(s)

    def hess(self, s):
        return self.c * self.base.hess(s)


def normalized(base, d=3):
    """base scaled to Young's modulus E = 1 at rest (Poisson's ratio unchanged)."""
    E, _ = youngs_poisson(base, d)
    return Scaled(base, 1.0 / E)


def with_poisson(base, nu, d=3):
    """E = 1 energy with Poisson's ratio nu: a * base + kappa (J - 1)^2 for a lambda = 0 base.
    For a normalized lambda = 0 base (mu = 1/2): a = 1 / (1 + nu), kappa = nu / (2 (1 + nu) (1 - 2 nu))."""
    lam0, mu0 = lame_at_rest(base, d)
    assert abs(lam0) < 1e-9 * max(1.0, abs(mu0)), "base must have lambda = 0"
    mu, lam = 1 / (2 * (1 + nu)), nu / ((1 + nu) * (1 - 2 * nu))
    a = mu / mu0
    return WithVolume(Scaled(base, a), lam / 2)


def kappa_for_poisson(base, nu, d=3):
    """kappa such that base + kappa (J - 1)^2 has Poisson's ratio nu (3D: nu = lambda / (2 (lambda + mu)))."""
    lam0, mu = lame_at_rest(base, d)
    lam = 2 * mu * nu / (1 - 2 * nu)
    return (lam - lam0) / 2
