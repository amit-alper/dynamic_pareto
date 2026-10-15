"""
Geometry-agnostic static stability analysis for the collective Pareto flow.

Generalizes the hand-derived, per-geometry transverse eigenvalues in
bandwidth_pareto_flow.tex (two-target axial lambda_split(y), Sec. 4) and
perpendicular_common.py (three-target lambda_x(z)) into a single operator
that works for any archetype set, any ambient dimension, and any point --
not just points on a pre-identified symmetry axis.

Model recap (bandwidth_pareto_flow.tex, Sec. "Transverse stability"):
For N cells co-located at a point x, with archetypes a_i, d_i = ||x - a_i||,
r_hat_i = (x - a_i) / d_i, the Jacobian of the collective flow restricted to
zero-sum (symmetry-breaking) perturbations is

    Lambda_split(x) = M(x) - tau(x) * I,

    M(x)   = (beta*gamma/N) * sum_i d_i^(gamma-2) * (beta*gamma*d_i^gamma - gamma + 2) * r_hat_i r_hat_i^T
    tau(x) = (beta*gamma/N) * sum_i d_i^(gamma-2)

and the directional eigenvalue along a unit vector e is

    lambda(e) = e^T Lambda_split(x) e
              = (beta*gamma/N) * sum_i d_i^(gamma-2) * [(beta*gamma*d_i^gamma - gamma+2)*(e.r_hat_i)^2 - 1].

At N=1 this instead governs a single, uncoupled cell's own trajectory
(Proposition "single-cell stability tensor"):

    J1(x) = -beta*gamma * sum_i d_i^(gamma-2) * [I + (gamma-2) * r_hat_i r_hat_i^T].

Validated against the closed forms in bandwidth_pareto_flow.tex (two-target
axial) and perpendicular_common.py (perpendicular three-target) by
tests/test_stability_bifurcation.py.
"""

import numpy as np

EPS = 1e-12


def _geometry(x, archetypes):
    """x: (D,), archetypes: (M, D) -> distances (M,), unit vectors (M, D)."""
    disp = x[None, :] - archetypes          # (M, D), points FROM archetype TO x
    d = np.linalg.norm(disp, axis=1)
    d_safe = np.maximum(d, EPS)
    rhat = disp / d_safe[:, None]
    return d, rhat


def transverse_operator(x, archetypes, gamma, beta=1.0, N=1.0):
    """Lambda_split(x): the (D, D) symmetry-breaking stability operator for N
    co-located cells at x. Eigenvalues > 0 mark unstable (splitting) directions.
    """
    x = np.asarray(x, dtype=float)
    archetypes = np.asarray(archetypes, dtype=float)
    D = x.shape[0]
    d, rhat = _geometry(x, archetypes)

    coeff = (beta * gamma / N) * d ** (gamma - 2.0)              # (M,)
    scale = beta * gamma * d ** gamma - gamma + 2.0              # (M,)

    M = np.einsum("i,i,ij,ik->jk", coeff, scale, rhat, rhat)     # (D, D)
    tau = float(np.sum(coeff))
    return M - tau * np.eye(D)


def directional_eigenvalue(x, archetypes, gamma, direction, beta=1.0, N=1.0):
    """lambda(e) = e^T Lambda_split(x) e for a unit vector e, computed directly
    from the closed-form sum (eq. lambda_dir_beta) rather than via the full
    matrix -- the same quantity, useful for validating transverse_operator and
    for cases where only one direction (e.g. e_x) is of interest.
    """
    x = np.asarray(x, dtype=float)
    archetypes = np.asarray(archetypes, dtype=float)
    e = np.asarray(direction, dtype=float)
    e = e / np.linalg.norm(e)
    d, rhat = _geometry(x, archetypes)

    proj2 = (rhat @ e) ** 2                                      # (M,) (e . r_hat_i)^2
    bracket = (beta * gamma * d ** gamma - gamma + 2.0) * proj2 - 1.0
    return float((beta * gamma / N) * np.sum(d ** (gamma - 2.0) * bracket))


def transverse_eigen(x, archetypes, gamma, beta=1.0, N=1.0):
    """Full eigendecomposition of Lambda_split(x), sorted by descending eigenvalue.

    Returns (eigvals, eigvecs) with eigvals[0] the most unstable direction and
    eigvecs[:, 0] the corresponding unit direction.
    """
    L = transverse_operator(x, archetypes, gamma, beta, N)
    vals, vecs = np.linalg.eigh(L)                # symmetric -> real, ascending
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def restricted_eigen(matrix, exclude_directions):
    """Eigendecomposition of `matrix` restricted to the orthogonal complement
    of `exclude_directions` (a (k, D) array of vectors, not necessarily unit
    or orthogonal). Used to isolate the "transverse" eigenvalue orthogonal to
    a known direction of travel, in dimensions where the travel direction is
    itself an eigenvector of `matrix` (true at every symmetric-axis point of
    the geometries this project uses) but is not the quantity of interest.

    Returns (eigvals desc, eigvecs as (D, D-k) columns in the ORIGINAL basis).
    """
    matrix = np.asarray(matrix, dtype=float)
    D = matrix.shape[0]
    exclude = np.atleast_2d(np.asarray(exclude_directions, dtype=float))

    # Orthonormal basis of the complement via QR on [exclude^T | I], dropping
    # the columns that reproduce the excluded subspace.
    Q, _ = np.linalg.qr(np.concatenate([exclude.T, np.eye(D)], axis=1))
    k = np.linalg.matrix_rank(exclude)
    E = Q[:, k:]                                    # (D, D-k) orthonormal complement

    B = E.T @ matrix @ E                             # (D-k, D-k) restricted operator
    vals, vecs_local = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = E @ vecs_local[:, order]                  # back to original D-space
    return vals, vecs


def transverse_eigenvalue_excluding(x, archetypes, gamma, exclude_direction, beta=1.0, N=1.0):
    """Largest eigenvalue of Lambda_split(x) restricted to directions orthogonal
    to `exclude_direction` (typically the local direction of travel). This is
    the general-geometry analogue of lambda_split(y) / lambda_x(z): "how
    unstable is the population to a perturbation transverse to its own path."
    """
    L = transverse_operator(x, archetypes, gamma, beta, N)
    vals, vecs = restricted_eigen(L, exclude_direction)
    return float(vals[0]), vecs[:, 0]


# ── Single-cell (N=1) stability, for the individual/collective dichotomy ──────

def single_cell_operator(x, archetypes, gamma, beta=1.0):
    """J1(x): Hessian of an isolated cell's log-fitness (Proposition
    single_cell_stability). Negative eigenvalues everywhere = the shared
    compromise point is a global attractor for a lone cell (gamma >= 1);
    a positive eigenvalue means an isolated cell would itself drift off-axis.
    """
    x = np.asarray(x, dtype=float)
    archetypes = np.asarray(archetypes, dtype=float)
    D = x.shape[0]
    d, rhat = _geometry(x, archetypes)

    coeff = beta * gamma * d ** (gamma - 2.0)                    # (M,)
    inner = np.eye(D)[None, :, :] + (gamma - 2.0) * np.einsum("ij,ik->ijk", rhat, rhat)
    return -np.einsum("i,ijk->jk", coeff, inner)


def single_cell_eigen(x, archetypes, gamma, beta=1.0):
    vals, vecs = np.linalg.eigh(single_cell_operator(x, archetypes, gamma, beta))
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]
