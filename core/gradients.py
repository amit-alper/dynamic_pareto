"""
Analytical gradients of the two fitness objectives.

  tissue_gradient  —  grad_{x_i} F_tissue   (collective optimisation)
  cell_gradient    —  grad_{x_i} F_i         (each cell maximises its own score)

Both return (N, D).

Math (exponential mode, log=True, the standard for trajectory work):
  log F = sum_j w_j * log(S_j),   S_j = sum_i P_ij,   P_ij = exp(-beta * d_ij^gamma)

  d/dx_i [w_j * log(S_j)] = (w_j / S_j) * dP_ij/dx_i
  dP_ij/dx_i = P_ij * (-beta * gamma * d_ij^{gamma-2}) * (x_i - a_j)

`beta` is the kernel bandwidth of bandwidth_pareto_flow.tex: it enters the
flow purely through the prefactor gamma -> beta*gamma (the geometric factors
d^{gamma-2}, the (x_i - a_j) direction, and the normalising 1/S_j are
untouched). beta=1 recovers the base model.

Polynomial mode: chain-rule factor is 1 instead of P_ij, and the same
beta*gamma prefactor applies (d/dx max(A - beta d^gamma, 0)).
"""

import numpy as np
from core.performance import pairwise_distances, P_poly, P_exp

EPS = 1e-10


def _get_P(d, gamma, A, mode, beta=1.0):
    if mode == "poly":
        return P_poly(d, gamma, A, beta)
    elif mode == "exp":
        return P_exp(d, gamma, beta)
    raise ValueError(f"Unknown mode '{mode}'. Use 'poly' or 'exp'.")


def _chain_factor(P, mode):
    """Chain-rule factor: 1 for poly (hard cutoff), P for exp (soft)."""
    return np.ones_like(P) if mode == "poly" else P


def tissue_gradient(points, archetypes, gamma=2.0, A=10.0, mode="poly",
                    log=False, weights=None, beta=1.0) -> np.ndarray:
    """
    Gradient of tissue fitness w.r.t. all cell positions.

    weights : (M,) per-archetype weights. If provided, implies weighted
              log-fitness; log flag is ignored (weights require log form).
    beta    : kernel bandwidth; enters as the prefactor gamma -> beta*gamma.
    Returns (N, D).
    """
    distances = pairwise_distances(points, archetypes)
    distances = np.maximum(distances, EPS)

    perf_matrix   = _get_P(distances, gamma, A, mode, beta)
    chain_factors = _chain_factor(perf_matrix, mode)

    task_sums = np.sum(perf_matrix, axis=0)         # (M,)
    task_sums = np.maximum(task_sums, EPS)
    tissue_fitness = np.prod(task_sums)

    radial = -beta * gamma * distances ** (gamma - 2)

    if weights is not None:
        weights_arr = np.asarray(weights, dtype=float)
        if log:
            grad_coefficients = (weights_arr[None, :] / task_sums[None, :]) * chain_factors * radial
        else:
            weighted_fitness  = float(np.prod(task_sums ** weights_arr))
            grad_coefficients = (weighted_fitness * weights_arr[None, :] / task_sums[None, :]) * chain_factors * radial
    elif log:
        grad_coefficients = (1.0 / task_sums[None, :]) * chain_factors * radial
    else:
        grad_coefficients = (tissue_fitness / task_sums[None, :]) * chain_factors * radial

    displacement_vecs = points[:, None, :] - archetypes[None, :, :]   # (N, M, D)
    return np.einsum("ij,ijk->ik", grad_coefficients, displacement_vecs)


def cell_gradient(points, archetypes, gamma=2.0, A=10.0, mode="poly",
                  log=False, beta=1.0) -> np.ndarray:
    """
    Gradient of each cell's individual fitness F_i = prod_j P_ij.
    No coupling between cells — drives cells to the generalist centre.
    beta : kernel bandwidth; enters as the prefactor gamma -> beta*gamma.
    Returns (N, D).
    """
    distances = pairwise_distances(points, archetypes)
    distances = np.maximum(distances, EPS)

    perf_matrix   = _get_P(distances, gamma, A, mode, beta)
    chain_factors = _chain_factor(perf_matrix, mode)
    perf_safe     = np.maximum(perf_matrix, EPS)

    radial = -beta * gamma * distances ** (gamma - 2)

    if log:
        grad_coefficients = (1.0 / perf_safe) * chain_factors * radial
    else:
        cell_fitness      = np.prod(perf_matrix, axis=1)                # (N,)
        grad_coefficients = (cell_fitness[:, None] / perf_safe) * chain_factors * radial

    displacement_vecs = points[:, None, :] - archetypes[None, :, :]
    return np.einsum("ij,ijk->ik", grad_coefficients, displacement_vecs)
