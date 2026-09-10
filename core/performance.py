import numpy as np

EPS = 1e-10


def pairwise_distances(points: np.ndarray, archetypes: np.ndarray) -> np.ndarray:
    """(N, D) x (M, D) -> (N, M) Euclidean distances."""
    return np.linalg.norm(points[:, None, :] - archetypes[None, :, :], axis=2)


def P_poly(distances: np.ndarray, gamma: float = 2.0, A: float = 10.0,
           beta: float = 1.0) -> np.ndarray:
    """Polynomial performance: max(A - beta * d^gamma, 0). Hard zero outside radius.

    beta scales the cost term exactly as in the exponential kernel; beta=1
    recovers the original max(A - d^gamma, 0).
    """
    return np.maximum(A - beta * distances ** gamma, 0.0)


def P_exp(distances: np.ndarray, gamma: float = 2.0, beta: float = 1.0) -> np.ndarray:
    """Bandwidth-controlled exponential performance: exp(-beta * d^gamma).

    This is the kernel of bandwidth_pareto_flow.tex, p(d) = e^{-beta d^gamma}.
    gamma sets the *shape* of the decay (sub-exponential < 1, Gaussian-like at 2);
    beta sets the *scale*: the e-folding radius is beta**(-1/gamma), so larger
    beta gives a narrower, more sharply peaked kernel. beta=1 is the base model.
    """
    return np.exp(-beta * (distances ** gamma))


def performance_matrix(points, archetypes, gamma=2.0, A=10.0, mode="poly",
                       beta=1.0) -> np.ndarray:
    """
    (N, M) performance matrix P[i, j] = performance of cell i on task j.
    mode: "poly" uses P_poly(d, gamma, A, beta); "exp" uses P_exp(d, gamma, beta).
    beta: kernel bandwidth (see P_exp). beta=1 reproduces the base model.
    """
    distances = pairwise_distances(points, archetypes)
    if mode == "poly":
        return P_poly(distances, gamma, A, beta)
    elif mode == "exp":
        return P_exp(distances, gamma, beta)
    else:
        raise ValueError(f"Unknown performance mode '{mode}'. Use 'poly' or 'exp'.")


def local_performance(points, archetypes, gamma=2.0, A=10.0, mode="poly",
                      log=False, beta=1.0) -> np.ndarray:
    """
    Per-cell performance: F_i = prod_j P[i,j]  (or sum of logs).
    Returns (N,) array.
    """
    perf_matrix = performance_matrix(points, archetypes, gamma, A, mode, beta)
    if log:
        return np.sum(np.log(perf_matrix + EPS), axis=1)
    return np.prod(perf_matrix, axis=1)


def tissue_performance(points, archetypes, gamma=2.0, A=10.0, mode="poly",
                       log=False, weights=None, beta=1.0) -> float:
    """
    Collective tissue performance: F = prod_j S_j  where S_j = sum_i P[i,j].

    With weights (used when tasks have unequal importance):
      log F = sum_j w_j * log(S_j)

    weights: (M,) array of per-archetype weights. Default: all ones.
    beta: kernel bandwidth (see P_exp). beta=1 reproduces the base model.
    log=True is required when using weights (weighted log-sum form).
    """
    perf_matrix = performance_matrix(points, archetypes, gamma, A, mode, beta)
    task_sums   = np.sum(perf_matrix, axis=0)      # (M,) column sums
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if log:
            return float(np.sum(weights * np.log(task_sums + EPS)))
        return float(np.prod(task_sums ** weights))
    if log:
        return float(np.sum(np.log(task_sums + EPS)))
    return float(np.prod(task_sums))
