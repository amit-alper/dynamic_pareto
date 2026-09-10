"""
Landscape layer for trajectory optimisation.

Wraps core performance/gradient functions and adds optional elliptical
Gaussian barriers (obstacles in gene-expression space).

landscape_value    -- scalar fitness (used in objectives)
landscape_gradient -- (N, D) gradient (used in solvers)

The trajectory solvers always use mode="exp".
greedy_trajectory accepts mode as a parameter.
"""

import numpy as np
from core.performance import tissue_performance
from core.gradients import tissue_gradient


# ── Barrier (Gaussian obstacle) ───────────────────────────────────────────────

def make_barrier(strength=0, cx=0.6, cy=0.85, wx=0.25, wy=0.3):
    """Return barrier dict, or None if strength <= 0."""
    if strength <= 0:
        return None
    return {'cx': cx, 'cy': cy, 'wx': wx, 'wy': wy, 'strength': strength}


def barrier_value(positions, barrier):
    """Gaussian penalty at each cell position. Returns (N,)."""
    if barrier is None:
        return np.zeros(positions.shape[0])
    dx = (positions[:, 0] - barrier['cx']) / barrier['wx']
    dy = (positions[:, 1] - barrier['cy']) / barrier['wy']
    return barrier['strength'] * np.exp(-0.5 * (dx**2 + dy**2))


def barrier_gradient(positions, barrier):
    """Gradient of barrier penalty. Returns (N, D)."""
    if barrier is None:
        return np.zeros_like(positions)
    dx = (positions[:, 0] - barrier['cx']) / barrier['wx']
    dy = (positions[:, 1] - barrier['cy']) / barrier['wy']
    val = barrier['strength'] * np.exp(-0.5 * (dx**2 + dy**2))
    grad = np.zeros_like(positions)
    grad[:, 0] = -val * dx / barrier['wx']
    grad[:, 1] = -val * dy / barrier['wy']
    return grad


# ── Combined landscape ────────────────────────────────────────────────────────

def landscape_value(positions, archetypes, gamma, weights=None,
                    barrier=None, mode="exp", A=10.0, log=False, beta=1.0) -> float:
    """
    F(positions) [or log F if log=True] minus barrier penalty.

    log=True gives ∫ log F dt objective — better conditioned gradients,
    no vanishing signal far from archetypes.
    mode="exp"  for all constrained/penalty trajectory solvers.
    mode="poly" available for greedy comparisons.
    weights: (M,) per-archetype importance weights.
    beta: kernel bandwidth, p(d) = e^{-beta d^gamma}. beta=1 is the base model.
    A: polynomial offset for mode="poly". Caller should ensure A > max(d^gamma).
    """
    val = tissue_performance(positions, archetypes, gamma=gamma, A=A,
                             mode=mode, log=log, weights=weights, beta=beta)
    if barrier is not None:
        val -= float(np.sum(barrier_value(positions, barrier)))
    return val


def landscape_gradient(positions, archetypes, gamma, weights=None,
                        barrier=None, mode="exp", A=10.0, log=False, beta=1.0) -> np.ndarray:
    """
    Gradient of (log F - barrier) w.r.t. all cell positions. Returns (N, D).
    beta: kernel bandwidth, p(d) = e^{-beta d^gamma}. beta=1 is the base model.
    A: polynomial offset for mode="poly". Caller should ensure A > max(d^gamma).
    log=False uses gradient of F directly (much larger with poly mode).
    """
    grad = tissue_gradient(positions, archetypes, gamma=gamma, A=A,
                           mode=mode, log=log, weights=weights, beta=beta)
    if barrier is not None:
        grad -= barrier_gradient(positions, barrier)
    return grad
