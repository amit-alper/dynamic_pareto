"""
Greedy baseline trajectory.

Cells follow the local landscape gradient at each step with optional
Langevin noise -- no speed or smoothness constraints.

opt_mode="tissue" (default): collective gradient, drives specialisation.
opt_mode="cell": each cell follows its own gradient, drives generalisation.

mode="exp" by default (matches trajectory solvers).
mode="poly" supported for comparison experiments.
"""

import numpy as np
from core.landscape import (
    landscape_gradient, landscape_value,
)
from core.gradients import cell_gradient
from core.performance import tissue_performance


def _noise_std(sigma: float, step: int, schedule: str, decay_rate: float) -> float:
    if schedule == "constant":
        return sigma
    elif schedule == "decay":
        return sigma * np.exp(-decay_rate * step)
    raise ValueError(f"Unknown noise schedule '{schedule}'. Use 'constant' or 'decay'.")


def greedy_trajectory(source, archetypes, gamma, N, K, T,
                      weights=None, barrier=None, sigma=0.005,
                      seed=42, mode="exp", A=None, log=True,
                      r_stop=None, opt_mode="tissue",
                      noise_schedule="constant", decay_rate=0.01,
                      beta=1.0):
    """
    Parameters
    ----------
    beta : float
        Kernel bandwidth of bandwidth_pareto_flow.tex: P_ij = exp(-beta d^gamma).
        Enters the flow as the prefactor gamma -> beta*gamma and inside every
        stability bracket as gamma d^gamma -> beta*gamma d^gamma. beta=1 is the
        base model; beta < 1 broadens the kernel, beta > 1 narrows it.
    opt_mode : "tissue" | "cell"
        "tissue" -- collective gradient grad log F_tissue (drives specialisation).
        "cell"   -- each cell independently follows grad F_i (drives generalisation).
    noise_schedule : "constant" | "decay"
        "constant" -- fixed sigma throughout.
        "decay"    -- sigma(t) = sigma * exp(-decay_rate * t).
    decay_rate : float
        Decay constant for the "decay" noise schedule.
    mode : "exp" | "poly"
        Performance function. "exp" matches the constrained solvers.
        "poly" available for comparison studies.
    A : float or None
        Polynomial offset for mode="poly". Auto-computed from source-archetype
        distances when None, ensuring A > max(d^gamma) everywhere.
    r_stop : float or None
        Radius around each archetype where cells freeze. Prevents gradient
        explosions when gamma < 1 causes |grad f| -> inf near targets.

    Returns
    -------
    dict with keys:
        trajectory, history, fates, final_dist, performance, integrated_perf,
        max_speed, mean_speed, max_acc, mean_acc,
        speed_profile, acc_profile, sigma_history, converged,
        lateral_var  -- (K+1,) Var_j[x_j] at each step (splitting observable),
        axial_var    -- (K+1,) Var_j[y_j] at each step (stretch observable),
        y_mean       -- (K+1,) mean_j[y_j] at each step (maps steps → theory height)
    """
    if opt_mode not in ("tissue", "cell"):
        raise ValueError(f"Unknown opt_mode '{opt_mode}'. Use 'tissue' or 'cell'.")

    if mode == "poly" and A is None:
        dists = np.linalg.norm(archetypes - source[None, :], axis=1)
        A = max(float(np.max(dists) ** gamma * 3.0), 1.0)
    elif A is None:
        A = 10.0

    rng = np.random.default_rng(seed)
    D = source.shape[0]; dt = T / K; n_archetypes = len(archetypes)

    # Euler stability diagnostics at the source (2-archetype symmetric axis).
    # Two rates compete: axial drift |ẏ| and lateral splitting λ_split.
    # With the bandwidth kernel these carry the prefactor gamma -> beta*gamma
    # and phi_beta(u) = beta*gamma*u^{γ/2} − u − (γ−2)  (bandwidth_pareto_flow.tex).
    # Axial:   |ẏ| = 2βγ|h|·d^{γ-2}/N              safe when dt·|ẏ|      < 0.1
    # Lateral: λ_split = (2βγ/N)·d^{γ-4}·φ_β(u)    safe when dt·λ_split  < 1.0
    #          φ_β(u) = βγu^{γ/2} − u − (γ−2),  u = 1+h²  (positive ↔ splitting region)
    # Recommended K targets: dt·axial < 0.05,  dt·λ_split < 0.5
    _d_source = float(np.linalg.norm(source))
    if _d_source > 0 and D >= 2:
        _h            = float(abs(source[1]))
        _d_axis       = float(np.sqrt(1.0 + _h ** 2))
        _u            = _d_axis ** 2                          # 1 + h²
        _d_gamma      = _d_axis ** gamma                      # d^γ  (= u^{γ/2})
        _axial_rate   = 2.0 * beta * gamma * _h * _d_axis ** (gamma - 2.0) / N
        _phi          = beta * gamma * _d_gamma - (_u + gamma - 2.0)
        _lambda_split = max(0.0, (2.0 * beta * gamma / N) * _d_axis ** (gamma - 4.0) * _phi)

        _K_axial   = int(np.ceil(T * _axial_rate   / 0.05)) if _axial_rate   > 0 else K
        _K_lateral = int(np.ceil(T * _lambda_split / 0.50)) if _lambda_split > 0 else K
        _K_rec     = max(_K_axial, _K_lateral)

        _warn_axial   = _axial_rate   * dt > 0.1
        _warn_lateral = _lambda_split * dt > 1.0

        if _warn_axial or _warn_lateral:
            print(f"[greedy] WARNING (gamma={gamma}, h={_h:.1f}, N={N}, K={K}):")
        if _warn_axial:
            print(f"  axial   dt*|ydot|    = {_axial_rate * dt:.2f} > 0.1  "
                  f"(Euler step coarse along y)")
        if _warn_lateral:
            print(f"  lateral dt*lam_split = {_lambda_split * dt:.2f} > 1.0  "
                  f"(Euler step unstable laterally -- cells may freeze or diverge at source)")
        if _K_rec > K:
            print(f"  -> recommended  K >= {_K_rec},  dt <= {T / _K_rec:.5f}")
    if weights is None:
        weights = np.ones(n_archetypes)

    trajectory = np.zeros((K + 1, N, D))
    trajectory[0] = source[None, :]
    sigma_history = []

    for k in range(K):
        if r_stop is not None:
            min_dists = np.min(
                np.linalg.norm(trajectory[k, :, None, :] - archetypes[None, :, :], axis=2), axis=1
            )
            frozen = (min_dists < r_stop)[:, None]
        else:
            frozen = np.zeros((N, 1), dtype=bool)

        if opt_mode == "tissue":
            grad = landscape_gradient(trajectory[k], archetypes, gamma, weights, barrier, mode, A, log, beta)
        else:
            grad = cell_gradient(trajectory[k], archetypes, gamma=gamma, A=A, mode=mode, log=log, beta=beta)

        grad = np.where(frozen, 0.0, grad)
        step = dt * grad
        step = np.where(frozen, 0.0, step)

        noise_std = _noise_std(sigma, k, noise_schedule, decay_rate)
        sigma_history.append(noise_std)
        step += np.where(frozen, 0.0, noise_std * np.sqrt(dt) * rng.standard_normal((N, D)))

        trajectory[k + 1] = trajectory[k] + step

    steps = trajectory[1:] - trajectory[:-1]
    step_norms = np.linalg.norm(steps, axis=2)
    vel = steps / dt

    # Mode-amplitude tracking (cheap, always included)
    # lateral_var: Var_j[x_j] — the splitting observable; theory predicts growth via λ_split
    # axial_var:   Var_j[y_j] — the stretch observable; grows when λ_stretch > 0
    # y_mean:      mean height — converts step index → y for theory comparison
    lateral_var = np.var(trajectory[:, :, 0], axis=1)   # (K+1,)
    axial_var   = np.var(trajectory[:, :, 1], axis=1)   # (K+1,)
    y_mean      = np.mean(trajectory[:, :, 1], axis=1)  # (K+1,)

    fates, final_dists = [], []
    for j in range(N):
        archetype_dists = [np.linalg.norm(trajectory[-1, j] - archetypes[i]) for i in range(n_archetypes)]
        fates.append(np.argmin(archetype_dists)); final_dists.append(min(archetype_dists))

    acc_norms = None
    if K >= 2:
        acc = np.diff(vel, axis=0) / dt
        acc_norms = np.linalg.norm(acc, axis=2)

    return {
        'trajectory':      trajectory,
        'history':         list(trajectory),
        'fates':           np.array(fates),
        'final_dist':      np.array(final_dists),
        'performance':     tissue_performance(trajectory[-1], archetypes, gamma,
                                              A=A, mode=mode, log=True, weights=weights, beta=beta),
        'integrated_perf': sum(landscape_value(trajectory[k], archetypes, gamma, weights, barrier, mode, A,
                                               log=False, beta=beta) * dt
                               for k in range(K)),
        'max_speed':       np.max(step_norms) / dt,
        'mean_speed':      np.mean(step_norms) / dt,
        'max_acc':         np.max(acc_norms) if acc_norms is not None else 0,
        'mean_acc':        np.mean(acc_norms) if acc_norms is not None else 0,
        'speed_profile':   np.mean(step_norms / dt, axis=1),
        'acc_profile':     np.mean(acc_norms, axis=1) if acc_norms is not None else None,
        'sigma_history':   sigma_history,
        'converged':       True,
        'lateral_var':     lateral_var,
        'axial_var':       axial_var,
        'y_mean':          y_mean,
    }