"""Lean runner for the perpendicular three-target flow (same Euler-Maruyama scheme and RNG stream as
greedy.greedy_trajectory, opt_mode='tissue', mode='exp', log=True) with a numerically stable weight
w_ij = P_ij / S_j computed as a softmax over cells. Where S_j >= core's EPS clamp (1e-10) this is the
identical flow; where core's clamp bites (beta*d^gamma > ~19 for the farthest archetype) the core gradient
is wrong (frozen cells) and this one stays correct, which lets all cases use one common source depth."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
EPS = 1e-10


def archetypes3(e, b=1.0):
    return np.array([[-b, 0.0, 0.0], [b, 0.0, 0.0], [0.0, e, 0.0]])


def source_point(e, h):
    return np.array([0.0, e / 3.0, -h])


def grad_stable(X, A, gamma, beta):
    """(N,D) gradient of sum_j log S_j wrt each cell; identical to core.gradients.tissue_gradient (log=True)."""
    disp = X[:, None, :] - A[None, :, :]                     # (N,M,D)
    d = np.maximum(np.linalg.norm(disp, axis=2), EPS)        # (N,M)
    lp = -beta * d ** gamma
    lp = lp - lp.max(axis=0, keepdims=True)
    w = np.exp(lp); w = w / w.sum(axis=0, keepdims=True)     # P_ij / S_j (softmax over cells)
    coef = w * (-beta * gamma * d ** (gamma - 2.0))
    return np.einsum("ij,ijk->ik", coef, disp)


def simulate(e, gamma, beta, h, seed, N=40, T=800.0, K=40000, sigma=0.01, out_stride=5, r_stop=None,
             early_stop=True):
    """Returns dict of lean time series (stride out_stride) + final positions. Same noise stream as greedy."""
    A = archetypes3(e); src = source_point(e, h)
    rng = np.random.default_rng(seed); dt = T / K; D = 3
    if r_stop is None and gamma < 1.0:
        r_stop = 0.05
    X = np.tile(src, (N, 1))
    snaps = [X.copy()]; ks = [0]
    chk = 2000
    for k in range(K):
        g = grad_stable(X, A, gamma, beta)
        if r_stop is not None:
            fr = (np.linalg.norm(X[:, None, :] - A[None], axis=2).min(1) < r_stop)[:, None]
            g = np.where(fr, 0.0, g)
        step = dt * g + sigma * np.sqrt(dt) * rng.standard_normal((N, D))
        if r_stop is not None:
            step = np.where(fr, 0.0, step)
        X = X + step
        if (k + 1) % out_stride == 0:
            snaps.append(X.copy()); ks.append(k + 1)
        if early_stop and (k + 1) % chk == 0 and k + 1 >= 4 * chk:
            # settled: population stationary over the last chk steps (compare snapshots)
            prev = snaps[-1 - chk // out_stride]
            if np.abs(X - prev).max() < 0.02 and X[:, 2].mean() > -0.15:
                break
    S = np.array(snaps, dtype=np.float32)
    return dict(traj=S, steps=np.array(ks), dt=dt, A=A, src=src, e=e, gamma=gamma, beta=beta, h=h, seed=seed,
                N=N, sigma=sigma, K_run=ks[-1], T_run=ks[-1] * dt)
