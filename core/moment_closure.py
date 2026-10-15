"""Gaussian moment-closure bifurcation prediction (new module; core/bifurcation.py is untouched)."""
import numpy as np


def _whitened_ensemble(N, D, seed=0):
    """(N, D) deterministic sample with exactly zero mean and identity covariance
    (sample covariance, 1/N normalisation), antithetic by construction."""
    rng = np.random.default_rng(seed)
    half = rng.standard_normal((N // 2, D))
    z = np.concatenate([half, -half], axis=0)
    if N % 2:
        z = np.concatenate([z, np.zeros((1, D))], axis=0)
    cov = z.T @ z / N
    return z @ np.linalg.inv(np.linalg.cholesky(cov)).T


def moment_closure_path(source, archetypes, gamma, beta=1.0, pop_N=40,
                         T=800.0, K=40000, sigma=0.01, x_partial=0.20,
                         lateral_dim=0, height_dim=-1, jitter_seed=0, stop_at_detection=True):
    """Gaussian moment closure of the stochastic N-cell flow (mean m, covariance Sigma).

    The cloud is summarised by m(t) and Sigma(t). At every step the pop_N cells are
    re-drawn as m + L z_j (L = chol Sigma, z_j a fixed whitened ensemble) and the
    *true* N-cell tissue gradient g_j is evaluated on them -- no co-location
    assumption, so the cloud feels the actual cross-cell weights P_ij/S_j. The
    moments then follow from Stein's lemma (statistical linearisation):
        dm/dt     = <g_j>
        dSigma/dt = Cov(g,x) + Cov(x,g)^T + sigma^2 I
    (Cov(g,x) = <g_j (x_j-m)^T> = Jbar Sigma for a Gaussian cloud). Same Euler
    step dt = T/K as greedy_trajectory. This replaces the coherent-path height
    and the single-mode V ODE by one closed system for the height and the full
    D x D lateral covariance; it reduces to dV/dt = 2 lambda V + sigma^2 along the
    coherent path when Sigma -> 0.

    Detection mirrors the empirical rule: first step with sqrt(Sigma[lat,lat]) >=
    x_partial; height = m[height_dim] there. Stops there by default.
    Returns dict(times, mean (k,D), lateral_std (k,), t, height); t/height None if no crossing.
    """
    from core.landscape import landscape_gradient
    source = np.asarray(source, float); archetypes = np.asarray(archetypes, float)
    D = source.shape[0]; dt = T / K
    z = _whitened_ensemble(pop_N, D, jitter_seed)
    w = np.ones(len(archetypes))
    m = source.copy(); Sigma = np.zeros((D, D))
    means, stds = [m.copy()], [0.0]
    t_det = h_det = None
    for k in range(K):
        L = np.linalg.cholesky(Sigma + 1e-300 * np.eye(D)) if Sigma.any() else np.zeros((D, D))
        pts = m[None, :] + z @ L.T
        g = landscape_gradient(pts, archetypes, gamma, w, None, "exp", 10.0, True, beta)
        dx = pts - m[None, :]
        gc = g - g.mean(axis=0, keepdims=True)
        C = gc.T @ dx / pop_N                                   # Cov(g, x)
        m = m + dt * g.mean(axis=0)
        Sigma = Sigma + dt * (C + C.T + sigma ** 2 * np.eye(D))
        Sigma = 0.5 * (Sigma + Sigma.T)
        w_, V_ = np.linalg.eigh(Sigma)                           # keep PSD against Euler error
        if w_[0] < 0:
            Sigma = (V_ * np.maximum(w_, 0.0)) @ V_.T
        means.append(m.copy()); stds.append(float(np.sqrt(max(Sigma[lateral_dim, lateral_dim], 0.0))))
        if t_det is None and stds[-1] >= x_partial:
            t_det, h_det = (k + 1) * dt, float(m[height_dim])
            if stop_at_detection:
                break
    return {"times": np.arange(len(stds)) * dt, "mean": np.array(means),
            "lateral_std": np.array(stds), "t": t_det, "height": h_det}
