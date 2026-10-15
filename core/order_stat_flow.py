"""Order-statistics particle model of the early (large-gamma) split.  New module; existing code untouched.

Idea. Noise only has to create the initial lateral offsets; the collective flow then amplifies them. While
the offsets are small the cloud is Gaussian with the variance V(t) of the linear theory
(core.bifurcation.theoretical_bifurcation_point). At a seed time t0, where sqrt(V) is still in the linear
regime, the N cells are placed at the expected order statistics of N draws from N(0, V(t0)) (Blom quantiles,
rescaled to sample std sqrt(V), mirror symmetric), all at the coherent-path position. From t0 the *exact
deterministic N-cell tissue flow* is integrated. Unlike a Gaussian closure the tail cells (the first to
leave) are kept as individual particles, so the nonlinear escape, the symmetry of the leaving pairs and the
slowed descent of the remaining bulk come out of the flow itself. Detection uses the same rule as the
measurement: first step at which std over cells of the lateral coordinate reaches x_partial; height = mean
height of the cells at that step.
"""
import numpy as np
from scipy.stats import norm

from core.bifurcation import theoretical_bifurcation_point
from core.landscape import landscape_gradient


def order_statistic_offsets(N, sd):
    """(N,) expected order statistics of N draws from N(0, sd^2): Blom quantiles, mirror symmetric,
    rescaled so the (1/N) sample std equals sd."""
    q = norm.ppf((np.arange(1, N + 1) - 0.375) / (N + 0.25))
    q = 0.5 * (q - q[::-1])                      # exact mirror symmetry
    return q / np.sqrt(np.mean(q ** 2)) * sd


def order_statistic_flow(source, archetypes, gamma, beta=1.0, pop_N=40, T=800.0, K=40000, sigma=0.01,
                         x_partial=0.20, seed_std=0.02, lateral_dim=0, height_dim=-1, direction=None):
    """Returns dict(t, height, t0, times, lateral_std, height_trace, positions_final); t/height None if no crossing."""
    source = np.asarray(source, float); archetypes = np.asarray(archetypes, float)
    th = theoretical_bifurcation_point(source, archetypes, gamma, beta=beta, N=pop_N, T=T, K=K, sigma=sigma,
                                       x_partial=x_partial, height_dim=height_dim, direction=direction)
    sdV = np.sqrt(np.maximum(th["V"], 0.0)); dt = T / K
    i0 = int(np.argmax(sdV >= seed_std)) if sdV.max() >= seed_std else None
    if i0 is None:
        return dict(t=None, height=None, t0=None)
    P = np.tile(th["positions"][i0], (pop_N, 1))
    P[:, lateral_dim] += order_statistic_offsets(pop_N, sdV[i0])
    w = np.ones(len(archetypes)); times, stds, hts = [], [], []
    t_det = h_det = None
    for k in range(i0, K):
        s = float(P[:, lateral_dim].std()); h = float(P[:, height_dim].mean())
        times.append(k * dt); stds.append(s); hts.append(h)
        if s >= x_partial:
            t_det, h_det = k * dt, h
            break
        P = P + dt * landscape_gradient(P, archetypes, gamma, w, None, "exp", 10.0, True, beta)
    return dict(t=t_det, height=h_det, t0=i0 * dt, times=np.array(times), lateral_std=np.array(stds),
                height_trace=np.array(hts), positions_final=P)


def _cov_step_exact(Sigma, L, sigma, dt):
    """Exact step of dSigma/dt = L Sigma + Sigma L^T + sigma^2 I for symmetric L held constant over dt
    (in L's eigenbasis: S_ij -> e^{(l_i+l_j)dt} S_ij, plus sigma^2 (e^{2 l_i dt}-1)/(2 l_i) on the diagonal).
    Same exponential-step philosophy as core.bifurcation.exact_variance, which this reduces to for one mode."""
    lam, V = np.linalg.eigh(0.5 * (L + L.T))
    S = V.T @ Sigma @ V
    S = S * np.exp((lam[:, None] + lam[None, :]) * dt)
    x = 2.0 * lam * dt
    inj = np.where(np.abs(x) < 1e-12, dt, np.expm1(np.clip(x, -700, 700)) / np.where(lam == 0, 1.0, 2.0 * lam))
    S = S + np.diag(sigma ** 2 * inj)
    return V @ S @ V.T


def full_covariance_flow(source, archetypes, gamma, beta=1.0, pop_N=40, T=800.0, K=40000, sigma=0.01,
                         x_partial=0.20, seed_std=0.02, lateral_dim=0, height_dim=-1, n_real=16,
                         noise=False, rng_seed=0, trace_until=None, keep_mean=False):
    """Like order_statistic_flow, but (i) seeds the *full* D x D covariance, obtained from
    dSigma/dt = L Sigma + Sigma L^T + sigma^2 I along the coherent path (L = transverse_operator, so the
    axial/stretch mode and the other in-plane direction are included, not just x), and (ii) draws n_real
    random Gaussian realisations of the N-cell cloud from it instead of one quantile ensemble, so the result
    is a mean +- sd directly comparable with the seed-averaged measurement. From the seed time each
    realisation follows the N-cell tissue flow (deterministically, or with the SDE noise if noise=True).
    Returns dict(heights, times, t0, height_mean, height_sd, t_mean, n_detected)."""
    from core.bifurcation import integrate_coherent_path
    from core.stability import transverse_operator
    source = np.asarray(source, float); archetypes = np.asarray(archetypes, float)
    D = source.shape[0]; dt = T / K
    times, pos = integrate_coherent_path(source, archetypes, gamma, beta, pop_N, T, K)
    Sigma = np.zeros((D, D)); i0 = None
    for k in range(K):
        L = transverse_operator(pos[k], archetypes, gamma, beta, pop_N)
        Sigma = _cov_step_exact(Sigma, L, sigma, dt)
        if np.sqrt(np.max(np.diag(Sigma))) >= seed_std:
            i0 = k + 1; break
    if i0 is None:
        return dict(heights=np.array([]), times=np.array([]), n_detected=0)
    w, V = np.linalg.eigh(Sigma); Lc = V * np.sqrt(np.maximum(w, 0.0))
    rng = np.random.default_rng(rng_seed); wts = np.ones(len(archetypes))
    heights, tdets, traces = [], [], []
    for _ in range(n_real):
        z = rng.standard_normal((pop_N, D))
        if not keep_mean:
            z -= z.mean(axis=0)               # keep_mean=True retains the cloud's random centre offset
        P = pos[i0][None, :] + z @ Lc.T
        done = False; th_, ts_ = [], []
        for k in range(i0, K):
            s_ = P[:, lateral_dim].std(); h_ = P[:, height_dim].mean()
            th_.append(h_); ts_.append(s_)
            if not done and s_ >= x_partial:
                heights.append(h_); tdets.append(k * dt); done = True
                if trace_until is None:
                    break
            if trace_until is not None and s_ >= trace_until:
                break
            P = P + dt * landscape_gradient(P, archetypes, gamma, wts, None, "exp", 10.0, True, beta)
            if noise:
                P = P + sigma * np.sqrt(dt) * rng.standard_normal(P.shape)
        traces.append((np.array(th_), np.array(ts_)))
    h = np.array(heights)
    return dict(heights=h, times=np.array(tdets), t0=i0 * dt, n_detected=len(h), traces=traces,
                height_mean=float(h.mean()) if len(h) else np.nan, height_sd=float(h.std()) if len(h) else np.nan,
                t_mean=float(np.mean(tdets)) if tdets else np.nan)
