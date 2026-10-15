"""
Dynamic bifurcation-point quantification, geometry-agnostic.

Connects two things that this project had already independently derived per
geometry (two-target axial closed forms in bandwidth_pareto_flow.tex; the
perpendicular three-target lambda_x(z)/Gamma_total in
perpendicular_common.py / phase_portrait.py) to the general operator in
core/stability.py, and ties the resulting *theoretical* bifurcation point to
an *empirical* one measured directly from noisy greedy_trajectory runs.

The Bass-Burdzy correspondence (see conversation / bandwidth_pareto_flow.tex
for the static side): their switching-ODE bifurcation time T* = sup{t: X_t=B_t}
is driven by an accumulated *local time* L_t. This system's population is not
switching but linearly unstable near the symmetric axis, so the transverse
coordinate obeys (after linearizing) an Ornstein-Uhlenbeck-type SDE
    dx = lambda(t) x dt + sigma dW,
whose variance grows as Var[x(t)] ~ sigma^2 * exp(2*Gamma(t)) with
    Gamma(t) = integral_0^t lambda(s) ds.
Gamma(t) is therefore this model's direct analogue of Bass-Burdzy's
accumulated local time: it is the log-amplification factor of the
symmetry-breaking noise, and a first-passage of Gamma through a detection
threshold is this model's analogue of their bifurcation time T*.

Two bifurcation-point estimators are provided:
  - "sign" (generalizes y_c / z_c): first time lambda_max(t) becomes
    positive and stays positive -- the classical, purely local criterion.
  - "visible" (generalizes Gamma_total / the white contour in
    phase_portrait.py): first time the accumulated growth Gamma(t) reaches
    ln(x_partial / sigma), i.e. a symmetry-breaking seed of size sigma has
    grown enough to be called "split" by the same threshold the rest of the
    codebase uses (perpendicular_common.X_PARTIAL / SIGMA_DEFAULT).
Both are computed along the *coherent* (noise-free, all-cells-co-located)
trajectory, then checked against the *empirical* split height measured from
actual stochastic greedy_trajectory runs.
"""

import contextlib
import io

import numpy as np

from core.stability import transverse_operator, restricted_eigen, directional_eigenvalue
from greedy.greedy import greedy_trajectory

EPS = 1e-12


# ── Coherent (noise-free, co-located) trajectory ──────────────────────────────

def coherent_velocity(x, archetypes, gamma, beta=1.0, N=1.0):
    """dot(p) for N co-located cells at p=x (Eq. coherent_velocity): the exact
    N-cell tissue gradient collapses to this closed form once every cell sits
    at the same point, since P_ij/S_j = 1/N identically.
    """
    x = np.asarray(x, dtype=float)
    archetypes = np.asarray(archetypes, dtype=float)
    disp = x[None, :] - archetypes                       # (M, D)
    d = np.maximum(np.linalg.norm(disp, axis=1), EPS)     # (M,)
    coeff = (beta * gamma / N) * d ** (gamma - 2.0)
    return -np.einsum("i,ij->j", coeff, disp)


def integrate_coherent_path(source, archetypes, gamma, beta=1.0, N=1.0,
                             T=800.0, K=40000):
    """Explicit-Euler integration of the deterministic coherent flow, same
    (dt = T/K) discretization convention as greedy_trajectory so theoretical
    and empirical bifurcation times are directly comparable.

    Returns times (K+1,), positions (K+1, D).
    """
    source = np.asarray(source, dtype=float)
    dt = T / K
    D = source.shape[0]
    positions = np.empty((K + 1, D))
    positions[0] = source
    for k in range(K):
        v = coherent_velocity(positions[k], archetypes, gamma, beta, N)
        positions[k + 1] = positions[k] + dt * v
    times = np.arange(K + 1) * dt
    return times, positions


# ── Eigenvalue and accumulated-growth profile along a path ───────────────────

def eigen_along_path(times, positions, archetypes, gamma, beta=1.0, N=1.0, direction=None):
    """At each point of a (coherent) path, the transverse eigenvalue, and the
    local speed.

    By default (`direction=None`) "transverse" means the largest eigenvalue of
    Lambda_split restricted to directions orthogonal to the local velocity --
    the general analogue of lambda_split(y) / lambda_x(z), correct whenever
    there is a single physically meaningful symmetry-breaking mode (as on the
    two-target axis, where the orthogonal complement of the direction of
    travel is one-dimensional).

    In ambient dimension D >= 3 that orthogonal complement can itself be
    multi-dimensional (e.g. the perpendicular three-target geometry off an
    equilateral configuration, where the population's own drift is not
    exactly along a coordinate axis) and may contain more than one candidate
    mode -- e.g. "split toward one of the base-pair archetypes" (e_x) versus
    "drift toward/away from the apex" (a different in-plane direction), which
    are physically distinct bifurcations. Pass a fixed `direction` (e.g.
    (1,0,0)) to pin the analysis to the specific mode of interest instead of
    automatically taking whichever is locally largest; this is required to
    reproduce lambda_x(z) exactly (perpendicular_common.py) for the general
    (non-equilateral) geometry, where the true velocity is not purely axial.

    Returns lambda_max (K+1,), speed (K+1,).
    """
    K1 = positions.shape[0]
    lambda_max = np.empty(K1)
    speed = np.empty(K1)
    for k in range(K1):
        v = coherent_velocity(positions[k], archetypes, gamma, beta, N)
        speed[k] = np.linalg.norm(v)
        if direction is not None:
            lambda_max[k] = directional_eigenvalue(positions[k], archetypes, gamma, direction, beta, N)
            continue
        s = speed[k]
        L = transverse_operator(positions[k], archetypes, gamma, beta, N)
        if s < EPS:
            vals = np.linalg.eigvalsh(L)
            lambda_max[k] = float(vals[-1])
        else:
            vals, _ = restricted_eigen(L, v / s)
            lambda_max[k] = float(vals[0])
    return lambda_max, speed


def accumulated_growth(times, lambda_max):
    """Gamma(t) = integral_0^t lambda_max(s) ds, via cumulative trapezoid.
    Gamma(t) is the log of the noise-amplification factor by time t (see
    module docstring) -- this project's analogue of Bass-Burdzy's local time.
    """
    dt = np.diff(times)
    avg = 0.5 * (lambda_max[:-1] + lambda_max[1:])
    Gamma = np.concatenate([[0.0], np.cumsum(avg * dt)])
    return Gamma


def exact_variance(times, lambda_max, sigma):
    """V(t) = Var[x(t)] for dx = lambda(t) x dt + sigma dW, V(0)=0, integrated
    exactly rather than approximated by the crude sigma^2*exp(2*Gamma(t))
    shortcut used elsewhere (accumulated_growth/theoretical_bifurcation_point's
    'visible' estimator).

    By Ito's lemma, V obeys the linear ODE dV/dt = 2*lambda(t)*V + sigma^2,
    V(0)=0 -- noise injected continuously from true co-location, not a lump
    of variance sigma^2 deposited at t=0. Integrated with an exponential
    step (piecewise-constant lambda per interval, exact within each one),
    which is far more accurate than Euler wherever lambda is large:
        V_{k+1} = V_k*exp(2*lam*dt) + sigma^2*(exp(2*lam*dt)-1)/(2*lam)   (lam != 0)
        V_{k+1} = V_k + sigma^2*dt                                        (lam == 0)
    This is the model's analogue of the exact Bass-Burdzy-style local-time
    accounting, as opposed to accumulated_growth's approximation of it.
    """
    V = np.empty_like(lambda_max)
    V[0] = 0.0
    for k in range(len(times) - 1):
        dt = times[k + 1] - times[k]
        lam = 0.5 * (lambda_max[k] + lambda_max[k + 1])
        exponent = 2.0 * lam * dt
        if abs(exponent) < 1e-12:
            V[k + 1] = V[k] + sigma ** 2 * dt
        else:
            # Clip before exp: once the exponent is this large, V is already
            # many orders of magnitude past any realistic x_partial threshold
            # (already-crossed, for every practical purpose), so clipping only
            # avoids an overflow warning -- it doesn't change whether/when a
            # threshold crossing is detected downstream.
            e = np.exp(np.clip(exponent, -700.0, 700.0))
            V[k + 1] = V[k] * e + sigma ** 2 * np.expm1(np.clip(exponent, -700.0, 700.0)) / (2.0 * lam)
    return V


# ── Theoretical bifurcation point ─────────────────────────────────────────────

def theoretical_bifurcation_point(source, archetypes, gamma, beta=1.0, N=1.0,
                                   T=800.0, K=40000, sigma=0.01, x_partial=0.20,
                                   height_dim=-1, direction=None):
    """Deterministic-theory bifurcation diagnostics along the coherent path.

    Returns a dict with:
      times, positions, lambda_max, speed, Gamma        -- full profiles
      t_sign, height_sign                                -- first (and lasting)
          lambda_max > 0 crossing: generalizes y_c / z_c
      t_visible, height_visible                          -- first Gamma(t) >=
          ln(x_partial/sigma) crossing: generalizes Gamma_total's threshold
    `height_dim` selects which coordinate of `positions` is reported as the
    "height" (source's descent coordinate); -1 matches this project's
    convention (y for the two-target axial case, z for the perpendicular
    three-target case). `direction`: see eigen_along_path.
    """
    times, positions = integrate_coherent_path(source, archetypes, gamma, beta, N, T, K)
    lambda_max, speed = eigen_along_path(times, positions, archetypes, gamma, beta, N, direction)
    Gamma = accumulated_growth(times, lambda_max)
    V = exact_variance(times, lambda_max, sigma)

    t_sign = _first_lasting_positive(times, lambda_max)
    t_visible = _first_threshold_crossing(times, Gamma, np.log(x_partial / sigma))
    t_exact = _first_threshold_crossing(times, np.sqrt(np.maximum(V, 0.0)), x_partial)

    def _height_at(t):
        if t is None:
            return None
        idx = int(np.searchsorted(times, t))
        idx = min(idx, positions.shape[0] - 1)
        return float(positions[idx, height_dim])

    return {
        "times": times, "positions": positions,
        "lambda_max": lambda_max, "speed": speed, "Gamma": Gamma, "V": V,
        "t_sign": t_sign, "height_sign": _height_at(t_sign),
        "t_visible": t_visible, "height_visible": _height_at(t_visible),
        "t_exact": t_exact, "height_exact": _height_at(t_exact),
    }


def _first_lasting_positive(times, values):
    """First index i such that values[i:] are all > 0 (a stable sign change,
    not a transient crossing) -- None if it never happens."""
    positive = values > 0.0
    if not positive.any():
        return None
    # last index where it's still (or again) <= 0; the crossing is right after
    nonpositive = np.where(~positive)[0]
    start = 0 if len(nonpositive) == 0 else nonpositive[-1] + 1
    if start >= len(values):
        return None
    return float(times[start])


def _first_threshold_crossing(times, values, threshold):
    idx = np.where(values >= threshold)[0]
    if len(idx) == 0:
        return None
    return float(times[idx[0]])


# ── Empirical bifurcation point, from actual noisy simulation ────────────────

def trace_lateral_std(trajectory, T, lateral_dim=0, height_dim=-1):
    """From a single (K+1, N, D) trajectory, the full time series of the
    population's lateral spread (std across cells of the lateral coordinate)
    and its mean height -- the raw signal empirical_bifurcation_time detects
    a crossing in, exposed on its own so it can be plotted for inspection.

    Returns times (K+1,), lateral_std (K+1,), height (K+1,).
    """
    K1 = trajectory.shape[0]
    times = np.arange(K1) * (T / (K1 - 1))
    lateral_std = np.std(trajectory[:, :, lateral_dim], axis=1)
    height = np.mean(trajectory[:, :, height_dim], axis=1)
    return times, lateral_std, height


def empirical_bifurcation_time(trajectory, T, x_partial=0.20, lateral_dim=0,
                                 height_dim=-1):
    """From a single (K+1, N, D) trajectory (as returned inside
    greedy_trajectory's 'trajectory' key), the first step at which the
    population's lateral spread (std across cells of the lateral coordinate)
    reaches x_partial -- the empirical analogue of theoretical_bifurcation_point's
    "visible" crossing, using the same x_partial convention as
    perpendicular_common.X_PARTIAL.

    Returns dict(t, height) with t=None if the population never visibly splits
    within the simulated horizon.
    """
    times, lateral_std, height = trace_lateral_std(trajectory, T, lateral_dim, height_dim)
    idx = np.where(lateral_std >= x_partial)[0]
    if len(idx) == 0:
        return {"t": None, "height": None}
    step = int(idx[0])
    return {"t": float(times[step]), "height": float(height[step])}


def run_traced_seeds(source, archetypes, gamma, beta=1.0, pop_N=40, T=800.0, K=40000,
                      sigma=0.01, x_partial=0.20, n_seeds=6, seed0=0, r_stop=None,
                      lateral_dim=0, height_dim=-1):
    """Like run_empirical_seeds, but returns the full per-seed lateral_std(t)
    and height(t) traces plus the detected crossing (t, height) for each --
    the raw material for a diagnostic plot of "what did detection actually
    see." Runs are NOT cached (meant for a handful of seeds/gamma for
    inspection, not a full sweep).

    Returns a list of dicts, one per seed:
      {seed, times, lateral_std, height, t_detect, height_detect}
    (t_detect/height_detect are None if that seed never crossed x_partial).
    """
    runs = []
    for s in range(n_seeds):
        with contextlib.redirect_stdout(io.StringIO()):
            result = greedy_trajectory(source, archetypes, gamma, pop_N, K, T,
                                        mode="exp", log=True, sigma=sigma,
                                        seed=seed0 + s, beta=beta, r_stop=r_stop)
        times, lateral_std, height = trace_lateral_std(result["trajectory"], T,
                                                         lateral_dim, height_dim)
        det = empirical_bifurcation_time(result["trajectory"], T, x_partial=x_partial,
                                          lateral_dim=lateral_dim, height_dim=height_dim)
        runs.append({
            "seed": seed0 + s, "times": times, "lateral_std": lateral_std, "height": height,
            "t_detect": det["t"], "height_detect": det["height"],
        })
    return runs


def run_single_trajectory(source, archetypes, gamma, beta=1.0, pop_N=40, T=800.0, K=40000,
                           sigma=0.01, x_partial=0.20, seed=0, r_stop=None,
                           lateral_dim=0, height_dim=-1):
    """Like run_traced_seeds, but for exactly one seed and keeping the full
    (K+1, N, D) trajectory -- for plotting the actual spatial paths (not just
    the abstract lateral_std(t) signal) with the detected point marked on them.

    Returns dict with trajectory, times, lateral_std, height, t_detect,
    step_detect, height_detect (t_detect/step_detect/height_detect are None
    if this seed never crossed x_partial within T).
    """
    with contextlib.redirect_stdout(io.StringIO()):
        result = greedy_trajectory(source, archetypes, gamma, pop_N, K, T,
                                    mode="exp", log=True, sigma=sigma,
                                    seed=seed, beta=beta, r_stop=r_stop)
    trajectory = result["trajectory"]
    times, lateral_std, height = trace_lateral_std(trajectory, T, lateral_dim, height_dim)
    idx = np.where(lateral_std >= x_partial)[0]
    step_detect = int(idx[0]) if len(idx) else None
    return {
        "trajectory": trajectory, "times": times, "lateral_std": lateral_std, "height": height,
        "step_detect": step_detect,
        "t_detect": float(times[step_detect]) if step_detect is not None else None,
        "height_detect": float(height[step_detect]) if step_detect is not None else None,
    }


def run_empirical_seeds(source, archetypes, gamma, beta=1.0, pop_N=40, T=800.0, K=40000,
                         sigma=0.01, x_partial=0.20, n_seeds=10, seed0=0, r_stop=None,
                         lateral_dim=0, height_dim=-1):
    """Run n_seeds independent stochastic greedy_trajectory simulations and
    return the empirical bifurcation heights (those that split within T).
    """
    heights = []
    times = []
    for s in range(n_seeds):
        with contextlib.redirect_stdout(io.StringIO()):
            result = greedy_trajectory(source, archetypes, gamma, pop_N, K, T,
                                        mode="exp", log=True, sigma=sigma,
                                        seed=seed0 + s, beta=beta, r_stop=r_stop)
        out = empirical_bifurcation_time(result["trajectory"], T, x_partial=x_partial,
                                          lateral_dim=lateral_dim, height_dim=height_dim)
        if out["t"] is not None:
            heights.append(out["height"])
            times.append(out["t"])
    return np.array(times), np.array(heights)


# ── Gamma sweep: quantify "sharper landscape -> earlier bifurcation" ─────────

def gamma_sweep(source, archetypes, gammas, beta=1.0, N=1.0, sigma=0.01,
                 x_partial=0.20, pop_N=40, T=800.0, K=40000, n_seeds=10, seed0=0,
                 height_dim=-1, r_stop=None, direction=None, lateral_dim=0):
    """For each gamma: the theoretical sign-crossing and Gamma-threshold
    bifurcation heights, plus the empirical mean/std bifurcation height from
    n_seeds noisy simulations. This is the quantitative counterpart of the
    qualitative "trunk shortens as gamma grows" observation in
    sharper_landscape.py / gamma_effect_density.py.

    `direction` (see eigen_along_path) pins the theory to one physical mode
    (e.g. (1,0,0) for the perpendicular geometry's base-pair split) instead of
    automatically taking the largest eigenvalue transverse to travel;
    `lateral_dim` must then match it for the empirical side.

    Returns a dict of arrays, all indexed the same as `gammas`.
    """
    height_sign, height_visible, height_exact = [], [], []
    emp_mean, emp_std, emp_frac = [], [], []

    for gamma in gammas:
        theory = theoretical_bifurcation_point(
            source, archetypes, gamma, beta=beta, N=N, T=T, K=K,
            sigma=sigma, x_partial=x_partial, height_dim=height_dim, direction=direction,
        )
        height_sign.append(theory["height_sign"])
        height_visible.append(theory["height_visible"])
        height_exact.append(theory["height_exact"])

        rs = r_stop if r_stop is not None else (0.05 if gamma < 1.0 else None)
        _, heights = run_empirical_seeds(
            source, archetypes, gamma, beta=beta, pop_N=pop_N, T=T, K=K,
            sigma=sigma, x_partial=x_partial, n_seeds=n_seeds, seed0=seed0, r_stop=rs,
            lateral_dim=lateral_dim, height_dim=height_dim,
        )
        emp_frac.append(len(heights) / n_seeds)
        emp_mean.append(float(np.mean(heights)) if len(heights) else None)
        emp_std.append(float(np.std(heights)) if len(heights) else None)

    return {
        "gammas": np.asarray(gammas, dtype=float),
        "height_sign": np.array(height_sign, dtype=float),
        "height_visible": np.array(height_visible, dtype=float),
        "height_exact": np.array(height_exact, dtype=float),
        "empirical_mean": np.array(emp_mean, dtype=float),
        "empirical_std": np.array(emp_std, dtype=float),
        "empirical_split_fraction": np.array(emp_frac, dtype=float),
    }
