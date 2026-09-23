"""
Shared geometry, theory, simulation and 3D plotting for the perpendicular-source
three-target experiments (companion to axial_fixed_point (1).pdf, Section 12).

Geometry (base half-spacing b fixed at 1, matching the paper):
    a1 = (-b, 0, 0),  a2 = (b, 0, 0),  a3 = (0, e, 0)      (triangle in the z=0 plane)
    source = (0, e/3, -h)                                  (on the centroid line, below the plane)

Theory (eq. 12-13 of the paper): the transverse eigenvalue lambda_x(z) along the
centroid line, and the critical height z_c := sup{z<0 : lambda_x(z)=0} at which the
population commits sideways to the base pair (a1, a2). Bandwidth beta and shape
gamma both push |z_c| up (earlier split); the height of the apex e barely moves it
(near-independence, Sec. 12.5) -- both are reproduced empirically in the sibling
scripts (gamma_effect.py, beta_effect.py, geometry_effect.py).

Simulation uses the project's actual bandwidth-Pareto flow (greedy.greedy_trajectory)
in 3 dimensions -- no change to core/greedy code needed, since every function there
is dimension-agnostic. Stochastic symmetry breaking (sigma) is what actually splits
the population; lambda_x/z_c is the linear prediction of *where* that split starts.

Run each sibling script from anywhere, e.g.:
    python "paper figures/3 tragets source perpendicular/gamma_effect.py"
"""

import os
import sys
import io
import contextlib

import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from greedy.greedy import greedy_trajectory
from core.cache import cached_run

CACHE_DIR = os.path.join(HERE, ".cache")

# ── Geometry ──────────────────────────────────────────────────────────────────

B = 1.0  # base half-spacing, fixed throughout (matches the paper's convention)


def archetypes3(e=1.0, b=B):
    """Base pair at (+-b, 0, 0), apex at (0, e, 0). Triangle lies in the z=0 plane."""
    return np.array([[-b, 0.0, 0.0], [b, 0.0, 0.0], [0.0, e, 0.0]])


def source_point(e=1.0, h=2.5):
    """Source on the centroid line, a distance h below the plane."""
    return np.array([0.0, e / 3.0, -h])


LABELS = (r"$a_1$", r"$a_2$", r"$a_3$")
COLORS = ("#1f77b4", "#d62728", "#2ca02c")   # a1=blue, a2=red, a3=green (paper Fig. 6)
COHERENT_COLOR = "#8c8c8c"   # genuinely still near x=0 / the interior fixed point
PARTIAL_COLOR = "#9b59b6"    # laterally split (|x| well past 0) but short of R_ARRIVE


# ── Theory: transverse eigenvalue and critical height z_c ────────────────────

def d_e(z, e=1.0, b=B):
    """Common distance from the centroid-line point (0, e/3, z) to a1 and a2."""
    return np.sqrt(b ** 2 + (e / 3.0) ** 2 + z ** 2)


def d_3(z, e=1.0):
    """Distance from the centroid-line point (0, e/3, z) to the apex a3."""
    return np.sqrt((2.0 * e / 3.0) ** 2 + z ** 2)


def lambda_x(z, gamma, beta, e=1.0, b=B, N=1.0):
    """Transverse (x) eigenvalue along the centroid line, eq. (13) of the paper."""
    de, d3 = d_e(z, e, b), d_3(z, e)
    return (beta * gamma / N) * (
        2.0 * de ** (gamma - 4.0) * (beta * gamma * de ** gamma - (gamma - 2.0) - de ** 2)
        - d3 ** (gamma - 2.0)
    )


def max_safe_h(gamma, beta, e=1.0, b=B, target_exp=15.0, cap=2.5):
    """Largest source depth h for which beta * d_e(z=-h)^gamma <= target_exp.

    The 'exp' kernel used by greedy_trajectory is P_ij = exp(-beta*d^gamma); the
    log-objective gradient divides P_ij by a per-target sum that core code clamps
    to a floor of EPS=1e-10 (core/gradients.py, core/performance.py) but does NOT
    clamp the numerator (the chain-rule P_ij factor) the same way. Once beta*d^gamma
    exceeds ~23 (exp(-23) ~ 1e-10), P_ij itself underflows below that floor and the
    ratio collapses to a spurious near-zero gradient instead of the correct
    polynomial-order value -- cells simply freeze in place, which looks exactly
    like a converged/trapped population but is a numerical artefact, not real
    dynamics. Capping h keeps beta*d^gamma comfortably below that cliff (using
    target_exp=15 for headroom) so every reported 'trapped' outcome in these
    scripts reflects genuine convergence, verified by re-running at longer T.
    """
    floor = b ** 2 + (e / 3.0) ** 2
    d_e_max = (target_exp / beta) ** (1.0 / gamma)
    if d_e_max ** 2 <= floor:
        return 0.0
    return min(cap, float(np.sqrt(d_e_max ** 2 - floor)))


def z_critical(gamma, beta, e=1.0, b=B, zmax=60.0, n=8000):
    """z_c := sup{z < 0 : lambda_x(z) = 0}.

    Conventions (Sec. 12.2): -inf if lambda_x < 0 throughout (population stays
    coherent all the way to the plane -- 'trapped' topology); 0.0 if lambda_x > 0
    throughout (unstable from arbitrarily far away -- splits immediately).
    """
    zs = -np.linspace(1e-4, zmax, n)[::-1]
    vals = lambda_x(zs, gamma, beta, e, b)
    roots = []
    for i in range(len(zs) - 1):
        if vals[i] == 0.0:
            roots.append(zs[i])
        elif vals[i] * vals[i + 1] < 0:
            roots.append(brentq(lambda z: lambda_x(z, gamma, beta, e, b), zs[i], zs[i + 1]))
    if not roots:
        return -np.inf if vals[-1] < 0 else 0.0
    return max(roots)


# ── Simulation ────────────────────────────────────────────────────────────────

N_DEFAULT, SIGMA_DEFAULT = 40, 0.01
T_DEFAULT, K_DEFAULT = 800.0, 40000
R_ARRIVE = 0.15    # distance below which a cell counts as "arrived" at an archetype
X_PARTIAL = 0.20   # |x| above this (and not arrived) counts as "partial" not "coherent"


def _hash_key(**kw):
    return kw


def run_perp(seed, gamma, beta, e=1.0, b=B, h=2.5, N=N_DEFAULT,
             T=T_DEFAULT, K=K_DEFAULT, sigma=SIGMA_DEFAULT, cache_dir=CACHE_DIR):
    """One cached greedy run in 3D. Returns (K+1, N, 3) float32 trajectory."""
    r_stop = 0.05 if gamma < 1.0 else None
    A = archetypes3(e, b)
    src = source_point(e, h)
    params = dict(kind="perp3d", seed=seed, gamma=gamma, beta=beta, e=e, b=b, h=h,
                  N=N, T=T, K=K, sigma=sigma, r_stop=r_stop)

    def _compute():
        with contextlib.redirect_stdout(io.StringIO()):
            res = greedy_trajectory(src, A, gamma, N, K, T, mode="exp", log=True,
                                     sigma=sigma, seed=seed, beta=beta, r_stop=r_stop)
        return res["trajectory"].astype(np.float32)

    return cached_run(cache_dir, "traj", params, _compute)


def classify(final_positions, archetypes, r_arrive=R_ARRIVE, x_partial=X_PARTIAL):
    """Per-cell fate: 0/1/2 for archetype index if arrived; else -1 (coherent,
    |x| < x_partial -- genuinely still near the interior/non-archetype fixed
    point) or -2 (partial -- laterally split, |x| >= x_partial, but short of
    R_ARRIVE of any archetype: a stable non-archetype two-cluster equilibrium,
    not a slow transient -- see the long-T verification in gamma_effect.py's
    docstring/notes).
    """
    d = np.linalg.norm(final_positions[:, None, :] - archetypes[None, :, :], axis=-1)
    amin = d.argmin(axis=1)
    arrived = d.min(axis=1) < r_arrive
    partial = (~arrived) & (np.abs(final_positions[:, 0]) >= x_partial)
    fate = np.where(arrived, amin, -1)
    return np.where(partial, -2, fate)


def fate_summary(fates, k=3):
    """(k,) arrival counts + coherent/partial counts, as fractions of len(fates)."""
    n = len(fates)
    counts = np.array([np.sum(fates == j) for j in range(k)])
    coherent = np.sum(fates == -1)
    partial = np.sum(fates == -2)
    return counts, coherent, partial, n


# ── 3D plotting (img.png style) ────────────────────────────────────────────────

def _trim_and_stride(traj, archetypes, r_arrive=R_ARRIVE, stride=25, dwell_frac=0.05):
    """(K+1, N, 3) -> per-cell decimated paths, each cut at first arrival + dwell."""
    K1, N, _ = traj.shape
    d = np.linalg.norm(traj[:, :, None, :] - archetypes[None, None, :, :], axis=-1)
    arrived = d.min(axis=2) < r_arrive                       # (K1, N)
    first = np.where(arrived.any(0), arrived.argmax(0), K1 - 1)
    dwell = max(1, int(dwell_frac * K1))
    cutoff = np.minimum(first + dwell, K1 - 1)
    paths = []
    for j in range(N):
        idx = np.arange(0, cutoff[j] + 1, stride)
        if idx[-1] != cutoff[j]:
            idx = np.append(idx, cutoff[j])
        paths.append(traj[idx, j, :])
    return paths


def sample_lines(traj_list, archetypes, max_lines=48, stride=25, rng=None, min_per_fate=6):
    """Pool trimmed per-cell paths across seeds -> (lines, fates) subsample for plotting.

    Stratified: up to `min_per_fate` lines are guaranteed for every fate present in
    the population (so a rare escapee topology is still visible), then the rest of
    the budget is filled by uniform random draw from what remains.
    """
    rng = rng or np.random.default_rng(0)
    all_paths, all_finals = [], []
    for traj in traj_list:
        paths = _trim_and_stride(traj, archetypes, stride=stride)
        all_paths.extend(paths)
        all_finals.append(np.array([p[-1] for p in paths]))
    finals = np.concatenate(all_finals, axis=0)
    fates = classify(finals, archetypes)
    n = len(all_paths)

    chosen = []
    for fate_val in (0, 1, 2, -1, -2):
        pool = np.where(fates == fate_val)[0]
        if len(pool):
            take = rng.choice(pool, size=min(min_per_fate, len(pool)), replace=False)
            chosen.extend(take.tolist())
    remaining_budget = max_lines - len(chosen)
    if remaining_budget > 0:
        rest = np.setdiff1d(np.arange(n), chosen)
        if len(rest):
            extra = rng.choice(rest, size=min(remaining_budget, len(rest)), replace=False)
            chosen.extend(extra.tolist())

    chosen = np.array(chosen)
    return [all_paths[i] for i in chosen], fates[chosen]


def shared_lims_3d(point_arrays, pad=1.15):
    P = np.vstack(point_arrays)
    lims = []
    for col in range(3):
        lo, hi = P[:, col].min(), P[:, col].max()
        span = max(hi - lo, 1e-6)
        margin = (pad - 1.0) * span
        lims.append((lo - margin, hi + margin))
    return lims


def plot_perp_panel(ax, archetypes, source, lines, line_fates, *,
                     zc=None, title=None, elev=16, azim=-55, lims=None,
                     show_legend=False, title_fontsize=11):
    """One 3D panel: archetype triangle, source, sampled trajectories, final dots."""
    tri = Poly3DCollection([archetypes], facecolor="#a9cce3", edgecolor="none", alpha=0.25)
    ax.add_collection3d(tri)
    for i in range(3):
        p, q = archetypes[i], archetypes[(i + 1) % 3]
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color="#1a3d63", lw=1.4, zorder=2)

    for pt, lab, col in zip(archetypes, LABELS, COLORS):
        ax.scatter(*pt, s=55, marker="o", color=col, edgecolor="black", linewidth=0.8, zorder=6)
        ax.text(pt[0], pt[1], pt[2], "  " + lab, fontsize=10, zorder=7)

    ax.scatter(*source, s=70, marker="^", color="#2ecc71", edgecolor="black",
               linewidth=0.8, zorder=6, label="source (start)" if show_legend else None)

    fate_color = {0: COLORS[0], 1: COLORS[1], 2: COLORS[2], -1: COHERENT_COLOR, -2: PARTIAL_COLOR}
    finals, final_colors = [], []
    for path, fate in zip(lines, line_fates):
        c = fate_color[fate]
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color=c, lw=0.9, alpha=0.55, zorder=3)
        finals.append(path[-1])
        final_colors.append(c)
    if finals:
        finals = np.array(finals)
        ax.scatter(finals[:, 0], finals[:, 1], finals[:, 2], s=28, c=final_colors,
                   edgecolor="black", linewidth=0.5, zorder=7,
                   label="final position" if show_legend else None)

    if zc is not None and np.isfinite(zc) and lims is not None:
        xs = np.linspace(*lims[0], 6)
        ys = np.linspace(*lims[1], 6)
        Xp, Yp = np.meshgrid(xs, ys)
        Zp = np.full_like(Xp, zc)
        ax.plot_surface(Xp, Yp, Zp, color="0.4", alpha=0.10, linewidth=0, zorder=1)

    if lims is not None:
        ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
    ax.set_xlabel("x", labelpad=-6, fontsize=9)
    ax.set_ylabel("y", labelpad=-6, fontsize=9)
    ax.set_zlabel("z", labelpad=-6, fontsize=9)
    ax.tick_params(labelsize=7, pad=0)
    ax.view_init(elev=elev, azim=azim)
    if title:
        ax.set_title(title, fontsize=title_fontsize, pad=2)
    if show_legend:
        ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.85)


# ── Many-seed density cloud + true-3D density panel ────────────────────────────

def trimmed_points(traj_list, archetypes, stride=25, dwell_frac=0.05):
    """Pool every cell's trimmed, decimated path (across seeds) into one (M, 3)
    point cloud -- the raw material for a density-coloured scatter, as opposed
    to sample_lines's small stratified subsample for individually drawn lines."""
    pts = []
    for traj in traj_list:
        pts.extend(_trim_and_stride(traj, archetypes, stride=stride, dwell_frac=dwell_frac))
    return np.concatenate(pts, axis=0)


def _point_density(points, bins=40):
    """Per-point local density via a 3D occupancy histogram (fast, no kde)."""
    H, edges = np.histogramdd(points, bins=bins)
    idx = np.empty((points.shape[0], 3), dtype=int)
    for d in range(3):
        idx[:, d] = np.clip(np.digitize(points[:, d], edges[d]) - 1, 0, bins - 1)
    return H[idx[:, 0], idx[:, 1], idx[:, 2]]


def plot_perp_density_panel_3d(ax, archetypes, source, points3d, *,
                                zc=None, title=None, elev=16, azim=-55, lims=None,
                                show_legend=False, title_fontsize=11,
                                cmap="viridis", gamma_pow=0.45, max_points=25000,
                                point_size=2.5, density_bins=40, rng=None):
    """Same 3D furniture as plot_perp_panel (triangle, archetypes, source,
    z_c plane, real Axes3D box/ticks/view) but the trajectory cloud -- pooled,
    arrival-trimmed points from many seeds -- is rendered as a density-
    coloured scatter instead of a handful of individually coloured lines.
    This keeps the genuine rotatable-3D geometry of the line-trajectory
    panels while still conveying path density from a much larger sample."""
    rng = rng or np.random.default_rng(0)
    tri = Poly3DCollection([archetypes], facecolor="#a9cce3", edgecolor="none", alpha=0.25)
    ax.add_collection3d(tri)
    for i in range(3):
        p, q = archetypes[i], archetypes[(i + 1) % 3]
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color="#1a3d63", lw=1.4, zorder=2)

    for pt, lab, col in zip(archetypes, LABELS, COLORS):
        ax.scatter(*pt, s=55, marker="o", color=col, edgecolor="black", linewidth=0.8, zorder=6)
        ax.text(pt[0], pt[1], pt[2], "  " + lab, fontsize=10, zorder=7)

    ax.scatter(*source, s=70, marker="^", color="#2ecc71", edgecolor="black",
               linewidth=0.8, zorder=6, label="source (start)" if show_legend else None)

    dens = _point_density(points3d, bins=density_bins)
    if points3d.shape[0] > max_points:
        keep = rng.choice(points3d.shape[0], size=max_points, replace=False)
    else:
        keep = np.arange(points3d.shape[0])
    order = np.argsort(dens[keep])  # draw denser points last, on top
    keep = keep[order]
    pts, d = points3d[keep], dens[keep]
    dn = d / (d.max() if d.max() > 0 else 1.0)
    sca = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=dn, cmap=cmap,
                      norm=PowerNorm(gamma_pow, vmin=0, vmax=1),
                      s=point_size, alpha=0.55, linewidths=0, zorder=3,
                      depthshade=False, label="path density" if show_legend else None)

    if zc is not None and np.isfinite(zc) and lims is not None:
        xs = np.linspace(*lims[0], 6)
        ys = np.linspace(*lims[1], 6)
        Xp, Yp = np.meshgrid(xs, ys)
        Zp = np.full_like(Xp, zc)
        ax.plot_surface(Xp, Yp, Zp, color="0.4", alpha=0.10, linewidth=0, zorder=1)

    if lims is not None:
        ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
    ax.set_xlabel("x", labelpad=-6, fontsize=9)
    ax.set_ylabel("y", labelpad=-6, fontsize=9)
    ax.set_zlabel("z", labelpad=-6, fontsize=9)
    ax.tick_params(labelsize=7, pad=0)
    ax.view_init(elev=elev, azim=azim)
    if title:
        ax.set_title(title, fontsize=title_fontsize, pad=2)
    if show_legend:
        ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.85)
    return sca


def topology_caption(fates, k=3):
    """Archetype fractions, then coherent/partial (if any) on their own line --
    kept short and line-broken deliberately so it doesn't overflow a narrow
    subplot column when several panels sit side by side."""
    counts, coherent, partial, n = fate_summary(fates, k)
    fracs = counts / n
    line1 = ", ".join(f"{LABELS[j]}={fracs[j]:.2f}" for j in range(k))
    extra = []
    if coherent:
        extra.append(f"coh={coherent / n:.2f}")
    if partial:
        extra.append(f"part={partial / n:.2f}")
    return line1 + ("\n" + ", ".join(extra) if extra else "")
