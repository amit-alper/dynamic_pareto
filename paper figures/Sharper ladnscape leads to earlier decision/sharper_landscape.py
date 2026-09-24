"""
"Sharper landscape leads to earlier decision" -- gamma effect at k = 2.

Same experiment and style as density_1_gamma_k2.png (two_targets.ipynb), but a
single row showing only gamma = 0.5, 3, 4. Two output files:

    sharper_landscape_beta1p6.png   BETA = 1.6  (the reference bandwidth)
    sharper_landscape_beta2p0.png   BETA = 2.0

Per-(gamma, beta) decimated, arrival-trimmed point clouds are cached in the
shared  <repo>/.cache/two_targets  store with the exact key layout used by the
notebook, so the beta = 1.6 panels reuse what the notebook already computed and
only the beta = 2.0 panels are run here.

Run from anywhere:
    python "paper figures/Sharper ladnscape leads to earlier decision/sharper_landscape.py"
"""

import os
import sys
import io
import contextlib

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from greedy.greedy import greedy_trajectory
from plotting.heatmap_plot import plot_trajectory_density_grid
from core.cache import cached_run

# ── Configuration (matches two_targets.ipynb) ────────────────────────────────
N, K, T, SIGMA = 70, 40000, 800.0, 0.005
A   = np.array([[-1.0, 0.0], [1.0, 0.0]])   # archetypes at +-1.0, D = 1.5
SRC = np.array([0.0, -1.3])

GAMMAS  = [0.5, 3.0, 4.0]
BETAS   = [1.6, 2.0]
N_SEEDS = 30
STRIDE  = 20

# Per-(beta, gamma) (T, K) override. At beta=2, gamma=4 the collective splitting
# mode is slow (effective lambda_split ~ 6e-3, so t_split ~ ln(1/sigma)/lambda ~
# 900). At T=800 the fork is truncated: ~8% of cells never reach an archetype and
# Var[x] is still growing at t=T, so the un-forked cells pile up as a source blob.
# K only sets the Euler step (dt = T/K) and was already stable, so refining it
# did nothing. Extending T lets the fork finish (arrival rises 0.92 -> 0.98);
# K is scaled with T to hold dt = 0.02.
TK_OVERRIDE = {(2.0, 4.0): (3000.0, 150000)}

CMAP, GAMMA_POW, BINS = "viridis", 0.45, 220
CACHE_DIR = os.path.join(ROOT, ".cache", "two_targets")   # shared with the notebook


def run_trial(seed, gamma, A, src, N, K, T, sigma, stride, beta=1.0,
              r_stop=None, R_frac=0.20):
    """One greedy run for a single seed -> decimated, arrival-trimmed (x, y) points.

    Verbatim from two_targets.ipynb: each cell's path is trimmed to its own
    first-arrival step (+ a short dwell buffer) so post-arrival equilibrium
    jitter doesn't dominate the density.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        result = greedy_trajectory(src, A, gamma, N, K, T, mode='exp', log=True,
                                    sigma=sigma, seed=seed, r_stop=r_stop, beta=beta)
    traj = result['trajectory']                                     # (K+1, N, 2)

    arm_dist = np.linalg.norm(A[:, None] - A[None, :], axis=-1)
    R = R_frac * arm_dist[arm_dist > 0].mean()                       # "arrived" radius

    dist_to_A = np.linalg.norm(traj[:, :, None, :] - A[None, None, :, :], axis=-1)
    arrived = dist_to_A.min(axis=2) < R                              # (K+1, N)

    K1 = traj.shape[0]
    first_arrival = np.where(arrived.any(axis=0), arrived.argmax(axis=0), K1 - 1)
    dwell = max(1, int(0.03 * K1))
    cutoff = np.minimum(first_arrival + dwell, K1 - 1)
    keep = np.arange(K1)[:, None] <= cutoff[None, :]

    return traj[keep][::stride].astype(np.float32)


def points_for(gamma, beta):
    """Cached decimated point cloud for one (gamma, beta), notebook key layout."""
    t, k = TK_OVERRIDE.get((beta, gamma), (T, K))

    def _compute():
        arr = np.concatenate(
            [run_trial(seed, gamma, A, SRC, N, k, t, SIGMA, STRIDE, beta=beta)
             for seed in range(N_SEEDS)],
            axis=0,
        )
        print(f"gamma={gamma}, beta={beta}, T={t:g}, K={k}: {arr.shape[0]} points from {N_SEEDS} seeds")
        return arr

    params = dict(N=N, K=k, T=t, SIGMA=SIGMA, STRIDE=STRIDE, N_SEEDS=N_SEEDS,
                  gamma=gamma, beta=beta, A=A.tolist(), SRC=SRC.tolist())
    return cached_run(CACHE_DIR, 'pts_gamma', params, _compute)


def make_figure(beta):
    pts_by_gamma = {g: points_for(g, beta) for g in GAMMAS}

    out = os.path.join(HERE, "sharper_landscape_beta" + f"{beta:.1f}".replace(".", "p") + ".png")
    ov = {g: tk for (b, g), tk in TK_OVERRIDE.items() if b == beta}
    tk_note = (f"T={T:g}, K={K}"
               + (" (" + ", ".join(rf"$\gamma$={g:g}: T={t:g}, K={k}" for g, (t, k) in ov.items()) + ")"
                  if ov else ""))
    lim_points = [A, SRC[None, :]]
    plot_trajectory_density_grid(
        pts_by_gamma, GAMMAS, [rf'$\gamma$ = {g:g}' for g in GAMMAS],
        lim_points=lim_points, bins=BINS, cmap=CMAP, gamma_pow=GAMMA_POW, pad=1.08,
        figsize_per_panel=(7.5, 7.5), ncols=3, minimal=True,
        suptitle=(rf'Sharper landscape leads to earlier decision  '
                  rf'($k = 2$, archetypes $\pm$1.0, D = 1.5, bandwidth $\beta$ = {beta:g})'),
        caption=(f'N={N}, {tk_note}, $\\sigma$={SIGMA}, $\\beta$={beta:g}, '
                 f'{N_SEEDS} seeds per panel. Sharper landscapes (higher $\\gamma$) '
                 'resolve the fork earlier -- the trunk is shorter and the population '
                 'splits sooner, closer to the source.'),
        savefig=out,
    )
    plt.close('all')
    print(f"saved: {out}")


def main():
    for beta in BETAS:
        make_figure(beta)
    print(f"cache: {CACHE_DIR}")


if __name__ == "__main__":
    main()
