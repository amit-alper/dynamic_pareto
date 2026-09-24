"""
Single-cell vs collective optimisation, two-target geometry, gamma = 2.0, beta = 1.

Two panels, same style as density_1_gamma_k2.png (multi-seed greedy path-density
heatmaps via plotting.heatmap_plot.plot_trajectory_density_grid):

    left  -- opt_mode="cell"    each cell follows its own gradient  (generalisation)
    right -- opt_mode="tissue"  collective gradient of log F_tissue  (specialisation)

The FULL trajectories -- every step, every seed -- are cached per opt_mode under
./.cache as compressed .npz. Decimation and arrival trimming happen at plot time
(from the cache), so the density reflects the journey, not the endpoint pile-up.

opt_mode="tissue" is the T=800 / K=40000 run (long transit, forks after ~4000
steps). opt_mode="cell" is a short, fine-timestep run (T=30 / K=15000, dt=0.002):
the per-cell gradient is a stiff linear pull to the centroid so the collapse is
over in ~0.3 time units -- a coarse dt would sample it with a dozen points, so
the path is stepped finely and kept raw (no interpolation) to show its real,
drift-dominated (nearly deterministic at this sigma) stochastic texture.

Each cell's path is cut at its first arrival within r_arrive of its attractor
(the two archetypes for "tissue", the centroid for "cell") plus `settle` steps.

Run from anywhere:
    python "paper figures/Single cell vs collective/single_cell_vs_collective.py"
"""

import os
import sys
import io
import json
import hashlib
import contextlib

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from greedy.greedy import greedy_trajectory
from plotting.heatmap_plot import plot_trajectory_density_grid, rotate90_cw

# ── Configuration ────────────────────────────────────────────────────────────
A   = np.array([[-1.0, 0.0], [1.0, 0.0]])   # archetypes at +-1.0
SRC = np.array([0.0, -1.3])                  # source below the archetype axis

GAMMA   = 2.0
BETA    = 1.0
N, SIGMA, N_SEEDS = 70, 0.005, 30

# Per-mode integration (goes into the cache key). "cell" collapses in ~0.3 time
# units, so it uses a short run with a fine timestep (dt = T/K = 0.002).
RUN = {
    "cell":   dict(T=30.0,  K=15000),
    "tissue": dict(T=800.0, K=40000),
}

# Plot-time only (not cached): per-mode decimation, arrival radius, settle buffer.
# r_arrive is small so "arrived" means genuinely on the attractor; settle then
# keeps a short dwell there (the cluster equilibrates ~0.02-0.05 off it).
TRIM = {
    "cell":   dict(stride=1,  r_arrive=0.30, settle=150),    # short dwell -> path stays visible
    "tissue": dict(stride=20, r_arrive=0.05, settle=900),    # triggers on the dot, modest dwell
}

OPT_MODES   = ["cell", "tissue"]
PANEL_TITLE = {"cell": "Single-cell optimisation", "tissue": "Collective optimisation"}

CMAP, GAMMA_POW, BINS = "viridis", 0.45, 220
CACHE_DIR = os.path.join(HERE, ".cache")
OUT_PATH  = os.path.join(HERE, "single_cell_vs_collective_gamma2.png")


def _hash(params):
    blob = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:10]


def full_run(opt_mode):
    """Cached full trajectory stack (N_SEEDS, K+1, N, 2) float32 for one opt_mode.

    Cached as compressed .npz keyed by the run parameters -- the entire run is
    kept; nothing is decimated or trimmed before it lands on disk.
    """
    t, k = RUN[opt_mode]["T"], RUN[opt_mode]["K"]
    params = dict(kind="single_cell_vs_collective_fullrun", opt_mode=opt_mode,
                  gamma=GAMMA, beta=BETA, N=N, K=k, T=t, SIGMA=SIGMA,
                  N_SEEDS=N_SEEDS, A=A.tolist(), SRC=SRC.tolist())
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"fullrun_{opt_mode}_{_hash(params)}.npz")

    if os.path.exists(path):
        print(f"[cache] loaded {opt_mode} from {path}")
        with np.load(path) as z:
            return z["traj"]

    seeds = []
    for seed in range(N_SEEDS):
        with contextlib.redirect_stdout(io.StringIO()):
            res = greedy_trajectory(SRC, A, GAMMA, N, k, t, mode="exp", log=True,
                                    sigma=SIGMA, seed=seed, opt_mode=opt_mode, beta=BETA)
        seeds.append(res["trajectory"].astype(np.float32))
    traj = np.stack(seeds, axis=0)                                   # (N_SEEDS, K+1, N, 2)
    np.savez_compressed(path, traj=traj, params=json.dumps(params))
    print(f"[cache] saved {opt_mode} ({traj.shape}, {os.path.getsize(path)/1e6:.0f} MB) to {path}")
    return traj


def density_points(opt_mode):
    """(M, 2) arrival-trimmed, decimated point cloud pooled over all seeds.

    Read from the cached full run; each cell's raw path is cut at its first step
    within r_arrive of its attractor plus `settle` steps, then strided. No
    interpolation -- every plotted point is a real Euler sample.
    """
    traj = full_run(opt_mode)                                        # (S, K1, N, 2)
    S, K1, Ncell, _ = traj.shape
    targets = A if opt_mode == "tissue" else A.mean(axis=0, keepdims=True)
    cfg = TRIM[opt_mode]
    stride, r_arrive, settle = cfg["stride"], cfg["r_arrive"], cfg["settle"]

    steps = np.arange(K1)
    out = []
    for s in range(S):
        d = np.linalg.norm(traj[s][:, :, None, :] - targets[None, None, :, :],
                           axis=-1).min(axis=2)                      # (K1, N)
        arrived = d < r_arrive
        first = np.where(arrived.any(0), arrived.argmax(0), K1 - 1)  # (N,)
        cutoff = np.minimum(first + settle, K1 - 1)
        keep = steps[:, None] <= cutoff[None, :]                     # (K1, N)
        out.append(traj[s][keep][::stride])
    return np.concatenate(out, axis=0)


def main():
    pts_by_key = {m: density_points(m) for m in OPT_MODES}

    lim_points = [A, SRC[None, :]]
    fig, axes = plot_trajectory_density_grid(
        pts_by_key, OPT_MODES, [PANEL_TITLE[m] for m in OPT_MODES],
        lim_points=lim_points, bins=BINS, cmap=CMAP, gamma_pow=GAMMA_POW, pad=1.08,
        figsize_per_panel=(7.5, 7.5), ncols=2, minimal=True,
        suptitle=(rf"Single-cell vs collective optimisation "
                  rf"($\gamma = {GAMMA:g}$, $\beta = {BETA:g}$, $k = 2$)"),
        caption=(f"N={N}, $\\sigma$={SIGMA}, {N_SEEDS} seeds per panel, paths "
                 f"arrival-trimmed. tissue: T={RUN['tissue']['T']:g}, K={RUN['tissue']['K']}; "
                 f"cell: T={RUN['cell']['T']:g}, K={RUN['cell']['K']} (fast collapse, fine dt). "
                 "Single-cell dynamics collapse the population onto the generalist "
                 "centroid; the collective gradient splits it to the two archetypes."),
        savefig=False,
    )

    # Overlay source + archetypes (small dots), in the same rotated frame the
    # density grid uses: (x, y) -> (y, -x).
    A_r   = rotate90_cw(A)
    src_r = rotate90_cw(SRC)
    for ax in axes[:len(OPT_MODES)]:
        ax.scatter(A_r[:, 0], A_r[:, 1], s=45, marker="o", facecolor="white",
                   edgecolor="black", linewidth=1.2, zorder=6, label="archetypes")
        ax.scatter([src_r[0]], [src_r[1]], s=55, marker="^", facecolor="#ff5d3b",
                   edgecolor="black", linewidth=1.0, zorder=6, label="source")
    axes[0].legend(loc="lower left", fontsize=9, frameon=True, framealpha=0.85)

    fig.savefig(OUT_PATH, dpi=190, bbox_inches="tight", facecolor="white")
    print(f"saved: {OUT_PATH}")
    print(f"cache: {CACHE_DIR}")


if __name__ == "__main__":
    main()
