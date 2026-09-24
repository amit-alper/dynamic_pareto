"""
Collective (mean-field) energy landscape, gamma = 3, beta = 0.85 vs 1.6.

F_tissue = prod_j S_j,  S_j = sum_i P_ij  depends on the whole population, so
it has no landscape over a single point x the way single-cell performance
does. What *is* well defined is the mean-field potential felt by one test
cell given the rest of the population fixed:

    U(x) = log F_tissue(background U {x})
         = sum_j log( S_j^bg + P(x, a_j) ),   S_j^bg = sum_{i in background} P_ij

This is exactly the field that "tissue" gradient descent (core/gradients.py,
tissue_gradient) differentiates for each cell -- so it is the real potential
driving specialisation, not an approximation invented for the plot.

Two snapshots of the background population are used, taken from an actual
greedy "tissue" trajectory (opt_mode="tissue"): t = 0 (all cells still at the
source, before the collective decision) and t = T (final, after the
population has specialised). Comparing the two shows how the effective
per-cell landscape sharpens from a single trunk-aligned peak into a
winner-take-all double well as the collective locks in its split.

Run from the repo root:
    python "paper figures/energy landscape/energy_landscape_collective_meanfield.py"
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from greedy.greedy import greedy_trajectory
from core.performance import performance_matrix, EPS

# ── Configuration (matches sharper_landscape.py) ─────────────────────────────
ARCHETYPES = np.array([[-1.0, 0.0], [1.0, 0.0]])
SRC        = np.array([0.0, -1.3])
N, K, T, SIGMA = 70, 40000, 800.0, 0.005
GAMMA      = 3.0
BETAS      = [0.85, 1.6]
SEED       = 0

GRID_N = 300
EXTENT = (-2.2, 2.2, -2.0, 1.8)
LEVELS = 40

HERE     = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "energy_landscape_collective_meanfield_gamma3.png")


def run_snapshot_pair(beta):
    """Background population positions at t=0 and t=T for one beta."""
    result = greedy_trajectory(SRC, ARCHETYPES, GAMMA, N, K, T, mode="exp", log=True,
                               sigma=SIGMA, seed=SEED, beta=beta, opt_mode="tissue")
    traj = result["trajectory"]          # (K+1, N, 2)
    return {"before": traj[0], "after": traj[-1]}


def meanfield_field(background, beta):
    """(GRID_N, GRID_N) mean-field potential U(x) felt by a test cell at x."""
    xs = np.linspace(EXTENT[0], EXTENT[1], GRID_N)
    ys = np.linspace(EXTENT[2], EXTENT[3], GRID_N)
    gx, gy = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])

    bg_perf     = performance_matrix(background, ARCHETYPES, gamma=GAMMA, mode="exp", beta=beta)
    bg_task_sum = bg_perf.sum(axis=0)                                  # (M,)

    grid_perf = performance_matrix(grid_pts, ARCHETYPES, gamma=GAMMA, mode="exp", beta=beta)
    task_sums = bg_task_sum[None, :] + grid_perf                       # (G, M)
    U = np.sum(np.log(task_sums + EPS), axis=1)
    return gx, gy, U.reshape(GRID_N, GRID_N)


def main():
    snapshots = {beta: run_snapshot_pair(beta) for beta in BETAS}
    rows = ["before", "after"]
    row_titles = {"before": "before decision (t = 0)", "after": "after decision (t = T)"}

    fields = {(beta, row): meanfield_field(snapshots[beta][row], beta)
              for beta in BETAS for row in rows}

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.6), constrained_layout=True)

    for r, row in enumerate(rows):
        row_fields = [fields[(beta, row)] for beta in BETAS]
        vmin = min(F.min() for _, _, F in row_fields)
        vmax = max(F.max() for _, _, F in row_fields)
        levels = np.linspace(vmin, vmax, LEVELS)

        cf = None
        for c, beta in enumerate(BETAS):
            ax = axes[r, c]
            gx, gy, U = fields[(beta, row)]
            cf = ax.contourf(gx, gy, U, levels=levels, cmap="viridis",
                             vmin=vmin, vmax=vmax, extend="both")
            ax.contour(gx, gy, U, levels=levels, colors="white",
                       linewidths=0.3, alpha=0.25)

            bg = snapshots[beta][row]
            ax.scatter(bg[:, 0], bg[:, 1], s=6, c="white", edgecolor="black",
                      linewidth=0.3, alpha=0.7, zorder=4)
            ax.scatter(ARCHETYPES[:, 0], ARCHETYPES[:, 1], s=90, marker="o",
                      facecolor="none", edgecolor="red", linewidth=1.6, zorder=5)

            ax.set_title(rf"$\beta = {beta:g}$" if r == 0 else "", fontsize=14)
            ax.set_aspect("equal")
            ax.set_xlim(EXTENT[0], EXTENT[1])
            ax.set_ylim(EXTENT[2], EXTENT[3])
            if c == 0:
                ax.set_ylabel(row_titles[row] + "\ngene expression 2", fontsize=11)
            if r == 1:
                ax.set_xlabel("gene expression 1")

        cbar = fig.colorbar(cf, ax=axes[r, :], fraction=0.046, pad=0.02)
        cbar.set_label(r"$U(x) = \log F_{\rm tissue}(\{$background$\} \cup \{x\})$", fontsize=10)

    fig.suptitle(
        rf"Collective mean-field energy landscape, $k = 2$, $\gamma = {GAMMA:g}$"
        "\n(white dots: fixed background population; red circles: archetypes)",
        fontsize=15)

    fig.savefig(OUT_PATH, dpi=200, facecolor="white")
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
