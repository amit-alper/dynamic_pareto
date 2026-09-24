"""
Spatial (real x-y geometry) counterpart of diagnostic_empirical_detection.py:
for each gamma, plots the actual N-cell trajectories in the plane (not the
abstract lateral_std(t) signal), with the detected bifurcation point marked
directly on them -- the population's actual (x_j, y_j) positions at the exact
timestep empirical_bifurcation_time flagged as "split".

beta=1.6 only for now (per request).

Run from anywhere:
    python "paper figures/Bifurcation quantification/diagnostic_spatial_trajectories.py"
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.bifurcation import run_single_trajectory, theoretical_bifurcation_point

ARCHETYPES = np.array([[-1.0, 0.0], [1.0, 0.0]])
SOURCE = np.array([0.0, -1.3])
BETA = 1.6
GAMMAS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

POP_N, SIGMA, X_PARTIAL = 40, 0.01, 0.20
T, K = 100.0, 10000
SEED = 0


def main():
    ncols = 3
    nrows = -(-len(GAMMAS) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 5.4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, gamma in zip(axes, GAMMAS):
        r_stop = 0.05 if gamma < 1.0 else None
        theory = theoretical_bifurcation_point(SOURCE, ARCHETYPES, gamma, beta=BETA,
                                                N=POP_N, T=T, K=K, sigma=SIGMA,
                                                x_partial=X_PARTIAL)
        run = run_single_trajectory(SOURCE, ARCHETYPES, gamma, beta=BETA, pop_N=POP_N,
                                     T=T, K=K, sigma=SIGMA, x_partial=X_PARTIAL,
                                     seed=SEED, r_stop=r_stop)
        traj = run["trajectory"]  # (K+1, N, 2)

        final_x = traj[-1, :, 0]
        colors = np.where(final_x < 0, "#1f77b4", "#d62728")
        for j in range(POP_N):
            ax.plot(traj[:, j, 0], traj[:, j, 1], color=colors[j], lw=0.6, alpha=0.35)

        if run["step_detect"] is not None:
            step = run["step_detect"]
            ax.scatter(traj[step, :, 0], traj[step, :, 1], s=28, color="black",
                        zorder=6, label="cells at detected step")
            ax.axhline(run["height_detect"], color="black", lw=0.9, ls=":", alpha=0.7)

        if theory["height_sign"] is not None:
            ax.axhline(theory["height_sign"], color="#1f77b4", lw=1.3, ls="-",
                        label="theory: sign crossing")
        if theory["height_visible"] is not None:
            ax.axhline(theory["height_visible"], color="#d62728", lw=1.3, ls="--",
                        label=r"theory: $\Gamma$-threshold")

        ax.scatter(*SOURCE, marker="^", s=90, color="#2ecc71", edgecolor="black",
                    zorder=7, label="source")
        ax.scatter(ARCHETYPES[:, 0], ARCHETYPES[:, 1], marker="*", s=170,
                    color="gold", edgecolor="black", zorder=7, label="archetypes")

        n_txt = f"split at t={run['t_detect']:.2f}" if run["t_detect"] is not None else "no split by T"
        ax.set_title(rf"$\gamma$={gamma:g}  ({n_txt})", fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.4, 0.3)
        ax.set_aspect("equal")

    for ax in axes[len(GAMMAS):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05),
               ncol=5, fontsize=9, frameon=True)
    fig.suptitle(rf"Actual spatial trajectories with detected split marked, "
                  rf"$\beta$={BETA:g} (two-target axial, seed={SEED})", fontsize=13, y=1.12)
    caption = (
        f"Two archetypes at $x=\\pm1$ (gold stars), source at $(0,-1.3)$ (green triangle), "
        f"N={POP_N} cells, $\\sigma$={SIGMA:g}, T={T:g}, K={K}, single representative seed. "
        "Thin lines: every cell's full path, colored by which archetype it ends up nearer "
        "to. Black dots: every cell's actual position at the exact timestep detected as "
        "'split' (first crossing of the x_partial=0.2 lateral-std threshold); dotted black "
        "line: the height at that step. Blue/red horizontal lines: the theoretical "
        "sign-crossing and Gamma-threshold height predictions, for comparison."
    )
    fig.text(0.5, -0.03, caption, ha="center", fontsize=9, color="0.3", wrap=True)

    out = os.path.join(HERE, "diagnostic_spatial_trajectories.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
