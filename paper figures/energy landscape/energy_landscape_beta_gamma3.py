"""
Energy landscape (single-cell performance) for two archetypes, gamma = 3.

Renders the single-cell fitness field
    F(x) = prod_j p(||x - a_j||) ,   p(d) = exp(-beta * d^gamma)
over 2D gene-expression space, side by side for beta = 0.85 and beta = 1.6
(the two bandwidths compared in "Sharper landscape leads to earlier
decision"). Uses core.performance directly.

Only F(x) = prod_j P_cell,task(x) is plotted here (the single-cell
landscape); the collective performance P_tissue = prod_j sum_i P_i,j is not
a function of a single point x and has no energy-landscape representation.

Run from the repo root:
    python "paper figures/energy landscape/energy_landscape_beta_gamma3.py"
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.performance import local_performance

# ── Configuration ────────────────────────────────────────────────────────────
ARCHETYPES = np.array([[-1.0, 0.0], [1.0, 0.0]])   # k = 2, the a_{1,2} = (-+1, 0) geometry
GAMMA      = 3.0
BETAS      = [0.85, 1.6]
GRID_N     = 500
EXTENT     = (-2.2, 2.2, -1.8, 1.8)                # x_min, x_max, y_min, y_max
LEVELS     = 40

HERE     = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "energy_landscape_gamma3_beta_0p85_1p6.png")


def performance_field(beta):
    """(GRID_N, GRID_N) single-cell performance F(x) on the plotting grid."""
    xs = np.linspace(EXTENT[0], EXTENT[1], GRID_N)
    ys = np.linspace(EXTENT[2], EXTENT[3], GRID_N)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    F = local_performance(pts, ARCHETYPES, gamma=GAMMA, mode="exp", beta=beta)
    return gx, gy, F.reshape(GRID_N, GRID_N)


def main():
    fields = [performance_field(beta) for beta in BETAS]

    # Shared colour scale across both panels.
    vmin = min(F.min() for _, _, F in fields)
    vmax = max(F.max() for _, _, F in fields)
    levels = np.linspace(vmin, vmax, LEVELS)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)

    cf = None
    for ax, beta, (gx, gy, F) in zip(axes, BETAS, fields):
        cf = ax.contourf(gx, gy, F, levels=levels, cmap="viridis",
                         vmin=vmin, vmax=vmax, extend="both")
        ax.contour(gx, gy, F, levels=levels, colors="white",
                   linewidths=0.3, alpha=0.25)
        ax.scatter(ARCHETYPES[:, 0], ARCHETYPES[:, 1], s=90, marker="o",
                   facecolor="white", edgecolor="black", linewidth=1.4, zorder=5)
        ax.set_title(rf"$\beta = {beta:g}$", fontsize=15)
        ax.set_xlabel("gene expression 1")
        ax.set_aspect("equal")
        ax.set_xlim(EXTENT[0], EXTENT[1])
        ax.set_ylim(EXTENT[2], EXTENT[3])
    axes[0].set_ylabel("gene expression 2")

    cbar = fig.colorbar(cf, ax=axes, fraction=0.046, pad=0.02)
    cbar.set_label(r"single-cell performance  $F(x) = \exp(-\beta \sum_j d_j^{\gamma})$")

    fig.suptitle(rf"Energy landscape (single-cell performance), $k = 2$, $\gamma = {GAMMA:g}$",
                 fontsize=16)

    fig.savefig(OUT_PATH, dpi=200, facecolor="white")
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
