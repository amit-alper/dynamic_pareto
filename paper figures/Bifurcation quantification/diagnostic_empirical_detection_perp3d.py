"""
Perpendicular three-target counterpart of diagnostic_empirical_detection.py.
Focuses on the beta=0.85 panel of quantify_bifurcation_vs_gamma_perp3d.py,
including gamma=3.0 where the empirical point diverged unusually far from the
Gamma-threshold theory in that figure -- this shows the raw signal behind
that specific point so the discrepancy can be inspected directly rather than
guessed at.

Run from anywhere:
    python "paper figures/Bifurcation quantification/diagnostic_empirical_detection_perp3d.py"
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PERP_DIR = os.path.join(ROOT, "paper figures", "3 tragets source perpendicular")
if PERP_DIR not in sys.path:
    sys.path.insert(0, PERP_DIR)

from core.bifurcation import run_traced_seeds, theoretical_bifurcation_point
import perpendicular_common as perp

B = 1.0
E_EQ = np.sqrt(3.0) * B
BETA = 0.85
H = 2.10
GAMMAS = [1.2, 1.9, 2.5, 3.0]   # subset of the beta<1 panel, including the flagged outlier

POP_N, SIGMA, X_PARTIAL = perp.N_DEFAULT, perp.SIGMA_DEFAULT, perp.X_PARTIAL
T, K = perp.T_DEFAULT / 2.0, perp.K_DEFAULT // 2   # matches quantify_bifurcation_vs_gamma_perp3d.py
N_SEEDS = 4
DIRECTION = (1.0, 0.0, 0.0)


def main():
    archetypes = perp.archetypes3(E_EQ, B)
    source = perp.source_point(E_EQ, H)

    ncols = 2
    nrows = -(-len(GAMMAS) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 4.4 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, gamma in zip(axes, GAMMAS):
        r_stop = 0.05 if gamma < 1.0 else None
        theory = theoretical_bifurcation_point(source, archetypes, gamma, beta=BETA,
                                                N=POP_N, T=T, K=K, sigma=SIGMA,
                                                x_partial=X_PARTIAL, direction=DIRECTION)
        runs = run_traced_seeds(source, archetypes, gamma, beta=BETA, pop_N=POP_N,
                                 T=T, K=K, sigma=SIGMA, x_partial=X_PARTIAL,
                                 n_seeds=N_SEEDS, seed0=0, r_stop=r_stop)

        for run in runs:
            ax.plot(run["times"], run["lateral_std"], lw=1.0, alpha=0.7, color="#2ca02c")
            if run["t_detect"] is not None:
                ax.plot(run["t_detect"], X_PARTIAL, "o", color="#1a6b1a", ms=6, zorder=5)

        ax.axhline(X_PARTIAL, color="0.4", lw=1.0, ls="--", label=r"$x_{\rm partial}$ threshold")
        if theory["t_sign"] is not None:
            ax.axvline(theory["t_sign"], color="#1f77b4", lw=1.4, ls="-",
                        label="theory: sign crossing")
        if theory["t_visible"] is not None:
            ax.axvline(theory["t_visible"], color="#d62728", lw=1.4, ls="--",
                        label=r"theory: $\Gamma$-threshold")

        n_split = sum(1 for r in runs if r["t_detect"] is not None)
        flag = "  <-- outlier in main figure" if gamma == 3.0 else ""
        ax.set_title(rf"$\gamma$={gamma:g}  ({n_split}/{len(runs)} seeds split by $T${flag})",
                      fontsize=11)
        ax.set_xlabel("time $t$")
        ax.set_ylim(0, max(0.35, X_PARTIAL * 1.6))

    for i in range(0, len(axes), ncols):
        axes[i].set_ylabel(r"lateral std $\mathrm{std}_j[x_j(t)]$")
    for ax in axes[len(GAMMAS):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05),
               ncol=3, fontsize=9.5, frameon=True)
    fig.suptitle(rf"Empirical detection QC: perpendicular 3-target, $\beta$={BETA:g}, "
                  f"equilateral, $h$={H:g}", fontsize=13, y=1.14)
    caption = (
        f"Equilateral triangle ($e=\\sqrt{{3}}\\,b$, $b=1$), source on centroid line, "
        f"N={POP_N} cells, $\\sigma$={SIGMA:g}, T={T:g}, K={K}. Thin green lines: "
        f"{N_SEEDS} individual noisy simulations' lateral spread over time; dark green "
        "dots: detected split step. Blue/red vertical lines: theoretical sign-crossing "
        "and Gamma-threshold predictions, pinned to the base-pair split direction "
        "(1,0,0). gamma=3.0 is the point that diverged from theory more than any other "
        "in quantify_bifurcation_vs_gamma_perp3d.py's beta<1 panel."
    )
    fig.text(0.5, -0.04, caption, ha="center", fontsize=9, color="0.3", wrap=True)

    out = os.path.join(HERE, "diagnostic_empirical_detection_perp3d.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
