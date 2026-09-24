"""
QC figure for quantify_bifurcation_vs_gamma.py's "empirical" curve: shows the
actual raw signal -- lateral_std(t) (population std of the x-coordinate
across N cells) for a handful of individual seeds -- with the detected
crossing point marked, so the empirical detection in core/bifurcation.py can
be checked by eye rather than trusted blindly.

Same source/archetypes/beta as quantify_bifurcation_vs_gamma.py; a subset of
its gamma grid (dense sweeps make for a busy grid of subplots). Each panel
also shows core/bifurcation.py's theoretical predictions (t_sign, t_visible)
as vertical lines, so the theory-vs-empirical comparison in the main figure
can be traced back to this raw signal too.

Run from anywhere:
    python "paper figures/Bifurcation quantification/diagnostic_empirical_detection.py"
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.bifurcation import run_traced_seeds, theoretical_bifurcation_point

ARCHETYPES = np.array([[-1.0, 0.0], [1.0, 0.0]])
SOURCE = np.array([0.0, -1.3])
BETA = 1.6
GAMMAS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]   # subset of quantify_bifurcation_vs_gamma.GAMMAS

POP_N, SIGMA, X_PARTIAL = 40, 0.01, 0.20
T, K = 100.0, 10000
N_SEEDS = 5   # same seed0=0 as the main sweep -> literally the same simulations


def main():
    ncols = 3
    nrows = -(-len(GAMMAS) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.2 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, gamma in zip(axes, GAMMAS):
        r_stop = 0.05 if gamma < 1.0 else None
        theory = theoretical_bifurcation_point(SOURCE, ARCHETYPES, gamma, beta=BETA,
                                                N=POP_N, T=T, K=K, sigma=SIGMA,
                                                x_partial=X_PARTIAL)
        runs = run_traced_seeds(SOURCE, ARCHETYPES, gamma, beta=BETA, pop_N=POP_N,
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
        ax.set_title(rf"$\gamma$={gamma:g}  ({n_split}/{len(runs)} seeds split by $T$)",
                      fontsize=11)
        ax.set_xlabel("time $t$")
        ax.set_ylim(0, max(0.35, X_PARTIAL * 1.6))

    for ax in axes[: len(GAMMAS)]:
        pass
    for i in range(0, len(axes), ncols):
        axes[i].set_ylabel(r"lateral std $\mathrm{std}_j[x_j(t)]$")
    for ax in axes[len(GAMMAS):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04),
               ncol=3, fontsize=9.5, frameon=True)
    fig.suptitle(rf"Empirical detection QC: raw lateral-std signal, $\beta$={BETA:g} "
                  "(two-target axial)", fontsize=13, y=1.1)
    caption = (
        f"Two archetypes at $x=\\pm1$, source at $(0,-1.3)$, N={POP_N} cells, "
        f"$\\sigma$={SIGMA:g}, T={T:g}, K={K}. Thin green lines: "
        f"{N_SEEDS} individual noisy simulations' lateral spread over time "
        "(std across cells of the x-coordinate); dark green dots: the step each "
        "one is detected as 'split' (first crossing of the x_partial threshold, "
        "core/bifurcation.py:empirical_bifurcation_time). Blue/red vertical lines: "
        "the theoretical sign-crossing and Gamma-threshold predictions for the "
        "same (gamma,beta), for direct comparison against where detection actually fires."
    )
    fig.text(0.5, -0.03, caption, ha="center", fontsize=9, color="0.3", wrap=True)

    out = os.path.join(HERE, "diagnostic_empirical_detection.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
