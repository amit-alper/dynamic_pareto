"""
Quantifies "sharper landscape leads to earlier decision" (companion figure:
paper figures/Sharper ladnscape leads to earlier decision/sharper_landscape.py)
numerically instead of only visually, using the general core/stability.py +
core/bifurcation.py machinery (validated against the project's closed forms
in tests/test_stability_bifurcation.py).

For a sweep of gamma at fixed beta, two-target axial geometry (archetypes at
+-1, source at (0,-1.3), matching sharper_landscape.py exactly), this plots
three different notions of "when does the population bifurcate":

  - height_sign     the classical, purely local sign-crossing critical height
                     y_c: first height (walking down from the source) at which
                     the transverse eigenvalue becomes and stays positive.
                     Saturates at the source height once the axis is unstable
                     "everywhere" -- it cannot discriminate further increases
                     in gamma past that point.
  - height_visible   the accumulated-growth threshold height: first height at
                     which Gamma(t) = integral of the transverse eigenvalue
                     over time reaches ln(x_partial/sigma). This is this
                     project's analogue of Bass-Burdzy's local-time-driven
                     bifurcation time (see core/bifurcation.py docstring) --
                     it keeps discriminating gamma even once height_sign has
                     saturated, because a larger gamma still makes the
                     instability grow *faster*, not just *sooner*.
  - empirical         mean +/- std bifurcation height measured directly from
                     noisy greedy_trajectory simulations (the actual
                     stochastic system, not the linearized theory).

Run from anywhere:
    python "paper figures/Bifurcation quantification/quantify_bifurcation_vs_gamma.py"
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.bifurcation import gamma_sweep
from core.cache import cached_run

CACHE_DIR = os.path.join(HERE, ".cache")

ARCHETYPES = np.array([[-1.0, 0.0], [1.0, 0.0]])
SOURCE = np.array([0.0, -1.3])
GAMMAS = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
BETAS = [1.0, 1.6]

POP_N, SIGMA, X_PARTIAL = 40, 0.01, 0.20
T, K = 100.0, 10000
N_SEEDS = 8


def sweep_for(beta):
    params = dict(gammas=GAMMAS, beta=beta, pop_N=POP_N, sigma=SIGMA,
                  x_partial=X_PARTIAL, T=T, K=K, n_seeds=N_SEEDS)

    def _compute():
        return gamma_sweep(SOURCE, ARCHETYPES, GAMMAS, beta=beta, N=POP_N,
                            sigma=SIGMA, x_partial=X_PARTIAL, pop_N=POP_N,
                            T=T, K=K, n_seeds=N_SEEDS, seed0=0)

    return cached_run(CACHE_DIR, "gamma_sweep", params, _compute)


def main():
    fig, axes = plt.subplots(1, len(BETAS), figsize=(7.2 * len(BETAS), 5.6), sharey=True)

    for ax, beta in zip(axes, BETAS):
        res = sweep_for(beta)
        g = res["gammas"]

        ax.plot(g, res["height_sign"], "o-", color="#1f77b4", lw=1.8, ms=5,
                 label=r"theory: sign crossing $y_c$")
        ax.plot(g, res["height_visible"], "s--", color="#d62728", lw=1.8, ms=5,
                 label=r"theory: $\Gamma$-threshold (visible by arrival)")
        ok = ~np.isnan(res["empirical_mean"])
        ax.errorbar(g[ok], res["empirical_mean"][ok], yerr=res["empirical_std"][ok],
                     fmt="^", color="#2ca02c", ms=7, capsize=3, lw=1.5,
                     label="empirical (mean $\\pm$ std, noisy simulation)")

        ax.axhline(-1.3, color="0.6", lw=0.8, ls=":")
        ax.text(GAMMAS[0], -1.3, "source height ", va="bottom", ha="left",
                 fontsize=8, color="0.5")
        ax.set_xlabel(r"$\gamma$ (landscape shape exponent)")
        ax.set_title(rf"$\beta$ = {beta:g}", fontsize=12)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("bifurcation height $y^*$\n(more negative = earlier, closer to source)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=3, fontsize=9.5, frameon=True)
    fig.suptitle("Quantifying \"sharper landscape leads to earlier decision\": "
                  "bifurcation height vs. $\\gamma$", fontsize=13, y=1.14)
    caption = (
        f"Two archetypes at $x=\\pm1$, source at $(0,-1.3)$, N={POP_N} cells, "
        f"$\\sigma$={SIGMA:g}, x_partial={X_PARTIAL:g}, {N_SEEDS} seeds/point, T={T:g}, K={K}. "
        "Theory curves from core/bifurcation.py (deterministic coherent path + "
        "the general transverse eigenvalue of core/stability.py, validated against "
        "bandwidth_pareto_flow.tex's closed forms in tests/test_stability_bifurcation.py). "
        "The sign-crossing $y_c$ saturates at the source height once the axis is "
        "unstable from the very start (the 'unstable everywhere' regime); the "
        "$\\Gamma$-threshold and empirical curves keep moving earlier past that point, "
        "because a larger $\\gamma$ still makes the instability grow faster, not just "
        "sooner -- generalizing the 'instability strengthens past the boundary' result "
        "(perpendicular_instability_strengthening.tex) from the perpendicular "
        "three-target geometry to this two-target axial case."
    )
    fig.text(0.5, -0.06, caption, ha="center", fontsize=9, color="0.3", wrap=True)

    out = os.path.join(HERE, "bifurcation_height_vs_gamma.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
