"""
Perpendicular three-target counterpart of quantify_bifurcation_vs_gamma.py.

Same idea, same core.bifurcation.gamma_sweep machinery, but on the equilateral
perpendicular three-target geometry of
paper figures/3 tragets source perpendicular/perpendicular_common.py and
perpendicular_instability_strengthening.tex, using that note's own
"recommended parameter sweeps" table (beta>1 and beta<1 rows) so the theory
curves here can be checked directly against its motivating figure
(gamma_effect_beta_gt1_equilateral.png) and its keybox table of
min_z lambda_x(z) values.

Because the population's own drift at (0, e/3, z) is not purely axial for a
general (non-equilateral) triangle (Appendix A of the companion note), but
*is* purely axial for the equilateral case used here, the theory is pinned to
direction=(1,0,0) (core.bifurcation's `direction` override) so it reproduces
lambda_x(z) exactly rather than "whichever transverse mode is locally
largest" -- see core/bifurcation.py:eigen_along_path's docstring. This is
validated against perpendicular_common.lambda_x and phase_portrait.z_dot /
Gamma_total in tests/test_stability_bifurcation.py.

Run from anywhere:
    python "paper figures/Bifurcation quantification/quantify_bifurcation_vs_gamma_perp3d.py"
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

from core.bifurcation import gamma_sweep
from core.cache import cached_run
import perpendicular_common as perp

CACHE_DIR = os.path.join(HERE, ".cache")

B = 1.0
E_EQ = np.sqrt(3.0) * B  # equilateral

# Recommended sweeps from perpendicular_instability_strengthening.tex, Sec.
# "Recommended parameter sweeps" -- brackets both sides of the critical
# bandwidth so one figure shows trapped -> finite z_c -> unstable-everywhere.
PANELS = {
    r"$\beta = 1.6$  ($\beta > 1$)": dict(
        beta=1.6, h=1.35, gammas=[0.8, 1.0, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0],
    ),
    r"$\beta = 0.85$  ($\beta < 1$)": dict(
        beta=0.85, h=2.10, gammas=[0.5, 0.8, 1.0, 1.2, 1.5, 1.9, 2.0, 2.5, 3.0],
    ),
}

POP_N, SIGMA, X_PARTIAL = perp.N_DEFAULT, perp.SIGMA_DEFAULT, perp.X_PARTIAL
# Half of perp.T_DEFAULT/K_DEFAULT, same dt=0.02 -- keeps the per-step
# stability behavior identical to the project's own convention while cutting
# wall-clock cost roughly in half for this demo sweep (not a publication run).
T, K = perp.T_DEFAULT / 2.0, perp.K_DEFAULT // 2
N_SEEDS = 5
DIRECTION = (1.0, 0.0, 0.0)  # base-pair split direction, matches lambda_x(z)


def sweep_for(beta, h, gammas):
    archetypes = perp.archetypes3(E_EQ, B)
    source = perp.source_point(E_EQ, h)
    params = dict(gammas=gammas, beta=beta, h=h, e=E_EQ, pop_N=POP_N, sigma=SIGMA,
                  x_partial=X_PARTIAL, T=T, K=K, n_seeds=N_SEEDS)

    def _compute():
        return gamma_sweep(source, archetypes, gammas, beta=beta, N=POP_N,
                            sigma=SIGMA, x_partial=X_PARTIAL, pop_N=POP_N,
                            T=T, K=K, n_seeds=N_SEEDS, seed0=0,
                            direction=DIRECTION, lateral_dim=0, height_dim=-1)

    return cached_run(CACHE_DIR, "gamma_sweep_perp3d", params, _compute)


def main():
    fig, axes = plt.subplots(1, len(PANELS), figsize=(7.4 * len(PANELS), 5.8))

    for ax, (title, cfg) in zip(axes, PANELS.items()):
        res = sweep_for(cfg["beta"], cfg["h"], cfg["gammas"])
        g = res["gammas"]

        ax.plot(g, res["height_sign"], "o-", color="#1f77b4", lw=1.8, ms=5,
                 label=r"theory: sign crossing $z_c$")
        ax.plot(g, res["height_visible"], "s--", color="#d62728", lw=1.8, ms=5,
                 label=r"theory: $\Gamma$-threshold (visible by arrival)")
        ok = ~np.isnan(res["empirical_mean"])
        ax.errorbar(g[ok], res["empirical_mean"][ok], yerr=res["empirical_std"][ok],
                     fmt="^", color="#2ca02c", ms=7, capsize=3, lw=1.5,
                     label="empirical (mean $\\pm$ std, noisy simulation)")

        ax.axhline(-cfg["h"], color="0.6", lw=0.8, ls=":")
        ax.text(min(cfg["gammas"]), -cfg["h"], "source height ", va="bottom", ha="left",
                 fontsize=8, color="0.5")
        ax.set_xlabel(r"$\gamma$ (landscape shape exponent)")
        ax.set_ylabel("bifurcation height $z^*$\n(more negative = earlier, closer to source)")
        ax.set_title(title, fontsize=12)
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04),
               ncol=3, fontsize=9.5, frameon=True)
    fig.suptitle("Perpendicular three-target geometry (equilateral): "
                  "bifurcation height vs. $\\gamma$", fontsize=13, y=1.16)
    caption = (
        f"Equilateral triangle ($e=\\sqrt{{3}}\\,b$, $b=1$), source on the centroid line, "
        f"N={POP_N} cells, $\\sigma$={SIGMA:g}, x_partial={X_PARTIAL:g}, {N_SEEDS} seeds/point, "
        f"T={T:g}, K={K}. Theory pinned to the base-pair split direction (1,0,0) "
        "(core/bifurcation.py's `direction` override), reproducing "
        "perpendicular_common.lambda_x(z) and phase_portrait.Gamma_total(gamma,beta) "
        "exactly (tests/test_stability_bifurcation.py). As in the two-target case, "
        "the sign-crossing $z_c$ (blue) saturates at the source height once the axis "
        "is unstable from the start, while the $\\Gamma$-threshold and empirical curves "
        "keep moving earlier past that point -- the same 'instability strengthens past "
        "the boundary' mechanism from perpendicular_instability_strengthening.tex, now "
        "measured against actual noisy 3D simulation rather than only the linear theory."
    )
    fig.text(0.5, -0.07, caption, ha="center", fontsize=9, color="0.3", wrap=True)

    out = os.path.join(HERE, "bifurcation_height_vs_gamma_perp3d.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
