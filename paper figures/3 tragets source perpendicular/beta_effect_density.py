"""
Density-style beta sweep at fixed gamma, mirroring gamma_effect_density.py but
varying beta instead: same true-3D panel furniture (rotatable Axes3D box,
ticks, triangle, z_c plane), many seeds pooled into a density-coloured
scatter per panel (perpendicular_common.plot_perp_density_panel_3d).

    gamma_effect_beta2_equilateral_gamma2.png
        equilateral geometry (e=sqrt(3)*b), gamma=2.0 fixed,
        beta in {0.85, 1.60, 2.00}, h=1.35 (same source depth as
        gamma_effect_beta_gt1_equilateral.png for direct comparison).

Note: at gamma=2.0 with this geometry, z_c(beta) is already 0 (unstable at
every depth) for all three beta values shown -- eq. 13's linear stability
prediction saturates past its own beta_perp_c (~0.75-0.8 here, see the sharp
-inf -> 0 jump checked over a finer beta grid). Any further sharpening of the
fork as beta increases is therefore a real nonlinear/empirical effect (faster
divergence rate despite an unchanged z_c sign), not a shift in the linear
z_c prediction -- exactly the situation sharper_landscape.py documents for
gamma at fixed beta.

Run from anywhere:
    python "paper figures/3 tragets source perpendicular/beta_effect_density.py"
"""

import numpy as np
import matplotlib.pyplot as plt

from perpendicular_common import (
    archetypes3, source_point, z_critical, run_perp, classify,
    trimmed_points, plot_perp_density_panel_3d, shared_lims_3d,
    topology_caption, HERE, N_DEFAULT,
)
from gamma_effect_density import _robust_cloud_bounds

N_SEEDS = 30
SIGMA = 0.02
STRIDE = 15
CMAP, GAMMA_POW = "viridis", 0.45
MAX_POINTS = 25000

CONFIGS = {
    "equilateral_gamma2": dict(gamma=2.0, e=np.sqrt(3.0), b=1.0, h=1.35,
                                betas=[0.70, 0.85, 1.60, 2.00, 2.40, 2.80],
                                geom_label=r"equilateral, $e=\sqrt{3}\,b$"),
    # gamma=1.5 sits below this geometry's own gamma_perp_c, so unlike the
    # gamma=2 panel above (z_c(beta) saturates at 0 for essentially every
    # beta>=0.8), z_c(beta) here keeps growing in magnitude across the whole
    # beta range (-0.36 at beta=0.85 to -9.0 at beta=3.0) -- deeper h=2.10 (vs
    # 1.35) so that growth is visible as a genuinely shrinking trunk instead
    # of an already-saturated fork-at-source. Lets you tell beta's effect
    # apart from gamma's: at gamma=2 any beta-driven change is a nonlinear
    # sharpening with an unmoved z_c; here z_c itself is moving.
    "equilateral_gamma1p5": dict(gamma=1.5, e=np.sqrt(3.0), b=1.0, h=2.10,
                                  betas=[0.70, 0.85, 1.20, 1.60, 2.00, 2.40, 3.00],
                                  geom_label=r"equilateral, $e=\sqrt{3}\,b$"),
}


def make_figure(label, cfg):
    e, b, h, gamma = cfg["e"], cfg["b"], cfg["h"], cfg["gamma"]
    A = archetypes3(e, b)
    src = source_point(e, h)
    betas = cfg["betas"]

    panels = []
    for beta in betas:
        zc = z_critical(gamma, beta, e, b)
        trajs = [run_perp(seed, gamma, beta, e, b, h, sigma=SIGMA) for seed in range(N_SEEDS)]
        finals_all = np.concatenate([t[-1] for t in trajs], axis=0)
        fates_all = classify(finals_all, A)
        cloud = trimmed_points(trajs, A, stride=STRIDE)
        panels.append((beta, zc, cloud, fates_all))
        print(f"{label}: beta={beta}, gamma={gamma:g}, h={h:g}: z_c={zc:.3g}  {cloud.shape[0]} pts  "
              f"[{topology_caption(fates_all)}]")

    cloud_lo, cloud_hi = _robust_cloud_bounds([p[2] for p in panels])
    lim_points = [A, src[None, :], cloud_lo[None, :], cloud_hi[None, :]]
    lims = shared_lims_3d(lim_points, pad=1.12)

    n = len(panels)
    ncols = 3
    nrows = -(-n // ncols)
    fig = plt.figure(figsize=(5.4 * ncols, 5.6 * nrows))
    sca = None
    for i, (beta, zc, cloud, fates_all) in enumerate(panels):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        if zc == -np.inf:
            zc_str = "trapped"
        elif zc == 0.0:
            zc_str = "unstable"
        else:
            zc_str = f"{zc:.2f}"
        title = rf"$\beta$={beta:g}  ($z_c$={zc_str})" + "\n" + topology_caption(fates_all)
        sca = plot_perp_density_panel_3d(ax, A, src, cloud, zc=zc, title=title, lims=lims,
                                          show_legend=(i == 0), title_fontsize=9.5,
                                          cmap=CMAP, gamma_pow=GAMMA_POW, max_points=MAX_POINTS)

    fig.suptitle(rf"Perpendicular source, {cfg['geom_label']}: beta effect at $\gamma={gamma:g}$",
                 fontsize=15, y=1.02)
    fig.subplots_adjust(wspace=0.5, hspace=0.55, right=0.88)
    cax = fig.add_axes([0.91, 0.30, 0.014, 0.42])
    cbar = fig.colorbar(sca, cax=cax)
    cbar.set_label("Path density")
    caption = (
        f"N={N_DEFAULT} cells/seed, {N_SEEDS} seeds/panel, e={e:.3g}, b={b:g}, h={h:g} "
        f"(fixed for every panel), gamma={gamma:g}, sigma={SIGMA:g}. Each cell's path is "
        "trimmed at first arrival (+ short dwell) and pooled across seeds into one point "
        "cloud per panel, coloured by local point density (viridis, power-law normalised, "
        f"gamma_pow={GAMMA_POW:g}). At this gamma, z_c(beta) is already 0 (unstable at "
        "every depth) for every beta shown -- any further sharpening of the fork with "
        "increasing beta is a real empirical/nonlinear effect, not a shift in the linear "
        "z_c prediction (see script docstring)."
    )
    fig.text(0.5, -0.02, caption, ha="center", fontsize=9.5, color="0.3", wrap=True)

    out = f"{HERE}/gamma_effect_beta_effect_{label}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}\n")


def main():
    for label, cfg in CONFIGS.items():
        make_figure(label, cfg)


if __name__ == "__main__":
    main()
