"""
Density-style recreation of geometry_effect.png: same perpendicular
three-target geometry, theory and true-3D panel furniture (rotatable Axes3D
box, ticks, triangle, z_c plane) as geometry_effect.py -- but built from many
more seeds, with each cell's path trimmed at first arrival and pooled into a
single point cloud per panel, rendered as a density-coloured 3D scatter
(perpendicular_common.plot_perp_density_panel_3d) instead of a handful of
individually coloured trajectory lines.

    geometry_effect_density.png        gamma=1.3, beta=1.6, h=2.5 (same as
                                        geometry_effect.py), e/b swept over
                                        {0.6, 1.0, 1.73, 2.0, 2.5, 3.0}
    geometry_effect_density_gamma2.png same e/b sweep, beta and h unchanged,
                                        but gamma=2.0 -- at this gamma z_c(e)
                                        is already 0 (unstable at every depth)
                                        for every e/b shown, so any change in
                                        the panels is purely how the triangle
                                        shape redistributes the population,
                                        not a shift in the linear z_c onset
                                        (mirrors the gamma=2.0 saturation seen
                                        in gamma_effect_density.py).

Run from anywhere:
    python "paper figures/3 tragets source perpendicular/geometry_effect_density.py"
"""

import numpy as np
import matplotlib.pyplot as plt

from perpendicular_common import (
    archetypes3, source_point, z_critical, run_perp, classify,
    trimmed_points, plot_perp_density_panel_3d, shared_lims_3d,
    topology_caption, HERE, N_DEFAULT, B,
)
from gamma_effect_density import _robust_cloud_bounds

N_SEEDS = 30
SIGMA = 0.02
STRIDE = 15
CMAP, GAMMA_POW = "viridis", 0.45
MAX_POINTS = 25000

E_RATIOS = [0.6, 1.0, 1.73, 2.0, 2.5, 3.0]
E_LABELS = ["0.6 (flattened)", "1.0 (symmetric)", "1.73 (equilateral)",
            "2.0", "2.5 (elongated)", "3.0 (very elongated)"]

CONFIGS = {
    "density": dict(gamma=1.3, beta=1.6, h=2.5, e_ratios=E_RATIOS, e_labels=E_LABELS,
                     title=r"$\gamma$=1.3, $\beta$=1.6"),
    # Same sweep at gamma=2.0: z_c(e) saturates to 0 (unstable at every depth)
    # for every e/b here, just as gamma=2.0 does in gamma_effect_density.py --
    # isolates the pure geometry/redistribution effect from any z_c shift.
    # h=1.35 (not 2.5): at h=2.5, gamma=2's "unstable everywhere" landscape lets
    # a large share of cells never commit to any archetype (up to 90% at
    # e/b=0.6), and trimmed_points keeps each non-arriving cell's ENTIRE T=800
    # trajectory (no early cutoff) -- these slow, never-converging cells wander
    # deep past the source's own depth (down to z~-3.45 vs h=2.5) over the full
    # run, which is genuine (smooth, no Euler-instability-warning, no single-step
    # jump) but swamps every panel in a single diffuse ball with the geometry
    # signal unreadable. h=1.35 matches the gamma=2.0 depth already used
    # elsewhere (gamma_effect_density.py's beta_gt1/beta_gt1_equilateral) where
    # the same saturated z_c produces a much smaller non-arrived fraction and a
    # legible panel.
    "density_gamma2": dict(gamma=2.0, beta=1.6, h=1.35, e_ratios=E_RATIOS, e_labels=E_LABELS,
                            title=r"$\gamma$=2.0, $\beta$=1.6"),
}


def make_figure(label, cfg):
    gamma, beta, h = cfg["gamma"], cfg["beta"], cfg["h"]
    e_ratios, e_labels = cfg["e_ratios"], cfg["e_labels"]

    panels = []
    for e, elabel in zip(e_ratios, e_labels):
        A = archetypes3(e, B)
        src = source_point(e, h)
        zc = z_critical(gamma, beta, e, B)
        trajs = [run_perp(seed, gamma, beta, e, B, h, sigma=SIGMA) for seed in range(N_SEEDS)]
        finals_all = np.concatenate([t[-1] for t in trajs], axis=0)
        fates_all = classify(finals_all, A)
        cloud = trimmed_points(trajs, A, stride=STRIDE)
        panels.append((e, elabel, A, src, zc, cloud, fates_all))
        print(f"{label}: e/b={e}, h={h:g}: z_c={zc:.3g}  {cloud.shape[0]} pts  "
              f"[{topology_caption(fates_all)}]")

    cloud_lo, cloud_hi = _robust_cloud_bounds([p[5] for p in panels])
    lim_points = []
    for _, _, A, src, _, _, _ in panels:
        lim_points += [A, src[None, :]]
    lim_points += [cloud_lo[None, :], cloud_hi[None, :]]
    lims = shared_lims_3d(lim_points, pad=1.12)

    n = len(panels)
    ncols = 3
    nrows = -(-n // ncols)
    fig = plt.figure(figsize=(5.4 * ncols, 5.6 * nrows))
    sca = None
    for i, (e, elabel, A, src, zc, cloud, fates_all) in enumerate(panels):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        if zc == -np.inf:
            zc_str = "trapped"
        elif zc == 0.0:
            zc_str = "unstable"
        else:
            zc_str = f"{zc:.2f}"
        title = rf"$e/b$={elabel}  ($z_c$={zc_str})" + "\n" + topology_caption(fates_all)
        sca = plot_perp_density_panel_3d(ax, A, src, cloud, zc=zc, title=title, lims=lims,
                                          show_legend=(i == 0), title_fontsize=9.5,
                                          cmap=CMAP, gamma_pow=GAMMA_POW, max_points=MAX_POINTS)

    fig.suptitle(rf"Perpendicular source: geometry effect (apex distance $e/b$) at {cfg['title']}",
                 fontsize=15, y=1.02)
    fig.subplots_adjust(wspace=0.5, hspace=0.55, right=0.88)
    cax = fig.add_axes([0.91, 0.30, 0.014, 0.42])
    cbar = fig.colorbar(sca, cax=cax)
    cbar.set_label("Path density")
    caption = (
        f"N={N_DEFAULT} cells/seed, {N_SEEDS} seeds/panel, b={B:g}, h={h:g}, "
        f"beta={beta:g} (fixed for every panel), gamma={gamma:g}, sigma={SIGMA:g}. Each "
        "cell's path is trimmed at first arrival (+ short dwell) and pooled across seeds "
        "into one point cloud per panel, coloured by local point density (viridis, "
        f"power-law normalised, gamma_pow={GAMMA_POW:g}) instead of by final archetype. "
        "z_c is nearly independent of e/b (Sec. 12.5 of the companion note) -- what changes "
        "with the triangle shape is how the population divides between the base pair and "
        "the apex once it reaches the plane."
    )
    fig.text(0.5, -0.02, caption, ha="center", fontsize=9.5, color="0.3", wrap=True)

    out = f"{HERE}/geometry_effect_{label}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}\n")


def main():
    for label, cfg in CONFIGS.items():
        make_figure(label, cfg)


if __name__ == "__main__":
    main()
