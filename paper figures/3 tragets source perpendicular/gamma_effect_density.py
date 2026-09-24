"""
Density-style recreation of gamma_effect_beta_gt1.png / gamma_effect_beta_lt1.png.

Same perpendicular three-target geometry, theory and true-3D panel furniture
(rotatable Axes3D box, ticks, triangle, z_c plane) as gamma_effect.py -- but
built from many more seeds, with each cell's path trimmed at first arrival
and pooled into a single point cloud per panel, rendered as a density-
coloured 3D scatter (perpendicular_common.plot_perp_density_panel_3d) instead
of a handful of individually coloured trajectory lines. The other two
gamma_effect panels (beta_eq1, beta_1p5) are untouched and still come from
gamma_effect.py.

    gamma_effect_beta_gt1.png   beta=1.60, gamma in {1.2, 1.4, 2.5}, h=1.35
                                 (same source depth as the original figure)
    gamma_effect_beta_lt1.png   beta=0.85, gamma in {0.5, 1.2, 1.5, 1.9, 2.0, 2.5},
                                 h=2.10 (source pushed further away than the
                                 original 1.7, per request)

Run from anywhere:
    python "paper figures/3 tragets source perpendicular/gamma_effect_density.py"
"""

import numpy as np
import matplotlib.pyplot as plt

from perpendicular_common import (
    archetypes3, source_point, z_critical, run_perp, classify,
    trimmed_points, plot_perp_density_panel_3d, shared_lims_3d,
    topology_caption, HERE, N_DEFAULT,
)

E, B = 1.0, 1.0
N_SEEDS = 30
SIGMA = 0.02
STRIDE = 15
CMAP, GAMMA_POW = "viridis", 0.45
MAX_POINTS = 25000

CONFIGS = {
    "beta_gt1": dict(beta=1.60, h=1.35, gammas=[1.2, 2.0, 2.5],
                      title=r"$\beta = 1.60$  ($\beta > 1$)",
                      geom_label=r"$e/b=1$"),
    "beta_gt1_h1p5": dict(beta=1.60, h=1.50, gammas=[1.2, 2.0, 2.5],
                           title=r"$\beta = 1.60$  ($\beta > 1$)",
                           geom_label=r"$e/b=1$"),
    "beta_lt1": dict(beta=0.85, h=2.10, gammas=[0.5, 1.2, 1.5, 1.9, 2.0, 2.5],
                      title=r"$\beta = 0.85$  ($\beta < 1$)",
                      geom_label=r"$e/b=1$"),
    # Equilateral triangle (side 2*b): e = sqrt(3)*b. Same beta/gammas/h as
    # beta_gt1 -- z_c is checked near-independent of e (Sec. 12.5), and
    # max_safe_h at gamma=2.5 here is 2.158, comfortably above h=1.35.
    "beta_gt1_equilateral": dict(beta=1.60, e=np.sqrt(3.0), b=1.0, h=1.35,
                                  gammas=[2.0, 2.5, 3.0],
                                  title=r"$\beta = 1.60$  ($\beta > 1$)",
                                  geom_label=r"equilateral, $e=\sqrt{3}\,b$"),
    # h=1.50 (not 2.10): at h=2.10, gamma=3.5's exponent (beta*d_e^gamma~18)
    # sat past max_safe_h's conservative buffer (target_exp=15) and visibly
    # diverged deeper than the source's own start depth (final z down to
    # -3.54 despite h=2.10) -- a numerical breakdown, not real dynamics.
    # At h=1.50 the same gamma=3.5 exponent drops to ~7.9, comfortably safe.
    "beta_lt1_equilateral": dict(beta=0.85, e=np.sqrt(3.0), b=1.0, h=1.50,
                                  gammas=[0.5, 1.2, 1.5, 1.9, 2.0, 2.5, 3.0, 3.5],
                                  title=r"$\beta = 0.85$  ($\beta < 1$)",
                                  geom_label=r"equilateral, $e=\sqrt{3}\,b$"),
    # Same beta/h as beta_gt1_h1p5, but swept over lt1's broader gamma set
    # instead of gt1's -- lets the h=1.5 depth be compared across the full
    # low-to-high gamma range in the equilateral geometry.
    "beta_gt1_h1p5_equilateral": dict(beta=1.60, e=np.sqrt(3.0), b=1.0, h=1.50,
                                       gammas=[0.5, 1.2, 1.5, 1.9, 2.0, 2.5],
                                       title=r"$\beta = 1.60$  ($\beta > 1$)",
                                       geom_label=r"equilateral, $e=\sqrt{3}\,b$"),
    # Requested variant of beta_gt1_equilateral: source pushed further out
    # (h=2.10 vs 1.35) and a low/mid/high gamma set instead of the high-gamma
    # cluster above. max_safe_h at gamma=1.8 here is 2.5, comfortably above
    # h=2.10.
    "beta_gt1_h2p1_equilateral": dict(beta=1.60, e=np.sqrt(3.0), b=1.0, h=2.10,
                                       gammas=[0.5, 1.2, 1.8],
                                       title=r"$\beta = 1.60$  ($\beta > 1$)",
                                       geom_label=r"equilateral, $e=\sqrt{3}\,b$"),
}


def _robust_cloud_bounds(clouds, lo_pct=0.5, hi_pct=99.5):
    """Per-axis (lo, hi) bounds from percentiles pooled across panel clouds --
    used instead of raw min/max so a handful of sparse, far-flung outlier
    points at extreme gamma/beta don't dictate the shared view scale (and
    thereby visually shift the fixed archetypes/source between figure
    revisions that add or drop a panel)."""
    los = np.vstack([np.percentile(c, lo_pct, axis=0) for c in clouds])
    his = np.vstack([np.percentile(c, hi_pct, axis=0) for c in clouds])
    return los.min(axis=0), his.max(axis=0)


def make_figure(label, cfg):
    e, b = cfg.get("e", E), cfg.get("b", B)
    A = archetypes3(e, b)
    beta, h, gammas = cfg["beta"], cfg["h"], cfg["gammas"]
    src = source_point(e, h)

    panels = []
    for gamma in gammas:
        zc = z_critical(gamma, beta, e, b)
        trajs = [run_perp(seed, gamma, beta, e, b, h, sigma=SIGMA) for seed in range(N_SEEDS)]
        finals_all = np.concatenate([t[-1] for t in trajs], axis=0)
        fates_all = classify(finals_all, A)
        cloud = trimmed_points(trajs, A, stride=STRIDE)
        panels.append((gamma, zc, cloud, fates_all))
        print(f"{label}: gamma={gamma}, h={h:g}: z_c={zc:.3g}  {cloud.shape[0]} pts  "
              f"[{topology_caption(fates_all)}]")

    cloud_lo, cloud_hi = _robust_cloud_bounds([p[2] for p in panels])
    lim_points = [A, src[None, :], cloud_lo[None, :], cloud_hi[None, :]]
    lims = shared_lims_3d(lim_points, pad=1.12)

    n = len(panels)
    ncols = 3
    nrows = -(-n // ncols)
    fig = plt.figure(figsize=(5.4 * ncols, 5.6 * nrows))
    sca = None
    for i, (gamma, zc, cloud, fates_all) in enumerate(panels):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        if zc == -np.inf:
            zc_str = "trapped"
        elif zc == 0.0:
            zc_str = "unstable"
        else:
            zc_str = f"{zc:.2f}"
        title = rf"$\gamma$={gamma:g}  ($z_c$={zc_str})" + "\n" + topology_caption(fates_all)
        sca = plot_perp_density_panel_3d(ax, A, src, cloud, zc=zc, title=title, lims=lims,
                                          show_legend=(i == 0), title_fontsize=9.5,
                                          cmap=CMAP, gamma_pow=GAMMA_POW, max_points=MAX_POINTS)

    fig.suptitle(rf"Perpendicular source, {cfg['geom_label']}: gamma effect at {cfg['title']}",
                 fontsize=15, y=1.02)
    fig.subplots_adjust(wspace=0.5, hspace=0.55, right=0.88)
    cax = fig.add_axes([0.91, 0.30, 0.014, 0.42])
    cbar = fig.colorbar(sca, cax=cax)
    cbar.set_label("Path density")
    caption = (
        f"N={N_DEFAULT} cells/seed, {N_SEEDS} seeds/panel, e={e:.3g}, b={b:g}, h={h:g} "
        f"(fixed for every panel), sigma={SIGMA:g}. Each cell's path is trimmed at first "
        "arrival (+ short dwell) and pooled across seeds into one point cloud per panel, "
        "coloured by local point density (viridis, power-law normalised, "
        f"gamma={GAMMA_POW:g}) instead of by final archetype -- a much larger sample "
        "than the individually coloured line panels, still rendered as a genuine 3D "
        "scene. Dashed grey plane marks the theoretical critical height z_c (eq. 13-14 "
        "of the companion note)."
    )
    fig.text(0.5, -0.02, caption, ha="center", fontsize=9.5, color="0.3", wrap=True)

    out = f"{HERE}/gamma_effect_{label}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}\n")


def main():
    for label, cfg in CONFIGS.items():
        make_figure(label, cfg)


if __name__ == "__main__":
    main()
