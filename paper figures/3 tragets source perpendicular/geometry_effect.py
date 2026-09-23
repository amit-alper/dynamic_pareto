"""
Effect of the third archetype's distance (apex height e, base half-spacing b=1
fixed) on the perpendicular three-target split, at fixed gamma = 1.3, beta = 1.6.

e/b = 1.0 is the "symmetric" configuration used throughout gamma_effect.py and
beta_effect.py; this script moves the apex from close (e/b=0.6) through the
equilateral point (e/b=1.73) out to elongated (e/b=3.0), holding gamma and beta
fixed, with e/b=2.0 added around the equilateral point so the gradual handoff
between the base pair and the apex is resolved rather than jumping straight from
1.73 to 2.5.

The companion note's Sec. 12.5 predicts (and derives exactly for the equilateral
case) that the critical height z_c is nearly *independent* of e when the source
sits on the centroid line -- confirmed numerically here (z_c stays in
[-1.55, -1.46] across the whole e/b range, 0.6 to 3.0). What *does* change with
e/b is the shape of the triangle itself and, downstream of the split, how the
population divides between the base pair and the apex once it reaches the plane
-- that handoff is the thing this sweep is resolving.

Run from anywhere:
    python "paper figures/3 tragets source perpendicular/geometry_effect.py"
"""

import numpy as np
import matplotlib.pyplot as plt

from perpendicular_common import (
    archetypes3, source_point, z_critical, run_perp, sample_lines,
    shared_lims_3d, plot_perp_panel, topology_caption, classify,
    HERE, N_DEFAULT, B,
)

GAMMA, BETA, H = 1.3, 1.6, 2.5
E_RATIOS = [0.6, 1.0, 1.73, 2.0, 2.5, 3.0]
E_LABELS = ["0.6 (flattened)", "1.0 (symmetric)", "1.73 (equilateral)",
            "2.0", "2.5 (elongated)", "3.0 (very elongated)"]
N_SEEDS = 8


def main():
    panels = []
    for e, label in zip(E_RATIOS, E_LABELS):
        A = archetypes3(e, B)
        src = source_point(e, H)
        zc = z_critical(GAMMA, BETA, e, B)
        trajs = [run_perp(seed, GAMMA, BETA, e, B, H) for seed in range(N_SEEDS)]
        finals_all = np.concatenate([t[-1] for t in trajs], axis=0)
        fates_all = classify(finals_all, A)
        lines, line_fates = sample_lines(trajs, A, max_lines=44, rng=np.random.default_rng(0))
        panels.append((e, label, A, src, zc, lines, line_fates, fates_all))
        print(f"e/b={e}: z_c(theory)={zc:.3g}  [{topology_caption(fates_all)}]")

    lim_points = []
    for _, _, A, src, _, lines, _, _ in panels:
        lim_points += [A, src[None, :]] + list(lines)
    lims = shared_lims_3d(lim_points, pad=1.12)

    n = len(panels)
    fig = plt.figure(figsize=(4.8 * n, 4.2))
    for i, (e, label, A, src, zc, lines, line_fates, fates_all) in enumerate(panels):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        title = rf"$e/b$ = {label},  $z_c$ = {zc:.2f}" + "\n" + topology_caption(fates_all)
        plot_perp_panel(ax, A, src, lines, line_fates, zc=zc, title=title, lims=lims,
                        show_legend=(i == 0))

    fig.suptitle(rf"Perpendicular source: geometry effect (apex distance $e/b$) at "
                 rf"$\gamma$={GAMMA:g}, $\beta$={BETA:g}", fontsize=15, y=1.02)
    fig.text(0.5, -0.02,
              f"N={N_DEFAULT} cells/seed, {N_SEEDS} seeds, b={B:g}, h={H:g}. "
              "z_c is nearly independent of e/b (Sec. 12.5 of the companion note) -- the "
              "split from the base pair starts at almost the same height regardless of "
              "triangle shape -- while the shape of the triangle still changes how the "
              "population divides between the base pair and the apex once it reaches the plane.",
              ha="center", fontsize=9.5, color="0.3", wrap=True)

    out = f"{HERE}/geometry_effect.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
