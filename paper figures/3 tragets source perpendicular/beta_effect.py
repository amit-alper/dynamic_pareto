"""
Effect of bandwidth beta on the perpendicular three-target split, at fixed shape
gamma = 1.3, archetypes placed symmetrically (e/b = 1).

Three panels: beta = 0.7 (beta < 1), 1.0 (beta = 1), 1.6 (beta > 1). Per the theory
in perpendicular_common.z_critical, beta plays the same monotone role as gamma:
z_c is not finite at beta=0.7 (trapped -- the population stays coherent along the
whole descent), then |z_c| grows with beta (0.59 at beta=1.0, 1.55 at beta=1.6),
splitting earlier and earlier along the approach to the plane.

Run from anywhere:
    python "paper figures/3 tragets source perpendicular/beta_effect.py"
"""

import numpy as np
import matplotlib.pyplot as plt

from perpendicular_common import (
    archetypes3, source_point, z_critical, run_perp, sample_lines,
    shared_lims_3d, plot_perp_panel, topology_caption, classify,
    HERE, N_DEFAULT,
)

E, B, H = 1.0, 1.0, 2.5
GAMMA = 1.3
BETAS = [0.7, 1.0, 1.6]
N_SEEDS = 8


def main():
    A = archetypes3(E, B)
    src = source_point(E, H)

    panels = []
    for beta in BETAS:
        zc = z_critical(GAMMA, beta, E, B)
        trajs = [run_perp(seed, GAMMA, beta, E, B, H) for seed in range(N_SEEDS)]
        finals_all = np.concatenate([t[-1] for t in trajs], axis=0)
        fates_all = classify(finals_all, A)
        lines, line_fates = sample_lines(trajs, A, max_lines=44, rng=np.random.default_rng(0))
        panels.append((beta, zc, lines, line_fates, fates_all))
        print(f"beta={beta}, gamma={GAMMA}: z_c(theory)={zc:.3g}  [{topology_caption(fates_all)}]")

    lim_points = [A, src[None, :]]
    for _, _, lines, _, _ in panels:
        for p in lines:
            lim_points.append(p)
    lims = shared_lims_3d(lim_points, pad=1.12)

    n = len(panels)
    fig = plt.figure(figsize=(4.8 * n, 4.2))
    regime = {0.7: r"$\beta<1$", 1.0: r"$\beta=1$", 1.6: r"$\beta>1$"}
    for i, (beta, zc, lines, line_fates, fates_all) in enumerate(panels):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        zc_str = f"{zc:.2f}" if np.isfinite(zc) else ("trapped" if zc == -np.inf else "0")
        title = (rf"$\beta$ = {beta:g}  ({regime[beta]}),  $z_c$ = {zc_str}" + "\n"
                 + topology_caption(fates_all))
        plot_perp_panel(ax, A, src, lines, line_fates, zc=zc, title=title, lims=lims,
                        show_legend=(i == 0))

    fig.suptitle(rf"Perpendicular source, symmetric geometry ($e/b=1$): bandwidth effect at "
                 rf"$\gamma$ = {GAMMA:g}", fontsize=15, y=1.02)
    fig.text(0.5, -0.02,
              f"N={N_DEFAULT} cells/seed, {N_SEEDS} seeds, e={E:g}, b={B:g}, h={H:g}. "
              "Increasing bandwidth beta narrows the kernel and destabilizes the coherent "
              "descent earlier (larger |z_c|), exactly as it does for two targets -- here it "
              "additionally decides how much of the population ever leaves the interior "
              r"(non-archetype) fixed point rather than converging on $a_1,a_2,a_3$. Colour = "
              r"final archetype ($a_1$=blue, $a_2$=red, $a_3$=green); grey = coherent "
              "(genuinely still at the interior fixed point); purple = partial (laterally "
              "split but short of any archetype) -- see gamma_effect.py's docstring for how "
              "these were checked against a much longer run to rule out slow transients.",
              ha="center", fontsize=9.5, color="0.3", wrap=True)

    out = f"{HERE}/beta_effect.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
