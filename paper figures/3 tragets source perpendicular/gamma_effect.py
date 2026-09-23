"""
Effect of gamma on the perpendicular three-target split, at four fixed
bandwidths (beta = 0.85, 1.00, 1.50, 1.60), archetypes placed symmetrically
(e/b = 1).

ONE FIXED SOURCE DEPTH PER BETA -- no per-panel h (read this before touching H)
--------------------------------------------------------------------------------
This mirrors the two-target reference (sharper_landscape.py), which fixes
SRC=(0,-1.3) across its whole gamma sweep and lets the trunk-before-the-fork
shrink naturally as gamma sharpens. Earlier revisions of this script instead
shrank h per panel (to dodge the numerical issue described below) or split
gamma into a "gradual" tier at one h and a "beyond" tier at another -- both
defeat the point of a controlled sweep: everything should move except gamma.

H[beta] below is chosen once, as large as numerically safe at gamma=4 (see
max_safe_h), and then used for every single gamma value in GAMMAS[beta]. The
qualitative story that falls out is real, not tuned: at low gamma the
population is fully coherent (z_c=-inf, direct capture at the apex); as gamma
grows, z_c grows too and the visible coherent "trunk" below the dashed z_c
plane visibly shrinks; once |z_c| exceeds this fixed h the fork happens right
at the source for every larger gamma (whether because z_c is merely deeper
than h, or -- past each beta's own gamma_perp_c -- because z_c stops existing
at all: Sec. 12.3 of the companion note, lambda_x>0 at every depth). Both
"fork already happened at the source" cases look alike here on purpose --
that IS what a fixed, finite source depth shows once the theory's own
early-decision effect has run its course.

Source depth safety: the 'exp' kernel used by greedy_trajectory is
P_ij=exp(-beta*d^gamma); the log-objective gradient divides P_ij by a
per-target sum that core code clamps to a floor of EPS=1e-10 but does NOT
clamp the numerator (the chain-rule P_ij factor) the same way. Once
beta*d^gamma exceeds ~23, P_ij itself underflows below that floor and the
ratio collapses to a spurious near-zero gradient instead of the correct
polynomial-order value -- cells look frozen but it is purely numerical. H[beta]
is picked via max_safe_h so this never triggers, even at gamma=4.

Noise: sigma=0.02 (up from the 0.01 default) -- checked empirically that this
stays statistically stable (6-seed batches barely shift vs. sigma in
[0.01, 0.03]) while giving late-diverging escapee trajectories more room to
separate from the coherent bulk earlier and more visibly, rather than "riding
the bulk almost to the plane, then hooking off at the last moment" the way a
very low sigma does.

CLASSIFYING WHERE THE POPULATION ENDS UP
--------------------------------------------------------------------------------
Two genuinely different non-arrival outcomes are distinguished (perpendicular_
common.classify): "coherent" (grey, |x| stays near 0 -- the population truly
never leaves the interior, non-archetype fixed point) and "partial" (purple,
|x| grows well past 0 but never quite reaches R_ARRIVE of an archetype). Both
were checked against a 25x-longer run (T=20000 vs. the T=800 used here) before
trusting them: neither position moves at all over that much longer horizon, so
both are genuine stable equilibria of the full nonlinear system -- not slow
transients and not an r_stop freezing artefact (r_stop is None whenever
gamma>=1, i.e. for every "coherent"/"partial" case in these figures). The
"partial" outcome is exactly the companion note's Prop. 8.1: once the coherent
state is transversally unstable, the population does not fly on to the
archetypes -- it settles at a different, genuinely two-cluster nonlinear
equilibrium instead.

Four output figures:
    gamma_effect_beta_lt1.png   (beta = 0.85)
    gamma_effect_beta_eq1.png   (beta = 1.00)
    gamma_effect_beta_1p5.png   (beta = 1.50)
    gamma_effect_beta_gt1.png   (beta = 1.60)

Run from anywhere:
    python "paper figures/3 tragets source perpendicular/gamma_effect.py"
"""

import numpy as np
import matplotlib.pyplot as plt

from perpendicular_common import (
    archetypes3, source_point, z_critical, max_safe_h, run_perp, sample_lines,
    shared_lims_3d, plot_perp_panel, topology_caption, classify,
    HERE, N_DEFAULT,
)

E, B = 1.0, 1.0
N_SEEDS = 8
SIGMA = 0.02

# One fixed depth per beta, chosen via max_safe_h at the largest gamma we show
# (4.0) with a small safety margin -- see docstring. Never varies across a
# figure's gamma sweep.
H = {0.85: 1.7, 1.00: 1.6, 1.50: 1.4, 1.60: 1.35}

# A single flat gamma list per beta: one trapped bookend, then a run that
# stays inside z_c<H (visible trunk, shrinking), then several beyond the point
# where z_c exceeds H (fork at the source for the rest of the sweep).
GAMMAS = {
    0.85: [0.8, 1.3, 1.5, 1.7, 1.8, 1.9, 2.5, 4.0],
    1.00: [0.6, 1.0, 1.2, 1.4, 1.6, 1.8, 2.5, 4.0],
    1.50: [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 2.5, 4.0],
    1.60: [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 2.5, 4.0],
}

FIG_LABEL = {0.85: "beta_lt1", 1.00: "beta_eq1", 1.50: "beta_1p5", 1.60: "beta_gt1"}
FIG_TITLE = {0.85: r"$\beta = 0.85$  ($\beta < 1$)",
             1.00: r"$\beta = 1.00$  ($\beta = 1$)",
             1.50: r"$\beta = 1.50$  ($\beta > 1$)",
             1.60: r"$\beta = 1.60$  ($\beta > 1$)"}


def make_figure(beta):
    A = archetypes3(E, B)
    h = H[beta]
    src = source_point(E, h)
    gammas = GAMMAS[beta]

    panels = []
    for gamma in gammas:
        zc = z_critical(gamma, beta, E, B)
        trajs = [run_perp(seed, gamma, beta, E, B, h, sigma=SIGMA) for seed in range(N_SEEDS)]
        finals_all = np.concatenate([t[-1] for t in trajs], axis=0)
        fates_all = classify(finals_all, A)
        lines, line_fates = sample_lines(trajs, A, max_lines=44, rng=np.random.default_rng(0))
        trunk = h - abs(zc) if np.isfinite(zc) else None
        panels.append((gamma, zc, lines, line_fates, fates_all))
        trunk_str = f"trunk={trunk:.2f}" if trunk is not None else "trunk=n/a (trapped)"
        print(f"beta={beta}, gamma={gamma}, h={h:g}: z_c(theory)={zc:.3g}  {trunk_str}  "
              f"[{topology_caption(fates_all)}]")

    lim_points = [A, src[None, :]]
    for _, _, lines, _, _ in panels:
        for p in lines:
            lim_points.append(p)
    lims = shared_lims_3d(lim_points, pad=1.12)

    n = len(panels)
    ncols = 4
    nrows = -(-n // ncols)
    fig = plt.figure(figsize=(4.4 * ncols, 4.6 * nrows))
    for i, (gamma, zc, lines, line_fates, fates_all) in enumerate(panels):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        if zc == -np.inf:
            zc_str = "trapped"
        elif zc == 0.0:
            zc_str = "unstable"
        else:
            zc_str = f"{zc:.2f}"
        title = (rf"$\gamma$={gamma:g}  ($z_c$={zc_str})" + "\n" + topology_caption(fates_all))
        plot_perp_panel(ax, A, src, lines, line_fates, zc=zc, title=title, lims=lims,
                        show_legend=(i == 0), title_fontsize=9.5)

    fig.suptitle(rf"Perpendicular source, symmetric geometry ($e/b=1$): gamma effect at {FIG_TITLE[beta]}",
                 fontsize=15, y=1.02)
    caption = (
        f"N={N_DEFAULT} cells/seed, {N_SEEDS} seeds, e={E:g}, b={B:g}, h={h:g} (fixed for "
        f"every panel, like the two-target reference), sigma={SIGMA:g}. Dashed grey plane "
        "marks the theoretical critical height z_c (eq. 13-14 of the companion note); the "
        "coherent 'trunk' below it visibly shrinks as gamma grows. Once |z_c| exceeds this "
        "fixed h, the fork happens at the source for every larger gamma shown -- whether "
        r"because z_c is merely deeper than h, or (past this beta's own $\gamma_{perp,c}$, "
        "Sec. 12.3) because z_c stops existing at all and the descent is unstable at every "
        "depth. Colour = final archetype ($a_1$=blue, $a_2$=red, $a_3$=green); grey = "
        "coherent (genuinely still at the interior, non-archetype fixed point); purple = "
        "partial (laterally split but short of any archetype) -- both verified stable "
        "against a 25x-longer run, not transients or an r_stop artefact (see script "
        "docstring)."
    )
    fig.text(0.5, -0.02, caption, ha="center", fontsize=9.5, color="0.3", wrap=True)
    fig.subplots_adjust(wspace=0.45, hspace=0.55)

    out = f"{HERE}/gamma_effect_{FIG_LABEL[beta]}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}\n")


def main():
    for beta in GAMMAS:
        make_figure(beta)


if __name__ == "__main__":
    main()
