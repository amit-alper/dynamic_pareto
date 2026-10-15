"""3D trajectory grid: rows = gamma, columns = e/b (same panel style as geometry_effect.png).
Uses the run_perp cache of paper figures/3 tragets source perpendicular; missing runs are simulated."""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
PERP = os.path.join(HERE, "..", "..", "paper figures", "3 tragets source perpendicular")
sys.path.insert(0, PERP)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from perpendicular_common import (archetypes3, source_point, z_critical, run_perp, sample_lines,
                                  shared_lims_3d, plot_perp_panel, topology_caption, classify, B, max_safe_h)

BETA, H = 1.6, 2.5
GAMMAS = [float(g) for g in os.environ.get("GAMMAS", "1.0,1.3,1.6,2.0").split(",")]
E_RATIOS = [0.6, 1.0, 1.73, 2.0, 2.5, 3.0, 4.0]
N_SEEDS = 8

def main():
    panels = {}
    for g in GAMMAS:
        h = min(H, max_safe_h(g, BETA, 1.0)) if g > 1.3 else H
        h = float(os.environ.get("H_OVERRIDE", h))
        for e in E_RATIOS:
            t0 = time.time()
            A = archetypes3(e, B); src = source_point(e, h)
            trajs = [run_perp(s, g, BETA, e, B, h, sigma=float(os.environ.get('SIGMA', 0.01))) for s in range(N_SEEDS)]
            fates = classify(np.concatenate([t[-1] for t in trajs]), A)
            lines, lf = sample_lines(trajs, A, max_lines=44, rng=np.random.default_rng(0))
            panels[(g, e)] = (A, src, z_critical(g, BETA, e, B), lines, lf, fates, h)
            print(f"gamma={g} e/b={e} h={h:.2f} {time.time()-t0:.0f}s", flush=True)
    pts = []
    for A, src, zc, lines, lf, fates, h in panels.values():
        pts += [A, src[None, :]] + list(lines)
    lims = shared_lims_3d(pts, pad=1.08)
    nr, nc = len(GAMMAS), len(E_RATIOS)
    fig = plt.figure(figsize=(4.2 * nc, 4.0 * nr))
    for r, g in enumerate(GAMMAS):
        for c, e in enumerate(E_RATIOS):
            A, src, zc, lines, lf, fates, h = panels[(g, e)]
            ax = fig.add_subplot(nr, nc, r * nc + c + 1, projection="3d")
            plot_perp_panel(ax, A, src, lines, lf, zc=zc, lims=lims, show_legend=(r == 0 and c == 0),
                            title=rf"$\gamma$={g:g}, $e/b$={e:g}, $z_c$={zc:.2f}" + "\n" + topology_caption(fates),
                            title_fontsize=10)
    fig.suptitle(rf"Perpendicular source: trajectories vs $e/b$ (columns) and $\gamma$ (rows), $\beta$={BETA:g}", fontsize=16, y=0.995)
    out = os.path.join(HERE, os.environ.get("OUT", "trajectory_grid_gamma_eb.png"))
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white"); print("saved", out)

if __name__ == "__main__":
    main()
