"""
3D counterpart of diagnostic_spatial_trajectories.py for the perpendicular
three-target geometry: actual N-cell trajectories in real (x,y,z) space, with
every cell's position at the exact detected-split timestep marked on top.

beta=1.6 only for now (per request).

Run from anywhere:
    python "paper figures/Bifurcation quantification/diagnostic_spatial_trajectories_perp3d.py"
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PERP_DIR = os.path.join(ROOT, "paper figures", "3 tragets source perpendicular")
if PERP_DIR not in sys.path:
    sys.path.insert(0, PERP_DIR)

from core.bifurcation import run_single_trajectory, theoretical_bifurcation_point
import perpendicular_common as perp

B = 1.0
E_EQ = np.sqrt(3.0) * B
BETA = 1.6
H = 1.35
GAMMAS = [0.8, 1.2, 1.6, 2.0, 2.5, 3.0]

POP_N, SIGMA, X_PARTIAL = perp.N_DEFAULT, perp.SIGMA_DEFAULT, perp.X_PARTIAL
T, K = perp.T_DEFAULT / 2.0, perp.K_DEFAULT // 2
DIRECTION = (1.0, 0.0, 0.0)
SEED = 0

LABELS = (r"$a_1$", r"$a_2$", r"$a_3$")
ARCH_COLORS = ("#1f77b4", "#d62728", "#2ca02c")


def draw_panel(ax, archetypes, source, traj, step_detect, height_detect, title):
    tri = Poly3DCollection([archetypes], facecolor="#a9cce3", edgecolor="none", alpha=0.20)
    ax.add_collection3d(tri)
    for i in range(3):
        p, q = archetypes[i], archetypes[(i + 1) % 3]
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color="#1a3d63", lw=1.2, zorder=2)
    for pt, lab, col in zip(archetypes, LABELS, ARCH_COLORS):
        ax.scatter(*pt, s=45, marker="*", color="gold", edgecolor="black", linewidth=0.6, zorder=6)
        ax.text(pt[0], pt[1], pt[2], "  " + lab, fontsize=9, zorder=7)
    ax.scatter(*source, s=55, marker="^", color="#2ecc71", edgecolor="black", linewidth=0.6, zorder=6)

    final_x = traj[-1, :, 0]
    colors = np.where(final_x < -0.15, ARCH_COLORS[0], np.where(final_x > 0.15, ARCH_COLORS[1], "0.5"))
    for j in range(traj.shape[1]):
        ax.plot(traj[:, j, 0], traj[:, j, 1], traj[:, j, 2], color=colors[j], lw=0.5, alpha=0.3, zorder=3)

    if step_detect is not None:
        pts = traj[step_detect]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=18, color="black", zorder=7)

    ax.set_xlabel("x", fontsize=8, labelpad=-6)
    ax.set_ylabel("y", fontsize=8, labelpad=-6)
    ax.set_zlabel("z", fontsize=8, labelpad=-6)
    ax.tick_params(labelsize=7, pad=0)
    ax.view_init(elev=14, azim=-55)
    ax.set_title(title, fontsize=10.5, pad=0)


def main():
    archetypes = perp.archetypes3(E_EQ, B)
    source = perp.source_point(E_EQ, H)

    ncols = 3
    nrows = -(-len(GAMMAS) // ncols)
    fig = plt.figure(figsize=(6.0 * ncols, 5.8 * nrows))

    for i, gamma in enumerate(GAMMAS):
        r_stop = 0.05 if gamma < 1.0 else None
        theory = theoretical_bifurcation_point(source, archetypes, gamma, beta=BETA,
                                                N=POP_N, T=T, K=K, sigma=SIGMA,
                                                x_partial=X_PARTIAL, direction=DIRECTION)
        run = run_single_trajectory(source, archetypes, gamma, beta=BETA, pop_N=POP_N,
                                     T=T, K=K, sigma=SIGMA, x_partial=X_PARTIAL,
                                     seed=SEED, r_stop=r_stop)

        n_txt = f"split at t={run['t_detect']:.2f}" if run["t_detect"] is not None else "no split by T"
        title = rf"$\gamma$={gamma:g}  ({n_txt})"
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        draw_panel(ax, archetypes, source, run["trajectory"], run["step_detect"],
                    run["height_detect"], title)

    fig.suptitle(rf"Actual spatial trajectories with detected split marked, "
                  rf"$\beta$={BETA:g} (perpendicular 3-target, equilateral, seed={SEED})",
                  fontsize=13, y=0.995)
    caption = (
        f"Equilateral triangle ($e=\\sqrt{{3}}\\,b$, $b=1$), source at depth $h$={H:g} "
        f"on the centroid line (green triangle), N={POP_N} cells, $\\sigma$={SIGMA:g}, "
        f"T={T:g}, K={K}, single representative seed. Thin lines: every cell's full 3D "
        "path, colored by which base-pair archetype it ends up nearer to (gray = still "
        "near center). Black dots: every cell's actual position at the exact timestep "
        "detected as 'split' (first crossing of the x_partial=0.2 lateral-std threshold)."
    )
    fig.text(0.5, 0.0, caption, ha="center", fontsize=9, color="0.3", wrap=True)
    fig.subplots_adjust(top=0.93, bottom=0.08, hspace=0.25, wspace=0.05)

    out = os.path.join(HERE, "diagnostic_spatial_trajectories_perp3d.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
