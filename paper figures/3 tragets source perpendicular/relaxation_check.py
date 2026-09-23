"""
Has the "partial" (generalist-curve / interior-filling) population in
gamma_effect_beta_lt1.png and gamma_effect_beta_lt1_equilateral.png actually
relaxed to its final resting place by T=800, or is it still mid-transit off
the transversally-marginal interior saddle (0,y*) at that point?

Mechanism under test (see conversation write-up): at beta=0.85, the in-plane
interior fixed point (0,y*) has a transverse eigenvalue lambda_x(y*) very
close to its local threshold beta_c(y*,gamma) for gamma in [1.3,2.5] --
either barely stable or barely unstable. Either way the relaxation rate is
small, so cells may simply not have had time to finish sorting into tight
clusters within the standard T=800,K=40000 horizon (dt=0.02).

This script re-runs the SAME seeds/gammas/geometries at ~10x the horizon
(T=8000, K=400000, same dt=0.02, matching the "hold dt fixed, extend T"
convention already used for slow panels -- see gamma_effect.py notes) and
compares:
  (1) fate fractions (archetype-arrived / coherent / partial) at T=800 vs
      T=8000 for the same seeds -- if relaxation were complete already,
      these should barely move.
  (2) the density point-cloud itself, short vs long horizon, side by side.
  (3) |x(t)| for a handful of individual "partial" cells across the full
      long horizon, to see directly whether they are still drifting or
      have plateaued well before T=8000.

Run from anywhere (this is slow -- ~25-30 min, mostly the T=8000 runs):
    python "paper figures/3 tragets source perpendicular/relaxation_check.py"
"""

import numpy as np
import matplotlib.pyplot as plt

from perpendicular_common import (
    archetypes3, source_point, run_perp, classify, fate_summary,
    trimmed_points, plot_perp_density_panel_3d, shared_lims_3d,
    z_critical, HERE,
)

SEEDS = list(range(6))
BETA = 0.85
T_SHORT, K_SHORT = 800.0, 40000     # matches gamma_effect_beta_lt1*.png
T_LONG, K_LONG = 8000.0, 400000     # 10x horizon, same dt=0.02

CONFIGS = {
    "e1.0": dict(e=1.0, h=2.1, gammas=[1.5, 1.9, 2.5],
                 title=r"$e/b=1.0$ (gamma\_effect\_beta\_lt1.png)"),
    "equilateral": dict(e=np.sqrt(3.0), h=2.1, gammas=[1.3, 1.9, 2.5],
                         title=r"equilateral (gamma\_effect\_beta\_lt1\_equilateral.png)"),
}


def fractions(finals, A):
    fates = classify(finals, A)
    counts, coherent, partial, n = fate_summary(fates, k=3)
    return dict(archetype=counts.sum() / n, coherent=coherent / n, partial=partial / n)


def run_both_horizons(e, b, gamma, h):
    A = archetypes3(e, b)
    trajs_short = [run_perp(s, gamma, BETA, e, b, h, T=T_SHORT, K=K_SHORT) for s in SEEDS]
    trajs_long = [run_perp(s, gamma, BETA, e, b, h, T=T_LONG, K=K_LONG) for s in SEEDS]
    finals_short = np.concatenate([t[-1] for t in trajs_short], axis=0)
    finals_long = np.concatenate([t[-1] for t in trajs_long], axis=0)
    return A, trajs_short, trajs_long, finals_short, finals_long


def make_fraction_table():
    lines = []
    lines.append(f"{'geom':<13}{'gamma':>6}{'':>3}"
                  f"{'arch(T=800)':>12}{'coh(800)':>10}{'part(800)':>10}   ->   "
                  f"{'arch(T=8000)':>13}{'coh(8000)':>11}{'part(8000)':>11}")
    results = {}
    for label, cfg in CONFIGS.items():
        e, b, h = cfg["e"], 1.0, cfg["h"]
        for gamma in cfg["gammas"]:
            A, ts, tl, fs, fl = run_both_horizons(e, b, gamma, h)
            fr_s, fr_l = fractions(fs, A), fractions(fl, A)
            results[(label, gamma)] = dict(A=A, trajs_short=ts, trajs_long=tl,
                                            finals_short=fs, finals_long=fl,
                                            frac_short=fr_s, frac_long=fr_l)
            lines.append(
                f"{label:<13}{gamma:>6.2f}{'':>3}"
                f"{fr_s['archetype']:>12.3f}{fr_s['coherent']:>10.3f}{fr_s['partial']:>10.3f}   ->   "
                f"{fr_l['archetype']:>13.3f}{fr_l['coherent']:>11.3f}{fr_l['partial']:>11.3f}"
            )
    table_str = "\n".join(lines)
    print(table_str)
    with open(f"{HERE}/relaxation_check_table.txt", "w") as f:
        f.write(table_str + "\n")
    return results


def plot_density_comparison(results):
    n = sum(len(cfg["gammas"]) for cfg in CONFIGS.values())
    fig = plt.figure(figsize=(11, 5.4 * n))
    row = 0
    for label, cfg in CONFIGS.items():
        e, b, h = cfg["e"], 1.0, cfg["h"]
        for gamma in cfg["gammas"]:
            r = results[(label, gamma)]
            A = r["A"]
            src = source_point(e, h)
            zc = z_critical(gamma, BETA, e, b)
            cloud_short = trimmed_points(r["trajs_short"], A, stride=10)
            cloud_long = trimmed_points(r["trajs_long"], A, stride=100)
            lims = shared_lims_3d([A, src[None, :], cloud_short, cloud_long], pad=1.12)

            ax1 = fig.add_subplot(n, 2, 2 * row + 1, projection="3d")
            t1 = (rf"{cfg['title']}, $\gamma$={gamma:g}, T=800 (original)" + "\n" +
                  f"arch={r['frac_short']['archetype']:.2f}  coh={r['frac_short']['coherent']:.2f}  "
                  f"part={r['frac_short']['partial']:.2f}")
            plot_perp_density_panel_3d(ax1, A, src, cloud_short, zc=zc, title=t1,
                                        lims=lims, title_fontsize=9)

            ax2 = fig.add_subplot(n, 2, 2 * row + 2, projection="3d")
            t2 = (rf"same seeds, T=8000 (10x horizon)" + "\n" +
                  f"arch={r['frac_long']['archetype']:.2f}  coh={r['frac_long']['coherent']:.2f}  "
                  f"part={r['frac_long']['partial']:.2f}")
            plot_perp_density_panel_3d(ax2, A, src, cloud_long, zc=zc, title=t2,
                                        lims=lims, title_fontsize=9)
            row += 1

    fig.suptitle(r"Relaxation check: $\beta$=0.85, same seeds, 10x horizon (T=800 $\to$ T=8000, dt fixed)"
                 "\nIf the cloud/fractions barely change, the interior spread is a genuine "
                 "(quasi-)equilibrium, not a slow transient", fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = f"{HERE}/relaxation_check_density.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


def plot_trajectory_traces(results):
    """|x(t)| for a handful of 'partial' cells across the full T=8000 horizon,
    on a log-t axis, with T=800 marked -- direct visual answer to 'still
    moving, or plateaued already by the original horizon?'"""
    configs_to_trace = [("e1.0", 2.5), ("e1.0", 1.9), ("equilateral", 2.5), ("equilateral", 1.3)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (label, gamma) in zip(axes.ravel(), configs_to_trace):
        r = results[(label, gamma)]
        A = r["A"]
        traj = r["trajs_long"][0]  # (K+1, N, 3), seed 0
        finals = traj[-1]
        fates = classify(finals, A)
        partial_idx = np.where(fates == -2)[0]
        if len(partial_idx) == 0:
            partial_idx = np.where(fates == -1)[0]
        show = partial_idx[: min(10, len(partial_idx))]
        ts = np.linspace(0, T_LONG, traj.shape[0])
        for j in show:
            ax.plot(ts, np.abs(traj[:, j, 0]), lw=0.9, alpha=0.7)
        ax.axvline(T_SHORT, color="red", ls="--", lw=1.2, label="original horizon T=800")
        ax.set_xscale("log")
        ax.set_xlabel("t (log scale)")
        ax.set_ylabel("|x(t)|")
        ax.set_title(f"{label}, $\\gamma$={gamma:g}: |x(t)| for cells classified\n"
                     "'partial' (or 'coherent') at T=8000, seed 0", fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Are these cells still sliding, or have they plateaued?", fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = f"{HERE}/relaxation_check_traces.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


def main():
    results = make_fraction_table()
    plot_density_comparison(results)
    plot_trajectory_traces(results)


if __name__ == "__main__":
    main()
