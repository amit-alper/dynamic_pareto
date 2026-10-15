"""
Is there a second split in the non-equilateral perpendicular three-target geometry?

Uses the cached simulations of paper figures/3 tragets source perpendicular (geometry_effect.py:
gamma=1.3, beta=1.6, h=2.5, N=40, sigma=0.01, 8 seeds, e/b in E_RATIOS). No new simulation is run
unless the cache is missing. For each run the cells are grouped by the archetype they end at
(A=a1, B=a2, C=apex a3). Pairwise distances between group centroids give the split times:
the first pair to separate by DELTA is split 1, the last pair is split 2. Theory: linear lateral
eigenvalue along the coherent path in e_x (base pair) and e_y (toward apex).
"""
import os, sys, io, contextlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PERP = os.path.join(ROOT, "paper figures", "3 tragets source perpendicular")
sys.path.insert(0, ROOT); sys.path.insert(0, PERP)
from perpendicular_common import archetypes3, source_point, run_perp, classify
from core.bifurcation import theoretical_bifurcation_point
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

GAMMA, BETA, H = 1.3, 1.6, 2.5
E_RATIOS = [0.6, 1.0, 1.73, 2.0, 2.5, 3.0]
SEEDS = range(8)
T, K, SIGMA, N = 800.0, 40000, 0.01, 40
DELTA = float(os.environ.get("DELTA", 0.15))
STRIDE = 10


def split_events(traj, A, delta=DELTA):
    fates = classify(traj[-1], A)
    groups = [np.where(fates == g)[0] for g in range(3)]
    if min(len(g) for g in groups) == 0:
        return None
    tr = traj[::STRIDE]
    cen = np.stack([tr[:, g, :].mean(1) for g in groups], 1)           # (k,3,3)
    z = tr[:, :, 2].mean(1)
    pairs = [(0, 1), (0, 2), (1, 2)]
    times = {}
    for (i, j) in pairs:
        d = np.linalg.norm(cen[:, i] - cen[:, j], axis=1)
        idx = np.argmax(d > delta) if (d > delta).any() else None
        times[(i, j)] = None if idx is None else idx
    if any(v is None for v in times.values()):
        return None
    order = sorted(times.items(), key=lambda kv: kv[1])
    i1 = order[0][1]; i2 = order[-1][1]
    # lone leaf of split 1: the leaf common to the two earliest pairs
    first_two = [set(order[0][0]), set(order[1][0])]
    lone = list(first_two[0] & first_two[1])[0]
    return dict(i1=i1, i2=i2, z1=z[i1], z2=z[i2], lone=lone, n=[len(g) for g in groups],
                dt=STRIDE * T / K, t1=i1 * STRIDE * T / K, t2=i2 * STRIDE * T / K,
                pair_idx=times, lat_x=None)


def lateral_cross(traj, dim, thr=0.15):
    s = traj[::STRIDE][:, :, dim].std(1)
    z = traj[::STRIDE][:, :, 2].mean(1)
    idx = np.argmax(s > thr) if (s > thr).any() else None
    return None if idx is None else (idx * STRIDE * T / K, z[idx])


def theory(e):
    A = archetypes3(e); src = source_point(e, H)
    out = {}
    for name, d in [("x", (1, 0, 0)), ("y", (0, 1, 0))]:
        th = theoretical_bifurcation_point(src, A, GAMMA, beta=BETA, N=N, T=T, K=K, sigma=SIGMA,
                                           x_partial=DELTA, height_dim=-1, direction=np.array(d, float))
        out[name] = dict(t=th["t_exact"], z=th["height_exact"], zsign=th["height_sign"])
    return out


def main():
    rows = []
    for e in E_RATIOS:
        A = archetypes3(e)
        th = theory(e)
        ev, lx, ly = [], [], []
        for s in SEEDS:
            with contextlib.redirect_stdout(io.StringIO()):
                traj = run_perp(s, GAMMA, BETA, e, 1.0, H)
            r = split_events(traj, A); 
            if r: ev.append(r)
            lx.append(lateral_cross(traj, 0)); ly.append(lateral_cross(traj, 1))
        rows.append((e, th, ev, lx, ly))
        print(f"\n=== e/b={e}  theory: z*_x={th['x']['z']}  z*_y={th['y']['z']}  "
              f"(sign z_x={th['x']['zsign']}, z_y={th['y']['zsign']})")
        print(f"    usable seeds (all 3 groups populated): {len(ev)}/{len(list(SEEDS))}")
        for r in ev:
            print(f"    lone leaf={'A B C'.split()[r['lone']]}  t1={r['t1']:.1f} z1={r['z1']:+.2f}   "
                  f"t2={r['t2']:.1f} z2={r['z2']:+.2f}   gap_t={r['t2']-r['t1']:.1f}  sizes={r['n']}")
    # figure
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for e, th, ev, lx, ly in rows:
        if ev:
            axs[0].scatter([e] * len(ev), [r["z1"] for r in ev], c="C0", s=18, label="split 1" if e == E_RATIOS[0] else None)
            axs[0].scatter([e] * len(ev), [r["z2"] for r in ev], c="C3", s=18, marker="^", label="split 2" if e == E_RATIOS[0] else None)
        for k, c in (("x", "C2"), ("y", "C1")):
            if th[k]["z"] is not None:
                axs[0].scatter([e], [th[k]["z"]], c=c, marker="D", s=50, edgecolors="k",
                               label=f"linear theory, mode {k}" if e == E_RATIOS[0] else None)
    axs[0].set_xlabel("e/b"); axs[0].set_ylabel("height z of the population at the split"); axs[0].legend(fontsize=8)
    axs[0].set_title(f"split heights (delta={DELTA})")
    for e, th, ev, lx, ly in rows:
        g = [r["t2"] - r["t1"] for r in ev]
        axs[1].scatter([e] * len(g), g, c="C3", s=18)
    axs[1].set_xlabel("e/b"); axs[1].set_ylabel("time between split 1 and split 2"); axs[1].set_title("separation of the two splits")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, f"second_split_delta{DELTA}.png"), dpi=150)


if __name__ == "__main__":
    main()
