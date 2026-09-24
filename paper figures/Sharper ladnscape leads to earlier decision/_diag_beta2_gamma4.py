"""Diagnostic: measure the real timescales at beta=2, gamma=4 before touching T.

For a handful of seeds, report per-cell:
  - departure time   : first time the cell leaves radius r_src of the source
  - arrival time      : first time within R of an archetype (R = 0.2 * arm dist)
  - fraction arrived  by t = T
  - lateral Var[x](t) : where the split saturates
so we can pick T (and K for Euler stability) from numbers, not guesswork.
"""

import os, sys, io, contextlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from greedy.greedy import greedy_trajectory

A   = np.array([[-1.0, 0.0], [1.0, 0.0]])
SRC = np.array([0.0, -1.3])
N, K, T, SIGMA = 70, int(os.environ.get("KK", 40000)), float(os.environ.get("TT", 800.0)), 0.005
BETA, GAMMA = 2.0, 4.0
SEEDS = range(int(os.environ.get("NS", 6)))

R      = 0.20 * 2.0        # arrival radius (arm distance = 2.0)
R_SRC  = 0.30             # "left the source" radius
dt     = T / K
t_axis = np.arange(K + 1) * dt

dep_all, arr_all, arrived_frac = [], [], []
latvar_last = None
for seed in SEEDS:
    with contextlib.redirect_stdout(io.StringIO()):
        res = greedy_trajectory(SRC, A, GAMMA, N, K, T, mode="exp", log=True,
                                sigma=SIGMA, seed=seed, beta=BETA)
    traj = res["trajectory"]                                   # (K+1, N, 2)

    d_src = np.linalg.norm(traj - SRC[None, None, :], axis=-1)          # (K+1, N)
    d_arch = np.linalg.norm(traj[:, :, None, :] - A[None, None, :, :], axis=-1).min(2)  # (K+1,N)

    left    = d_src > R_SRC
    arrived = d_arch < R
    dep = np.where(left.any(0),    left.argmax(0),    K)      # step index
    arr = np.where(arrived.any(0), arrived.argmax(0), K)

    dep_all.append(dep * dt); arr_all.append(arr * dt)
    arrived_frac.append(arrived.any(0).mean())
    latvar_last = res["lateral_var"]

dep_all = np.concatenate(dep_all); arr_all = np.concatenate(arr_all)

def pct(x): return np.percentile(x, [10, 50, 90, 100])

print(f"beta={BETA}, gamma={GAMMA}, T={T}, K={K}, dt={dt:g}, sigma={SIGMA}, N={N}, {len(SEEDS)} seeds")
print(f"arrival radius R={R}, source-exit radius={R_SRC}")
print()
print(f"departure time  [p10 p50 p90 max] = {pct(dep_all)}")
print(f"arrival   time  [p10 p50 p90 max] = {pct(arr_all)}")
print(f"fraction arrived by t=T           = {np.mean(arrived_frac):.3f}  (per seed: {np.round(arrived_frac,3)})")
print()
# lateral variance: when does the split saturate?
lv = latvar_last / max(latvar_last.max(), 1e-12)
reach = lambda f: t_axis[np.argmax(lv > f)] if (lv > f).any() else np.nan
print(f"lateral Var[x] reaches 10% / 50% / 90% of its max at t = "
      f"{reach(0.1):.1f} / {reach(0.5):.1f} / {reach(0.9):.1f}")
print(f"lateral Var[x] max = {latvar_last.max():.4f} at t = {t_axis[latvar_last.argmax()]:.1f}")
