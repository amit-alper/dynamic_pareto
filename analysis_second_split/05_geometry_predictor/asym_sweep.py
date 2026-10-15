"""a2 slides along the base toward (and past) the apex foot: a1=(-1,0), a2=(s,0), a3=(0,2); source below the centroid. Per-seed first/second split."""
import os, sys, pickle
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from general import sim_one, pred
GAMMA, BETA, H, SIGMA, NS = 1.3, 1.6, 1.35, 0.02, 12
SV = [1.0, 0.75, 0.5, 0.25, 0.0, -0.25, -0.5]
def geo(s):
    A = np.array([[-1, 0, 0], [s, 0, 0], [0, 2.0, 0]], float); return A, np.array([*A[:, :2].mean(0), -H])
def job(a):
    s, seed = a; A, S = geo(s); return sim_one((A, S, GAMMA, BETA, SIGMA, seed))
if __name__ == "__main__":
    tasks = [(s, sd) for s in SV for sd in range(NS)]
    with Pool(10) as p: res = p.map(job, tasks)
    out = {}
    for (s, sd), r in zip(tasks, res): out.setdefault(s, []).append(r)
    pickle.dump(out, open("asym_sweep.pkl", "wb"))
    for s in SV:
        z1 = np.array([r["z1"] for r in out[s]]); z2 = np.array([r["z2"] for r in out[s]]); sh = np.mean([r["shares"] for r in out[s]], 0)
        print(f"s={s:+.2f} d12={1+s:.2f} z1 {np.nanmean(z1):+.3f}±{np.nanstd(z1):.3f} z2 {np.nanmean(z2):+.3f}±{np.nanstd(z2):.3f} (found {np.isfinite(z2).mean():.2f}) shares {np.round(sh,2)}")
