import os, sys, pickle
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from general import *
GAMMA, BETA, H, SIGMA, NS = float(os.environ.get("GAMMA", 1.3)), float(os.environ.get("BETA", 1.6)), 1.35, 0.02, 12
cen = lambda A: A[:, :2].mean(0)
def mk(A, s=None):
    A = np.array(A, float); c = cen(A) if s is None else np.array(s, float)
    return A, np.array([c[0], c[1], -H])
SW = {   # name -> (label, values, builder(v)->(A,S))
 "base width w (apex e=2)": ("w", [0.3, 0.5, 0.75, 1.0, 1.5], lambda v: mk([[-v, 0, 0], [v, 0, 0], [0, 2, 0]])),
 "a2 slides toward a1 (a1=-1, a2=s, apex over 0)": ("s", [1.0, 0.75, 0.5, 0.25, 0.0], lambda v: mk([[-1, 0, 0], [v, 0, 0], [0, 2, 0]])),
 "source in plane, x offset (symmetric tri e=2)": ("x_s", [0.0, 0.25, 0.5, 0.75, 1.0], lambda v: mk([[-1, 0, 0], [1, 0, 0], [0, 2, 0]], [v, 2 / 3])),
 "source in plane, x offset (a2 at 0.5)": ("x_s", [-0.5, -0.25, 0.0, 0.25, 0.5], lambda v: mk([[-1, 0, 0], [0.5, 0, 0], [0, 2, 0]], [v, 2 / 3])),
}
def main():
    res = {}
    with Pool(12) as pool:
        for name, (lab, vals, f) in SW.items():
            rows = []
            for v in vals:
                A, S = f(v); runs = pool.map(sim_one, [(A, S, GAMMA, BETA, SIGMA, s) for s in range(NS)])
                z1 = np.array([r["z1"] for r in runs]); z2 = np.array([r["z2"] for r in runs]); th = np.array([r["th"] for r in runs])
                p = pred(A, S, GAMMA, BETA, SIGMA)
                row = dict(v=v, s1=np.nanmedian(z1), s2=np.nanmedian(z2), sth=np.degrees(circ_mean_axis(th)), p1=p[0], p2=p[1], pth=np.degrees(p[2]) if not np.isnan(p[2]) else np.nan,
                           shares=np.mean([r["shares"] for r in runs], 0), A=A, S=S)
                rows.append(row)
                print(f"{name[:34]:34s} {lab}={v:5.2f} sim z1={row['s1']:6.2f} z2={row['s2']:6.2f} ax={row['sth']:6.1f}° | pred z1={row['p1']:6.2f} z2={row['p2']:6.2f} ax={row['pth']:6.1f}° | shares={np.round(row['shares'],2)}", flush=True)
            res[name] = (lab, rows)
    pickle.dump(res, open(os.path.join(HERE, f"unequal_g{GAMMA}_b{BETA}.pkl"), "wb"))
HERE = os.path.dirname(os.path.abspath(__file__))
if __name__ == "__main__": main()
