"""Sweep geometric asymmetries at fixed (gamma, beta); compare geometry-only prediction with population simulation."""
import os, sys, pickle
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predictor import predict, static_score, XP
from sim import run_summary, first_cross

GAMMA, BETA, H, SIGMA, NSEED = float(os.environ.get("GAMMA", 1.3)), float(os.environ.get("BETA", 1.6)), 1.35, 0.02, 12


def geom(e=2.0, sf=1/3, ax=0.0, sx=None):
    """base pair (+-1,0,0); apex (ax,e,0); source below (sx, sf*e, -H) (sx defaults to the centroid x = ax/3)"""
    A = np.array([[-1, 0, 0], [1, 0, 0], [ax, e, 0]], float)
    S = np.array([ax / 3 if sx is None else sx, sf * e, -H])
    return A, S


SWEEPS = {
    "elongation e/b": ("e/b", [0.6, 1.0, 1.4, 1.73, 2.0, 2.5, 3.0, 3.5, 4.0], lambda v: geom(e=v)),
    "source offset (y_s / e)": ("y_s/e", [0.0, 0.17, 0.33, 0.5, 0.67, 0.83, 1.0], lambda v: geom(e=2.0, sf=v)),
    "apex shifted sideways (x of a3)": ("x_a3/b", [0.0, 0.25, 0.5, 0.75, 1.0], lambda v: geom(e=2.0, ax=v)),
}


def main():
    res = {}
    with Pool(12) as pool:
        for name, (lab, vals, f) in SWEEPS.items():
            rows = []
            for v in vals:
                A, S = f(v)
                runs = pool.map(run_summary, [(A, S, GAMMA, BETA, SIGMA, s) for s in range(NSEED)])
                zx = np.array([first_cross(r["z"], r["sx"], XP) for r in runs]); zy = np.array([first_cross(r["z"], r["sy"], XP) for r in runs])
                fr = np.mean([np.bincount(r["fates"], minlength=3) for r in runs], 0) / 40
                p = predict(A, S, GAMMA, BETA, sigma=SIGMA)
                row = dict(v=v, sim_x=np.nanmedian(zx), sim_y=np.nanmedian(zy), fx=np.mean(~np.isnan(zx)), fy=np.mean(~np.isnan(zy)),
                           pred_x=np.nan if p["x"] is None else p["x"], pred_y=np.nan if p["y"] is None else p["y"],
                           score=static_score(A, S, GAMMA, BETA), shares=fr)
                rows.append(row)
                print(f"{name:34s} {lab}={v:5.2f}  sim zx={row['sim_x']:6.2f} zy={row['sim_y']:6.2f} (found {row['fx']:.2f}/{row['fy']:.2f})   "
                      f"pred zx={row['pred_x']:6.2f} zy={row['pred_y']:6.2f}   score={row['score']:+.3f}  shares={np.round(fr,2)}", flush=True)
            res[name] = (lab, rows)
    pickle.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"sweeps_g{GAMMA}_b{BETA}.pkl"), "wb"))


if __name__ == "__main__":
    main()
