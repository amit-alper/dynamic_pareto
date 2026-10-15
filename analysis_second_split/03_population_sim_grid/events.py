"""Event extraction from the derived run records (rungrid.derive) and per-cell aggregation."""
import os, sys, pickle, glob
import numpy as np
from scipy.stats import binom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simcore import HERE
from core.cache import _params_hash
import rungrid as rg
import rungrid3 as r3

PAIRS = [("A", "B"), ("A", "C"), ("B", "C")]       # A=a1, B=a2 (base pair), C=a3 (apex)
DELTAS = (0.10, 0.15, 0.25)
MIN_GROUP = 2
GAP_RATIO = 1.3      # gap must exceed this multiple of the equilateral-control mean gap
P_ORDER = 0.01       # binomial p-value for a consistent last-separating pair
ARREST_Z = -0.3        # mean final height below this: population arrested short of the plane (leader-take-all)
MIN_FINAL_SEP = 0.6    # median over seeds of the smallest final group-centroid separation: below = no complete 3-way split


def load_run(e, g, b, h, s, early=True, version=3):
    params = r3.params3(e, g, b, h, s, early) if version == 3 else dict(
        kind="sim3d", v=rg.VERSION, e=e, gamma=g, beta=b, h=h, seed=s, N=rg.N, T=rg.T, K=rg.K,
        sigma=rg.SIGMA, stride=rg.STRIDE, early=early)
    path = os.path.join(rg.CACHE, f"sim_{_params_hash(params)}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)["result"]


def _first_cross(t, x, thr):
    idx = np.where(x > thr)[0]
    return None if len(idx) == 0 else int(idx[0])


def events(rec, delta):
    """Pair-separation events of the final-fate groups. Returns None if degenerate (a group has < MIN_GROUP
    cells) or a pair never separates by `delta`."""
    if rec["sizes"].min() < MIN_GROUP:
        return None
    t, z, sep = rec["t"], rec["z"], rec["sep"]
    idx = []
    for k in range(3):
        i = _first_cross(t, sep[:, k], delta)
        if i is None:
            return None
        idx.append(i)
    idx = np.array(idx)
    order = np.argsort(idx, kind="stable")
    tp = t[idx]; zp = z[idx]
    loc = {}
    if "pos" in rec:
        A = rec["A"]; e_ = float(A[2, 1])
        for nm, i in (("1", int(idx.min())), ("2", int(idx.max()))):
            p = rec["pos"][i].astype(float)
            d = np.linalg.norm(A - p[None], axis=1)
            loc.update({"x" + nm: p[0], "y" + nm: p[1], "d_apex" + nm: d[2], "d_base" + nm: d[:2].min(),
                        "dist_to_plane" + nm: abs(p[2]), "disp" + nm: float(np.hypot(p[0], p[1] - e_ / 3.0))})
    return dict(loc=loc, tp=tp, zp=zp, t1=float(tp.min()), t2=float(tp.max()), z1=float(zp[idx.argmin()]),
                z2=float(zp[idx.argmax()]), gap=float(tp.max() - tp.min()), last=int(idx.argmax()),
                first=int(idx.argmin()), tp_sorted=tp[order])


def mode_cross(rec, thr=0.15):
    t, z = rec["t"], rec["z"]
    out = {}
    for k in ("stdx", "stdy"):
        i = _first_cross(t, rec[k], thr)
        out[k] = (None, None) if i is None else (float(t[i]), float(z[i]))
    return out


def load_cell(e, g, b, h, seeds=None, early=True):
    seeds = rg.SEEDS if seeds is None else seeds
    return [r for r in (load_run(e, g, b, h, s, early) for s in seeds) if r is not None]


def aggregate(h=2.5, delta=0.15, e_list=None, gammas=None, betas=None, floor_q=95):
    """One row (dict) per (e,gamma,beta)."""
    e_list = e_list or rg.E_LIST; gammas = gammas or rg.GAMMAS; betas = betas or rg.BETAS
    eq = 3 ** 0.5
    rows = []
    # noise floor of the gap from the equilateral control at each (gamma,beta)
    floor = {}; ctrl = {}
    for g in gammas:
        for b in betas:
            gaps = [ev["gap"] for r in load_cell(eq, g, b, h) if (ev := events(r, delta))]
            floor[(g, b)] = float(np.percentile(gaps, floor_q)) if len(gaps) >= 6 else np.nan
            ctrl[(g, b)] = float(np.mean(gaps)) if len(gaps) >= 6 else np.nan
    for e in e_list:
        for g in gammas:
            for b in betas:
                recs = load_cell(e, g, b, h)
                evs = [events(r, delta) for r in recs]
                ok = [(r, ev) for r, ev in zip(recs, evs) if ev]
                row = dict(e=e, gamma=g, beta=b, h=h, delta=delta, n_runs=len(recs), n_valid=len(ok),
                           floor=floor[(g, b)], z_end=float(np.mean([r["z"][-1] for r in recs])) if recs else np.nan,
                           T_run=float(np.mean([r["T_run"] for r in recs])) if recs else np.nan,
                           frac_apex=float(np.mean([r["sizes"][2] for r in recs]) / rg.N) if recs else np.nan)
                zend = row["z_end"]
                minsep = float(np.median([r["sep"][-1].min() for r in recs])) if recs else np.nan
                row["min_final_sep"] = minsep
                special = "arrested" if (recs and zend < ARREST_Z) else ("no split" if (recs and minsep < MIN_FINAL_SEP) else None)
                if len(ok) >= 8 and np.isfinite(ctrl[(g, b)]):
                    gap = np.array([ev["gap"] for _, ev in ok]); last = np.array([ev["last"] for _, ev in ok])
                    z1 = np.array([ev["z1"] for _, ev in ok]); z2 = np.array([ev["z2"] for _, ev in ok])
                    t1 = np.array([ev["t1"] for _, ev in ok]); t2 = np.array([ev["t2"] for _, ev in ok])
                    k_ap = int(np.sum(last == 0)); n_ok = len(last); k_bp = n_ok - k_ap
                    p_apex = float(binom.sf(k_ap - 1, n_ok, 1 / 3)); p_base = float(binom.sf(k_bp - 1, n_ok, 2 / 3))
                    ratio = float(gap.mean() / ctrl[(g, b)])
                    sig = ratio >= GAP_RATIO
                    frac_sig = float(sig)
                    if sig and p_apex < P_ORDER: typ = "apex-first"
                    elif sig and p_base < P_ORDER: typ = "base-peel"
                    elif sig: typ = "mixed"
                    else: typ = "simultaneous"
                    row.update(p_apex=p_apex, p_base=p_base, gap_ratio=ratio, ctrl_gap=ctrl[(g, b)])
                    row.update(type=special or typ, frac_sig=float(frac_sig), frac_apexfirst=float(np.mean(last == 0)),
                               z1_mean=z1.mean(), z1_sd=z1.std(), z2_mean=z2.mean(), z2_sd=z2.std(),
                               t1_mean=t1.mean(), t1_sd=t1.std(), t2_mean=t2.mean(), t2_sd=t2.std(),
                               gap_mean=gap.mean(), gap_sd=gap.std(), gap_median=float(np.median(gap)))
                    for k in ("x1", "x2", "y1", "y2", "d_apex1", "d_apex2", "d_base1", "d_base2",
                              "dist_to_plane1", "dist_to_plane2", "disp1", "disp2"):
                        v = np.array([ev["loc"][k] for _, ev in ok if k in ev["loc"]])
                        row[k] = float(v.mean()) if len(v) else np.nan
                        row[k + "_sd"] = float(v.std()) if len(v) else np.nan
                    mc = [mode_cross(r) for r, _ in ok]
                    for k in ("stdx", "stdy"):
                        zs = [m[k][1] for m in mc if m[k][1] is not None]
                        row[f"z_{k}_mean"] = float(np.mean(zs)) if len(zs) >= 6 else np.nan
                        row[f"frac_{k}"] = len(zs) / len(mc)
                else:
                    row.update(type=special or "unresolved")
                rows.append(row)
    return rows
