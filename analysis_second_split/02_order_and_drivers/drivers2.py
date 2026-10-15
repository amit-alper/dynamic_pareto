"""What sets the height/time of each decision: gamma or e/b?
Decision Y (apex): centroid of group C leaves the midpoint of the A,B centroids by DELTA.
Decision X (base pair): centroids of A and B separate by DELTA. Order-independent. Cached runs only."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import second_split as ss
from perpendicular_common import archetypes3, run_perp, classify
np.set_printoptions(precision=2, suppress=True, linewidth=160)
BETA = 1.6
GAMMAS = [1.0, 1.3, 1.6, 2.0, 2.5, 3.0]; EB = [0.6, 1.0, 1.73, 2.0, 2.5, 3.0, 4.0]
H_OF = {g: 1.35 for g in GAMMAS}; SG = {g: 0.02 for g in GAMMAS}
D = ss.DELTA; S = ss.STRIDE

def decisions(traj, A):
    fates = np.linalg.norm(traj[-1][:, None] - A[None], axis=-1).argmin(1)
    gr = [np.where(fates == k)[0] for k in range(3)]
    if min(len(g) for g in gr) < 2: return None
    tr = traj[::S]; cen = np.stack([tr[:, g, :].mean(1) for g in gr], 1); z = tr[:, :, 2].mean(1)
    dx = np.linalg.norm(cen[:, 0] - cen[:, 1], axis=1)
    dy = np.linalg.norm(cen[:, 2] - 0.5 * (cen[:, 0] + cen[:, 1]), axis=1)
    out = []
    for d in (dx, dy):
        if not (d > D).any(): return None
        i = np.argmax(d > D); out.append((i * S * ss.T / ss.K, z[i]))
    return out  # [(t_x,z_x),(t_y,z_y)]

res = {k: np.full((6, 7), np.nan) for k in ("zx", "zy", "tx", "ty", "n")}
for i, g in enumerate(GAMMAS):
    for j, e in enumerate(EB):
        A = archetypes3(e); r = []
        for s in range(8):
            d = decisions(run_perp(s, g, BETA, e, 1.0, H_OF[g], sigma=SG[g]), A)
            if d: r.append(d)
        res["n"][i, j] = len(r); res.setdefault('py', np.full((6, 7), np.nan))
        if r:
            r = np.array(r)  # (n,2,2)
            res['py'][i, j] = np.mean(r[:, 1, 0] < r[:, 0, 0])
            res["tx"][i, j], res["zx"][i, j] = np.median(r[:, 0, 0]), np.median(r[:, 0, 1])
            res["ty"][i, j], res["zy"][i, j] = np.median(r[:, 1, 0]), np.median(r[:, 1, 1])
print("rows gamma", GAMMAS, " cols e/b", EB)
for k, nm in [("zx", "z of base-pair decision X (a1|a2)"), ("zy", "z of apex decision Y (a3|base)"),
              ("tx", "t of X"), ("ty", "t of Y"), ("py", "P(apex decision Y before X)"), ("n", "usable seeds")]:
    print(f"\n{nm}\n{res[k]}")
# crude driver attribution on gamma<=1.6 (clean rows): fit z ~ a*gamma + b*log(e/b), standardized
rows = slice(0, 6)
G, E = np.meshgrid(GAMMAS, EB, indexing="ij")
for k in ("zx", "zy", "tx", "ty"):
    y = res[k][rows].ravel(); m = np.isfinite(y)
    Xm = np.c_[G.ravel(), np.log(E.ravel())][m]; Xs = (Xm - Xm.mean(0)) / Xm.std(0)
    ys = (y[m] - y[m].mean()) / y[m].std(); c, *_ = np.linalg.lstsq(np.c_[Xs, np.ones(m.sum())], ys, rcond=None)
    r2 = 1 - ((ys - np.c_[Xs, np.ones(m.sum())] @ c) ** 2).sum() / (ys ** 2).sum()
    print(f"{k}: standardized coef  gamma={c[0]:+.2f}  log(e/b)={c[1]:+.2f}   R2={r2:.2f}")
# spread attribution: variation along gamma (mean over e/b) vs along e/b (mean over gamma)
for k in ("zx", "zy", "tx", "ty"):
    M = res[k][rows]
    print(f"{k}: range across gamma (avg over e/b) = {np.nanmax(np.nanmean(M,1))-np.nanmin(np.nanmean(M,1)):.2f}; "
          f"range across e/b (avg over gamma) = {np.nanmax(np.nanmean(M,0))-np.nanmin(np.nanmean(M,0)):.2f}")
