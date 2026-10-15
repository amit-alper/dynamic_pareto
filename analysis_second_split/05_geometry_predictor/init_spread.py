"""Effect of the initial spread of cells at the source on where the population splits.
Same dynamics as greedy.greedy_trajectory (tissue gradient, constant noise) but cells start at source + s0 * N(0, I) in the x,y plane.
Per run we keep only the mean height z, the std over cells of x and of y, and the fates."""
import os, sys, pickle, hashlib, json
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from core.landscape import landscape_gradient
N, T, K, STRIDE = 40, 800.0, 40000, 10
GAMMA, BETA, H, SIGMA, XP = 1.3, 1.6, 1.35, 0.02, 0.15
CACHE = os.path.join(HERE, ".cache_init"); os.makedirs(CACHE, exist_ok=True)
S0 = [0.0, 0.01, 0.02, 0.04, 0.07, 0.10]
E = [1.0, 2.0, 3.0]
NS = 12


def geom(e):
    A = np.array([[-1, 0, 0], [1, 0, 0], [0, e, 0]], float); return A, np.array([0.0, e / 3, -H])


def run(args):
    e, s0, seed = args
    key = hashlib.sha1(json.dumps([e, s0, seed, GAMMA, BETA, H, SIGMA, N, T, K]).encode()).hexdigest()[:12]
    p = os.path.join(CACHE, key + ".pkl")
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    A, S = geom(e); rng = np.random.default_rng(seed); dt = T / K
    X = np.tile(S, (N, 1)); X[:, :2] += s0 * rng.standard_normal((N, 2))
    w = np.ones(3); z, sx, sy = [], [], []
    for k in range(K):
        if k % STRIDE == 0:
            z.append(X[:, 2].mean()); sx.append(X[:, 0].std()); sy.append(X[:, 1].std())
        g = landscape_gradient(X, A, GAMMA, w, None, "exp", 10.0, True, BETA)
        X = X + dt * g + SIGMA * np.sqrt(dt) * rng.standard_normal((N, 3))
    fates = np.linalg.norm(X[:, None] - A[None], axis=-1).argmin(1)
    out = dict(z=np.array(z), sx=np.array(sx), sy=np.array(sy), fates=fates)
    pickle.dump(out, open(p, "wb")); return out


def first_cross(z, s, thr):
    return z[np.argmax(s >= thr)] if (s >= thr).any() else np.nan


if __name__ == "__main__":
    tasks = [(e, s0, sd) for e in E for s0 in S0 for sd in range(NS)]
    with Pool(10) as pool: res = pool.map(run, tasks)
    out = {}
    for (e, s0, sd), r in zip(tasks, res):
        out.setdefault((e, s0), []).append(dict(zx=first_cross(r["z"], r["sx"], XP), zy=first_cross(r["z"], r["sy"], XP), shares=np.bincount(r["fates"], minlength=3) / N,
                                                 sx0=r["sx"][0], sy0=r["sy"][0]))
    pickle.dump(out, open(os.path.join(HERE, "init_spread.pkl"), "wb"))
    for e in E:
        for s0 in S0:
            rr = out[(e, s0)]
            zx = np.array([r["zx"] for r in rr]); zy = np.array([r["zy"] for r in rr])
            print(f"e={e} s0={s0:.2f}  z_base {np.nanmean(zx):+.3f}±{np.nanstd(zx):.3f}  z_apex {np.nanmean(zy):+.3f}±{np.nanstd(zy):.3f}  (found {np.isfinite(zx).mean():.2f}/{np.isfinite(zy).mean():.2f})")
