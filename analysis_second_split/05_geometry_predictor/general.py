"""General planar 3-target geometry + arbitrary in-plane source position.
Geometry-free split observables (no axes assumed): in the xy-plane covariance C of the cell cloud,
   z1 = height where sqrt(lambda_max(C)) first reaches XP   (first split),  axis1 = its principal direction
   z2 = height where sqrt(lambda_min(C)) first reaches XP   (second split, the other in-plane direction)."""
import os, sys, io, contextlib, pickle, hashlib, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from core.stability import transverse_operator
from greedy.greedy import greedy_trajectory
XP, N, T, K, STRIDE = 0.15, 40, 800.0, 40000, 10
CACHE = os.path.join(HERE, ".cache_general"); os.makedirs(CACHE, exist_ok=True)


def eig2(sxx, syy, sxy):
    m = 0.5 * (sxx + syy); r = np.sqrt((0.5 * (sxx - syy)) ** 2 + sxy ** 2)
    return np.sqrt(np.maximum(m + r, 0)), np.sqrt(np.maximum(m - r, 0)), 0.5 * np.arctan2(2 * sxy, sxx - syy)


def pred(A, S, gamma, beta, sigma, Kp=20000, floor=-0.02):
    A = np.asarray(A, float); x = np.asarray(S, float).copy(); dt = T / Kp; V = np.zeros((3, 3)); o = [np.nan, np.nan, np.nan]
    for k in range(Kp):
        if x[2] > floor: break
        r = x[None] - A; d = np.maximum(np.linalg.norm(r, axis=1), 1e-9)
        if d.min() < 0.05: break
        L = transverse_operator(x, A, gamma, beta, N); V = V + dt * (L @ V + V @ L + sigma ** 2 * np.eye(3))
        x = x - dt * (beta * gamma / N) * ((d ** (gamma - 2))[:, None] * r).sum(0)
        a, b, th = eig2(V[0, 0], V[1, 1], V[0, 1])
        if np.isnan(o[0]) and a >= XP: o[0] = x[2]; o[2] = th
        if np.isnan(o[1]) and b >= XP: o[1] = x[2]
    return o


def sim_one(args):
    A, S, gamma, beta, sigma, seed = args
    A = np.asarray(A, float); S = np.asarray(S, float)
    key = hashlib.sha1(json.dumps([A.tolist(), S.tolist(), gamma, beta, sigma, seed, N, T, K, "v2"]).encode()).hexdigest()[:12]
    p = os.path.join(CACHE, key + ".pkl")
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    with contextlib.redirect_stdout(io.StringIO()):
        res = greedy_trajectory(S, A, gamma, N, K, T, mode="exp", log=True, sigma=sigma, seed=seed, beta=beta)
    tr = res["trajectory"].astype(np.float64); ts = tr[::STRIDE]; X, Y = ts[:, :, 0], ts[:, :, 1]
    sxx, syy = X.var(1), Y.var(1); sxy = ((X - X.mean(1, keepdims=True)) * (Y - Y.mean(1, keepdims=True))).mean(1)
    a, b, th = eig2(sxx, syy, sxy); z = ts[:, :, 2].mean(1)
    fates = np.linalg.norm(tr[-1][:, None] - A[None], axis=-1).argmin(1)
    def fc(s):
        return z[np.argmax(s >= XP)] if (s >= XP).any() else np.nan
    i = np.argmax(a >= XP) if (a >= XP).any() else None
    out = dict(z1=fc(a), z2=fc(b), th=th[i] if i is not None else np.nan, shares=np.bincount(fates, minlength=3) / len(fates))
    pickle.dump(out, open(p, "wb")); return out


def circ_mean_axis(th):
    th = th[~np.isnan(th)]
    return 0.5 * np.arctan2(np.sin(2 * th).mean(), np.cos(2 * th).mean()) if len(th) else np.nan
