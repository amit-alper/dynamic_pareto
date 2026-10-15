"""Geometry-only estimate of the apex fraction f: a test cell is advected in the field of the coherent bulk (S_i = p_i(bulk)); the watershed y_w
between 'goes to a3' and 'goes to the base pair' is found by bisection at the apex-split moment; f = P(Y > y_w), Y ~ N(y_mean, XP^2)."""
import numpy as np
from scipy.stats import norm
from predictor import XP
from core.stability import transverse_operator


def f_estimate(A, S, gamma, beta, N=40, sigma=0.02, T=800.0, K=40000, xp=XP):
    A = np.asarray(A, float); dt = T / K
    # bulk path + record until the apex-direction std reaches xp
    x = np.asarray(S, float).copy(); V = np.zeros((3, 3)); path = [x.copy()]
    for k in range(K):
        r = x[None] - A; d = np.maximum(np.linalg.norm(r, axis=1), 1e-9)
        L = transverse_operator(x, A, gamma, beta, N)
        if np.sqrt(V[1, 1]) >= xp or x[2] > -0.02: break
        V = V + dt * (L @ V + V @ L + sigma**2 * np.eye(3))
        x = x - dt * (beta * gamma / N) * ((d ** (gamma - 2))[:, None] * r).sum(0); path.append(x.copy())
    path = np.array(path)
    if np.sqrt(V[1, 1]) < xp: return None
    i1 = len(path) - 1; y1 = path[i1, 1]

    def goes_apex(y0):
        c = np.array([0.0, y0, path[i1, 2]]); kk = i1; bulk = path[i1].copy()
        for _ in range(200000):
            rb = bulk[None] - A; db = np.maximum(np.linalg.norm(rb, axis=1), 1e-9)
            p = np.exp(-beta * db**gamma)
            rc = c[None] - A; dc = np.maximum(np.linalg.norm(rc, axis=1), 1e-9)
            pc = np.exp(-beta * dc**gamma)
            v = -((beta * gamma) * dc ** (gamma - 2) * pc / p)[:, None] * rc            # sum_i grad p_i(c)/S_i, times (-1)... sign: ascent on p
            c = c - dt * (beta * gamma / N) * 0 + dt * (-v).sum(0) / N * 0 if False else c + dt * (-v.sum(0)) / N
            bulk = bulk - dt * (beta * gamma / N) * ((db ** (gamma - 2))[:, None] * rb).sum(0)
            if c[2] > -0.01 or dc.min() < 0.05: break
        return np.linalg.norm(c - A[2]) < np.linalg.norm(c - A[0])
    lo, hi = y1 - 3 * xp, y1 + 3 * xp
    if goes_apex(lo) or not goes_apex(hi): return None
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if goes_apex(mid) else (mid, hi)
    yw = 0.5 * (lo + hi)
    return dict(f=1 - norm.cdf((yw - y1) / xp), yw=yw, y1=y1)
