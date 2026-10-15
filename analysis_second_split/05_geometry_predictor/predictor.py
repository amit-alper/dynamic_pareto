"""Geometry-only predictor of where (in height z) the base-pair split (x) and the apex split (y) happen.
Input: archetypes A, source S, gamma, beta, N, sigma.  No simulation of the cell population is used: the coherent
(co-located) path is integrated, and the lab-frame covariance V of the transverse fluctuation obeys
    dV/dt = Lam V + V Lam + sigma^2 I,   Lam = core.stability.transverse_operator(x(t))
Predicted split in direction u = first time sqrt(u^T V u) >= XP; reported as the mean height z at that time."""
import os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
from core.stability import transverse_operator

XP = 0.15


def predict(A, S, gamma, beta, N=40, sigma=0.02, T=800.0, K=20000, xp=XP, z_floor=-0.02):
    A = np.asarray(A, float); x = np.asarray(S, float).copy(); dt = T / K
    V = np.zeros((3, 3)); out = {"x": None, "y": None}
    for k in range(K):
        if x[2] > z_floor: break
        r = x[None] - A; d = np.maximum(np.linalg.norm(r, axis=1), 1e-9)
        if d.min() < 0.05: break
        L = transverse_operator(x, A, gamma, beta, N)
        V = V + dt * (L @ V + V @ L + sigma**2 * np.eye(3))
        x = x - dt * (beta * gamma / N) * ((d ** (gamma - 2))[:, None] * r).sum(0)
        for key, i in (("x", 0), ("y", 1)):
            if out[key] is None and np.sqrt(V[i, i]) >= xp: out[key] = x[2]
    return out


def static_score(A, S, gamma, beta, N=40):
    """one number from the geometry at the source: lam_y - lam_x of Lambda_split at S (>0: apex mode stronger first)"""
    L = transverse_operator(np.asarray(S, float), np.asarray(A, float), gamma, beta, N)
    return L[1, 1] - L[0, 0]
