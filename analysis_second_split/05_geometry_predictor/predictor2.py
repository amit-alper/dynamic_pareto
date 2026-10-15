"""Two-stage geometry predictor: stage 1 = coherent path (as predictor.py) until the apex-direction spread reaches XP; then the population is
replaced by two clusters (apex fraction f, base 1-f) separated along y with total std XP; their centres follow the cluster-ansatz flow and the
x-spread of each cluster grows with its INTERNAL operator H_c (core.cluster_ansatz). z of the base-pair split = mean height when the pooled
x-std reaches XP."""
import os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
from core.stability import transverse_operator
from core import cluster_ansatz as ca
XP = 0.15


def predict2(A, S, gamma, beta, f, N=40, sigma=0.02, T=800.0, K=40000, xp=XP, z_floor=-0.02):
    A = np.asarray(A, float); x = np.asarray(S, float).copy(); dt = T / K
    V = np.zeros((3, 3)); zy = None
    for k in range(K):
        if x[2] > z_floor: return dict(y=zy, x=None)
        r = x[None] - A; d = np.maximum(np.linalg.norm(r, axis=1), 1e-9)
        if d.min() < 0.05: return dict(y=zy, x=None)
        L = transverse_operator(x, A, gamma, beta, N)
        if np.sqrt(V[1, 1]) >= xp: break
        V = V + dt * (L @ V + V @ L + sigma**2 * np.eye(3))
        x = x - dt * (beta * gamma / N) * ((d ** (gamma - 2))[:, None] * r).sum(0)
    else:
        return dict(y=None, x=None)
    zy = x[2]
    sx_coh = np.sqrt(V[0, 0])
    if sx_coh >= xp: return dict(y=zy, x=zy)
    # clusters: separation along y (the unstable apex direction), weights (f, 1-f)
    w = np.array([f, 1 - f]); Delta = xp / np.sqrt(f * (1 - f))
    X = np.stack([x + np.array([0, (1 - f) * Delta, 0]), x - np.array([0, f * Delta, 0])])
    vx = np.array([V[0, 0], V[0, 0]])
    for k in range(K):
        zb = (w * X[:, 2]).sum()
        if zb > z_floor: return dict(y=zy, x=None)
        Hs = [ca.internal_operator(X, w, A, gamma, beta)[c][0, 0] / N for c in range(2)]
        vx = vx + dt * (2 * np.array(Hs) * vx + sigma**2)
        v = ca.velocity(X, w, A, gamma, beta) / N
        X = X + dt * v
        if np.sqrt((w * vx).sum()) >= xp: return dict(y=zy, x=(w * X[:, 2]).sum())
    return dict(y=zy, x=None)
