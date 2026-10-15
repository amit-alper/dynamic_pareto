"""Symmetric two-stage predictor. Stage 1: coherent path; the first direction (x = base pair, y = apex) whose spread reaches XP is the FIRST split.
Stage 2: two weighted clusters separated along that direction (apex-first: weights (f,1-f) along y; base-first: weights (1/2,1/2) along x, the
side clusters of a1 and a2). Cluster centres follow core.cluster_ansatz; the spread in the OTHER direction grows with each cluster's internal
operator H_c, and the second split is where the pooled spread reaches XP.   Returns first/second split heights and which was first."""
import os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
from core.stability import transverse_operator
from core import cluster_ansatz as ca
XP = 0.15


def predict3(A, S, gamma, beta, f=0.25, N=40, sigma=0.02, T=800.0, K=40000, xp=XP, z_floor=-0.02):
    A = np.asarray(A, float); x = np.asarray(S, float).copy(); dt = T / K
    V = np.zeros((3, 3)); first = None
    for k in range(K):
        if x[2] > z_floor: return dict(first=None, z1=None, z2=None)
        r = x[None] - A; d = np.maximum(np.linalg.norm(r, axis=1), 1e-9)
        if d.min() < 0.05: return dict(first=None, z1=None, z2=None)
        sx, sy = np.sqrt(max(V[0, 0], 0)), np.sqrt(max(V[1, 1], 0))
        if sx >= xp or sy >= xp:
            first = "y" if sy >= sx else "x"; break
        L = transverse_operator(x, A, gamma, beta, N)
        V = V + dt * (L @ V + V @ L + sigma**2 * np.eye(3))
        x = x - dt * (beta * gamma / N) * ((d ** (gamma - 2))[:, None] * r).sum(0)
    if first is None: return dict(first=None, z1=None, z2=None)
    z1 = x[2]
    if first == "y": ax_split, ax_other, w = 1, 0, np.array([f, 1 - f])
    else:            ax_split, ax_other, w = 0, 1, np.array([0.5, 0.5])
    Delta = xp / np.sqrt(w[0] * w[1])
    e_s = np.zeros(3); e_s[ax_split] = 1.0
    X = np.stack([x + w[1] * Delta * e_s, x - w[0] * Delta * e_s])
    vo = np.array([V[ax_other, ax_other]] * 2)
    for k in range(K):
        zb = (w * X[:, 2]).sum()
        if zb > z_floor: return dict(first=first, z1=z1, z2=None)
        H = ca.internal_operator(X, w, A, gamma, beta)
        h = np.array([H[c][ax_other, ax_other] for c in range(2)]) / N
        vo = vo + dt * (2 * h * vo + sigma**2)
        if not np.all(np.isfinite(vo)): return dict(first=first, z1=z1, z2=None)
        X = X + dt * ca.velocity(X, w, A, gamma, beta) / N
        if np.sqrt((w * vo).sum()) >= xp: return dict(first=first, z1=z1, z2=zb)
    return dict(first=first, z1=z1, z2=None)
