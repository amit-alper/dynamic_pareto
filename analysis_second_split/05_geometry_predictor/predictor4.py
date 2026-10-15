"""Direction-free two-stage predictor for an arbitrary planar 3-target geometry and in-plane source.
Stage 1: coherent path, in-plane covariance V; the first split is where the largest in-plane std reaches XP, along its principal axis u1.
The target peeled by that split is the one farthest along u1 from the mean of the others; its cell fraction f is an input (as in predictor3).
Stage 2: two clusters (f, 1-f) separated along u1; centres follow core.cluster_ansatz; the spread along u2 (perpendicular in-plane) grows with
u2^T H_c u2 of each cluster; second split where the pooled u2-std reaches XP.  Returns z1, z2, angle of u1, index of the peeled target."""
import os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
from core.stability import transverse_operator
from core import cluster_ansatz as ca
XP = 0.15


def peeled_target(A, u):
    p = A[:, :2] @ u
    return int(np.argmax([abs(p[i] - np.delete(p, i).mean()) for i in range(3)]))


def predict4(A, S, gamma, beta, f_of_target, N=40, sigma=0.02, T=800.0, K=40000, xp=XP, z_floor=-0.02):
    """f_of_target: array (3,) of the fraction of cells ending at each target (input)"""
    A = np.asarray(A, float); x = np.asarray(S, float).copy(); dt = T / K; V = np.zeros((3, 3))
    for k in range(K):
        if x[2] > z_floor: return dict(z1=None, z2=None, ang=np.nan, peel=None)
        r = x[None] - A; d = np.maximum(np.linalg.norm(r, axis=1), 1e-9)
        if d.min() < 0.05: return dict(z1=None, z2=None, ang=np.nan, peel=None)
        w_, v_ = np.linalg.eigh(V[:2, :2])
        if np.sqrt(max(w_[1], 0)) >= xp: break
        L = transverse_operator(x, A, gamma, beta, N)
        V = V + dt * (L @ V + V @ L + sigma**2 * np.eye(3))
        x = x - dt * (beta * gamma / N) * ((d ** (gamma - 2))[:, None] * r).sum(0)
    else:
        return dict(z1=None, z2=None, ang=np.nan, peel=None)
    z1 = x[2]; u1 = v_[:, 1]; u2 = np.array([-u1[1], u1[0]])
    ang = np.degrees(np.arctan2(u1[1], u1[0])); ang = (ang + 90) % 180 - 90
    pt = peeled_target(A, u1)
    f = float(np.clip(f_of_target[pt], 0.05, 0.6))
    w = np.array([f, 1 - f]); Delta = xp / np.sqrt(f * (1 - f))
    e1 = np.array([u1[0], u1[1], 0.0]); e2 = np.array([u2[0], u2[1], 0.0])
    X = np.stack([x + w[1] * Delta * e1, x - w[0] * Delta * e1])
    vo = np.array([e2 @ V @ e2] * 2)
    for k in range(K):
        zb = (w * X[:, 2]).sum()
        if zb > z_floor: return dict(z1=z1, z2=None, ang=ang, peel=pt)
        H = ca.internal_operator(X, w, A, gamma, beta)
        h = np.array([e2 @ H[c] @ e2 for c in range(2)]) / N
        vo = vo + dt * (2 * h * vo + sigma**2)
        if not np.all(np.isfinite(vo)): return dict(z1=z1, z2=None, ang=ang, peel=pt)
        X = X + dt * ca.velocity(X, w, A, gamma, beta) / N
        if np.sqrt((w * vo).sum()) >= xp: return dict(z1=z1, z2=zb, ang=ang, peel=pt)
    return dict(z1=z1, z2=None, ang=ang, peel=pt)
