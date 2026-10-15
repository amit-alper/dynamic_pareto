"""Theory helpers for the non-equilateral perpendicular three-target geometry (b=1).
a1=(-1,0,0) a2=(1,0,0) a3=(0,e,0); source (0,e/3,-h) on the centroid line.  Read-only use of core/."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from core.stability import transverse_operator

def arche(e):
    return np.array([[-1.0, 0, 0], [1.0, 0, 0], [0, e, 0]])

def source(e, h=2.5):
    return np.array([0.0, e / 3.0, -h])

# ---------- closed forms on the line x=0 at (0,Y,z) -------------------------------------------
def pieces(Y, z, e, gamma, beta, N=1.0):
    de2 = 1.0 + Y**2 + z**2
    d32 = (e - Y)**2 + z**2
    de, d3 = np.sqrt(de2), np.sqrt(d32)
    ce = beta * gamma / N * de**(gamma - 2); c3 = beta * gamma / N * d3**(gamma - 2)
    se = beta * gamma * de**gamma - gamma + 2; s3 = beta * gamma * d3**gamma - gamma + 2
    return de2, d32, ce, c3, se, s3

def lam_xy(Y, z, e, gamma, beta, N=1.0):
    """(lambda_xx, lambda_yy, lambda_yz) of Lambda_split at (0,Y,z): exact closed forms."""
    de2, d32, ce, c3, se, s3 = pieces(Y, z, e, gamma, beta, N)
    tau = 2 * ce + c3
    lx = 2 * ce * se / de2 - tau
    ly = 2 * ce * se * Y**2 / de2 + c3 * s3 * (e - Y)**2 / d32 - tau
    lyz = 2 * ce * se * Y * z / de2 + c3 * s3 * (Y - e) * z / d32
    return lx, ly, lyz

def lam_diff(Y, z, e, gamma, beta, N=1.0):
    """lambda_yy - lambda_xx = 2 c_e s_e (Y^2-1)/d_e^2 + c_3 s_3 (e-Y)^2/d_3^2   (exact)."""
    de2, d32, ce, c3, se, s3 = pieces(Y, z, e, gamma, beta, N)
    return 2 * ce * se * (Y**2 - 1) / de2 + c3 * s3 * (e - Y)**2 / d32

# ---------- coherent path (x stays 0) ----------------------------------------------------------
def coherent_path(e, gamma, beta, N=40.0, h=2.5, T=800.0, K=40000, Y0=None):
    A = arche(e); x = source(e, h)
    if Y0 is not None: x[1] = Y0
    dt = T / K
    P = np.empty((K + 1, 3)); P[0] = x
    for k in range(K):
        r = x[None, :] - A
        d = np.maximum(np.linalg.norm(r, axis=1), 1e-12)
        v = -(beta * gamma / N) * (d**(gamma - 2))[:, None] * r
        x = x + dt * v.sum(0); P[k + 1] = x
    return np.arange(K + 1) * dt, P

def mode_lambdas(P, e, gamma, beta, N=40.0):
    """lambda_x, lambda_y (e_y direction) and lambda_perp (direction perpendicular to the velocity in the y-z plane)
    along a path of points (x=0)."""
    Y, z = P[:, 1], P[:, 2]
    lx, ly, lyz = lam_xy(Y, z, e, gamma, beta, N)
    # lambda_zz needed for perp-direction value
    de2, d32, ce, c3, se, s3 = pieces(Y, z, e, gamma, beta, N)
    tau = 2 * ce + c3
    lz = 2 * ce * se * z**2 / de2 + c3 * s3 * z**2 / d32 - tau
    # velocity (y,z)
    A = arche(e)
    r = P[:, None, :] - A[None]
    d = np.maximum(np.linalg.norm(r, axis=2), 1e-12)
    v = -(beta * gamma / N) * ((d**(gamma - 2))[..., None] * r).sum(1)
    vy, vz = v[:, 1], v[:, 2]; nv = np.maximum(np.hypot(vy, vz), 1e-300)
    ny, nz = vz / nv, -vy / nv                       # unit vector perpendicular to v in the y-z plane
    lperp = ly * ny**2 + lz * nz**2 + 2 * lyz * ny * nz
    return dict(lx=lx, ly=ly, lz=lz, lyz=lyz, lperp=lperp, v=v)

def variance_cross(times, lam, sigma, thr, V0=0.0):
    """exact integration of dV/dt=2 lam V + sigma^2 per step; first index with sqrt(V)>=thr (None if never)."""
    dt = times[1] - times[0]
    l = lam[:-1]
    a = np.exp(2 * l * dt)
    small = np.abs(l) < 1e-12
    b = np.where(small, sigma**2 * dt, sigma**2 * (a - 1) / np.where(small, 1, 2 * l))
    V = np.empty(len(times)); V[0] = V0
    for k in range(len(l)):
        V[k + 1] = V[k] * a[k] + b[k]
    idx = np.argmax(np.sqrt(V) >= thr) if (np.sqrt(V) >= thr).any() else None
    return V, idx

def first_lasting_positive(lam):
    pos = lam > 0
    # last index where it is non-positive
    neg = np.where(~pos)[0]
    if len(neg) == 0: return 0
    i = neg[-1] + 1
    return i if i < len(lam) else None
