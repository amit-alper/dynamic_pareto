"""
Reduced cluster-ansatz (mean-field) model of the collective Pareto flow.

Inspired by the mean-field-trajectory analysis of Gorbonos, Gov & Couzin
(PRX Life 2, 013008): follow a low-dimensional solution branch and call a
bifurcation the point where a branch loses stability.

Here the N -> infinity limit of the N-cell collective flow is a weighted
particle system. C clusters sit at positions x_c in R^D with weights n_c
(fraction of cells, sum n_c = 1). With normalised task totals

    S_i = sum_c n_c p_i(x_c),     p_i(x) = exp(-beta ||x - a_i||^gamma),

each cell in cluster c follows (time rescaled by N, i.e. N = 1 units)

    dx_c/dt = sum_i grad p_i(x_c) / S_i .

Stability facts used below (all exact for this system):
  * Jacobian of the cluster system with fixed weights:
        dv_c/dx_d = delta_cd * sum_i Hess p_i(x_c)/S_i
                    - sum_i g_i(x_c) (n_d g_i(x_d))^T / S_i^2 .
  * A zero-sum perturbation INSIDE one cluster kills the second term, so the
    cluster's internal (splitting) operator is
        H_c = sum_i Hess p_i(x_c) / S_i .
    The cluster is internally stable iff H_c <= 0. With one cluster (n=1,
    S_i = p_i) this is exactly core/stability.transverse_operator at N=1.
  * The splitting rule only needs the vertex partition of a cluster, see
    ``split_cluster``.

Nothing here modifies existing modules.
"""

import numpy as np
from scipy.integrate import solve_ivp

EPS = 1e-12


# ----------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------
def simplex_vertices(k, R=1.0):
    """k vertices of a regular simplex of circumradius R, centred at the
    origin, in R^k: the simplex lives in the first k-1 coordinates and the
    last coordinate is the perpendicular source axis. Returns (k, k)."""
    E = np.eye(k) - 1.0 / k
    _, _, vt = np.linalg.svd(E)
    B = vt[: k - 1].T                          # (k, k-1) basis of sum-zero space
    V = E @ B
    V *= R / np.linalg.norm(V, axis=1, keepdims=True)
    return np.hstack([V, np.zeros((k, 1))])


# ----------------------------------------------------------------------
# kernel pieces
# ----------------------------------------------------------------------
def _kernel(x, A, gamma, beta):
    """p (k,), g (k,D) = grad p_i, Hs (k,D,D) = Hess p_i at a point x."""
    r = x[None, :] - A
    d = np.maximum(np.linalg.norm(r, axis=1), EPS)
    p = np.exp(-beta * d ** gamma)
    c = beta * gamma * d ** (gamma - 2.0)
    g = -(c * p)[:, None] * r
    rh = r / d[:, None]
    scale = beta * gamma * d ** gamma - gamma + 2.0
    D = x.shape[0]
    Hs = (c * p)[:, None, None] * (
        scale[:, None, None] * rh[:, :, None] * rh[:, None, :] - np.eye(D)[None]
    )
    return p, g, Hs


def task_totals(X, n, A, gamma, beta):
    return sum(n[c] * _kernel(X[c], A, gamma, beta)[0] for c in range(len(n)))


def log_fitness(X, n, A, gamma, beta):
    """sum_i log S_i (per-N collective log-fitness of the cluster state)."""
    return float(np.sum(np.log(task_totals(X, n, A, gamma, beta))))


def velocity(X, n, A, gamma, beta):
    S = task_totals(X, n, A, gamma, beta)
    V = np.zeros_like(X)
    for c in range(len(n)):
        _, g, _ = _kernel(X[c], A, gamma, beta)
        V[c] = (g / S[:, None]).sum(axis=0)
    return V


def internal_operator(X, n, A, gamma, beta):
    """H_c for every cluster: (C, D, D). Positive eigenvalue = cluster splits."""
    S = task_totals(X, n, A, gamma, beta)
    out = []
    for c in range(len(n)):
        _, _, Hs = _kernel(X[c], A, gamma, beta)
        out.append((Hs / S[:, None, None]).sum(axis=0))
    return np.array(out)


def internal_eigs(X, n, A, gamma, beta, exclude_axis=True):
    """Largest eigenvalue of H_c per cluster, (C,).

    exclude_axis=True restricts to the lateral (first D-1) coordinates: the
    source axis is the last coordinate, and its internal mode is dispersion
    of arrival times along the path (positive far from the targets), not a
    specialization split."""
    H = internal_operator(X, n, A, gamma, beta)
    if exclude_axis:
        H = H[:, :-1, :-1]
    return np.array([np.linalg.eigvalsh(0.5 * (h + h.T))[-1] for h in H])


def jacobian(X, n, A, gamma, beta):
    """Full (C*D, C*D) Jacobian of the cluster flow, weights held fixed."""
    C, D = X.shape
    S = task_totals(X, n, A, gamma, beta)
    G, Hl = [], []
    for c in range(C):
        _, g, Hs = _kernel(X[c], A, gamma, beta)
        G.append(g)
        Hl.append((Hs / S[:, None, None]).sum(axis=0))
    J = np.zeros((C * D, C * D))
    for c in range(C):
        for d in range(C):
            blk = -n[d] * np.einsum("ia,ib,i->ab", G[c], G[d], 1.0 / S ** 2)
            if c == d:
                blk = blk + Hl[c]
            J[c * D:(c + 1) * D, d * D:(d + 1) * D] = blk
    return J


# ----------------------------------------------------------------------
# splitting policies (vertex partitions of a cluster)
# ----------------------------------------------------------------------
def cluster_vertex_set(x, A, tol=1e-3):
    """Indices of the vertices a cluster 'covers': those within (1+tol) of the
    nearest one."""
    d = np.linalg.norm(x[None, :] - A, axis=1)
    return np.where(d <= d.min() * (1.0 + tol) + 1e-9)[0]


def split_cluster(x, w, A, policy, eps):
    """Replace one cluster (position x, weight w) by children.

    policy:
      'peel' - one vertex vs the rest        (binary, f = 1/a)
      'half' - vertex set cut in two halves  (binary, f = ceil(a/2)/a)
      'full' - one child per covered vertex  (a-way, weights w/a)
    Children are displaced eps toward their group's barycentre, keeping the
    weighted centre of mass fixed. Returns (positions (m,D), weights (m,)).
    """
    idx = cluster_vertex_set(x, A)
    a = len(idx)
    if a < 2:
        return x[None, :].copy(), np.array([w])
    if policy == "peel":
        groups = [idx[:1], idx[1:]]
    elif policy == "half":
        h = (a + 1) // 2
        groups = [idx[:h], idx[h:]]
    elif policy == "full":
        groups = [idx[i:i + 1] for i in range(a)]
    else:
        raise ValueError(policy)
    ws = np.array([w * len(g) / a for g in groups])
    dirs = []
    for g in groups:
        u = A[g].mean(axis=0) - x
        nu = np.linalg.norm(u)
        dirs.append(u / nu if nu > EPS else u)
    dirs = np.array(dirs)
    dirs -= (ws[:, None] * dirs).sum(axis=0) / ws.sum()    # zero-sum
    return x[None, :] + eps * dirs, ws


# ----------------------------------------------------------------------
# integration with event-driven splitting
# ----------------------------------------------------------------------
def _flow(X0, n, A, gamma, beta, dt):
    C, D = X0.shape

    def rhs(_t, y):
        return velocity(y.reshape(C, D), n, A, gamma, beta).ravel()

    sol = solve_ivp(rhs, (0.0, dt), X0.ravel(), method="LSODA", rtol=1e-8, atol=1e-10)
    return sol.y[:, -1].reshape(C, D)


def run_cascade(A, gamma, beta, h, policy="peel", T=60.0, dt=0.05,
                eps=1e-3, x_p=0.15, init=None, max_clusters=64, split=True,
                step_len=0.01):
    """Deterministic cluster flow from the axis point (0,...,0,-h) (or from a
    given ``init=(X, n)``) with event-driven splitting.

    A cluster splits when its internal growth rate has been positive long
    enough, Gamma_c = int max(lambda_c,0) dt >= ln(x_p/eps), the same
    accumulated-growth criterion as the Gamma estimator of
    bifurcation_time_and_specialization.tex. Children start 'immature' and
    only start accumulating once they are x_p apart, so a split is not
    re-triggered by the very instability that just produced it.

    Returns dict(t, nclust, zmean, events, X, n, logF, lam_hist).
    """
    D = A.shape[1]
    if init is None:
        X = np.zeros((1, D))
        X[0, -1] = -h
        n = np.array([1.0])
    else:
        X, n = (np.array(init[0], float), np.array(init[1], float))
    gam_thr = np.log(x_p / eps)
    Gam = np.zeros(len(n))
    mature = np.ones(len(n), bool)
    parent_of = -np.ones(len(n), int)           # sibling used for maturity
    events, ts, ncl, zm = [], [], [], []
    t = 0.0
    lam_prev = internal_eigs(X, n, A, gamma, beta)
    while t < T:
        # adaptive check interval: resolve Gamma along fast stretches of path
        vmax = np.linalg.norm(velocity(X, n, A, gamma, beta), axis=1).max()
        step = float(min(dt, max(step_len / max(vmax, 1e-9), 1e-6)))
        X = _flow(X, n, A, gamma, beta, step)
        t += step
        lam = internal_eigs(X, n, A, gamma, beta)
        lam_avg = 0.5 * (lam + lam_prev) if len(lam) == len(lam_prev) else lam
        # maturity: separation from nearest other cluster
        for c in range(len(n)):
            if not mature[c] and len(n) > 1:
                dmin = np.min([np.linalg.norm(X[c] - X[o]) for o in range(len(n)) if o != c])
                if dmin >= x_p:
                    mature[c], Gam[c] = True, 0.0
        for c in range(len(n)):
            if mature[c]:
                Gam[c] = max(0.0, Gam[c] + lam_avg[c] * step)
        lam_prev = lam
        ts.append(t)
        ncl.append(len(n))
        zm.append(float((n * X[:, -1]).sum()))
        if not split:
            continue
        for c in range(len(n)):
            if mature[c] and Gam[c] >= gam_thr and len(n) < max_clusters:
                pos, ws = split_cluster(X[c], n[c], A, policy, eps)
                if len(ws) < 2:
                    Gam[c] = 0.0
                    continue
                events.append(dict(t=t, z=float(X[c, -1]), n=float(n[c]),
                                   lam=float(lam[c]), children=len(ws),
                                   fracs=(ws / n[c]).tolist()))
                keep = [i for i in range(len(n)) if i != c]
                X = np.vstack([X[keep], pos])
                n = np.concatenate([n[keep], ws])
                Gam = np.concatenate([Gam[keep], np.zeros(len(ws))])
                mature = np.concatenate([mature[keep], np.zeros(len(ws), bool)])
                lam_prev = internal_eigs(X, n, A, gamma, beta)
                break                                   # one split per chunk
    return dict(t=np.array(ts), nclust=np.array(ncl), zmean=np.array(zm),
                events=events, X=X, n=n, logF=log_fitness(X, n, A, gamma, beta),
                lam=internal_eigs(X, n, A, gamma, beta))


def classify_final(X, n, A, tol=0.2):
    """Group clusters by nearest vertex; report mass per vertex and the mass
    sitting away from every vertex (generalists)."""
    d = np.linalg.norm(X[:, None, :] - A[None], axis=2)
    near = d.argmin(axis=1)
    mass = np.zeros(len(A))
    generalist = 0.0
    for c in range(len(n)):
        if d[c, near[c]] <= tol:
            mass[near[c]] += n[c]
        else:
            generalist += n[c]
    return mass, generalist


# ----------------------------------------------------------------------
# policy-free mean-field cloud (many equal-weight particles, no split rule)
# ----------------------------------------------------------------------
def velocity_fast(X, n, A, gamma, beta, soft=0.0):
    """Vectorised ``velocity`` (same formula) for large particle counts.
    ``soft`` > 0 replaces d by sqrt(d^2 + soft^2) inside the power law only
    (analogue of the r_stop radius of greedy_trajectory): it removes the
    cusp stiffness at the vertices for gamma < 2 and is otherwise a tiny
    perturbation when d >> soft."""
    r = X[:, None, :] - A[None, :, :]                      # (C,k,D)
    d = np.maximum(np.sqrt((r ** 2).sum(axis=2) + soft ** 2), EPS)
    p = np.exp(-beta * d ** gamma)                          # (C,k)
    S = (n[:, None] * p).sum(axis=0)                        # (k,)
    coef = -beta * gamma * d ** (gamma - 2.0) * p / S[None, :]
    return (coef[:, :, None] * r).sum(axis=1)


def run_cloud(A, gamma, beta, h, M=120, eps=1e-3, T=8.0, n_out=400, seed=0,
              soft=0.02, lock_axis=False):
    """Isotropic Gaussian seed (lateral) around the axis point, then pure
    mean-field flow. Returns times and positions (n_out, M, D).
    lock_axis=True removes the axial-dispersion mode by forcing every cell to
    the population-mean axial velocity, isolating the lateral (split) physics
    that the linear theory describes."""
    rng = np.random.default_rng(seed)
    D = A.shape[1]
    X0 = np.zeros((M, D))
    X0[:, :-1] = eps * rng.normal(size=(M, D - 1))
    X0[:, -1] = -h
    n = np.full(M, 1.0 / M)

    def rhs(_t, y):
        v = velocity_fast(y.reshape(M, D), n, A, gamma, beta, soft)
        if lock_axis:                  # all cells share one height (coherent path)
            v[:, -1] = (n * v[:, -1]).sum()
        return v.ravel()

    ts = np.geomspace(1e-4, T, n_out)
    sol = solve_ivp(rhs, (0.0, T), X0.ravel(), method="RK45", t_eval=ts,
                    rtol=1e-6, atol=1e-9)
    return sol.t, sol.y.T.reshape(len(sol.t), M, D)
