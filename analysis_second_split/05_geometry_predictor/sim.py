"""Population simulation summaries (std of x and of y over cells, mean z, fates) for arbitrary geometry; cached, parallel."""
import os, sys, io, contextlib, pickle, hashlib, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from greedy.greedy import greedy_trajectory
CACHE = os.path.join(HERE, ".cache"); os.makedirs(CACHE, exist_ok=True)
N, T, K, STRIDE = 40, 800.0, 40000, 10


def run_summary(args):
    A, S, gamma, beta, sigma, seed = args
    A = np.asarray(A, float); S = np.asarray(S, float)
    key = hashlib.sha1(json.dumps([A.tolist(), S.tolist(), gamma, beta, sigma, seed, N, T, K]).encode()).hexdigest()[:12]
    path = os.path.join(CACHE, key + ".pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    with contextlib.redirect_stdout(io.StringIO()):
        res = greedy_trajectory(S, A, gamma, N, K, T, mode="exp", log=True, sigma=sigma, seed=seed, beta=beta, r_stop=0.05 if gamma < 1 else None)
    tr = res["trajectory"].astype(np.float32); ts = tr[::STRIDE]
    fates = np.linalg.norm(tr[-1][:, None] - A[None], axis=-1).argmin(1)
    out = dict(z=ts[:, :, 2].mean(1), sx=ts[:, :, 0].std(1), sy=ts[:, :, 1].std(1), fates=fates)
    pickle.dump(out, open(path, "wb")); return out


def first_cross(z, s, thr):
    i = np.argmax(s >= thr)
    return z[i] if (s >= thr).any() else np.nan
