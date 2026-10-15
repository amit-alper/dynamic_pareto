"""Batch driver: every (e, gamma, beta, seed) run is cached (core.cache.cached_run, key = all parameters) as a
compact derived record. Run:  python rungrid.py [tag]   (tag 'main' default; 'h35' = deeper-source control)."""
import os, sys, time, itertools
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simcore import simulate, HERE, ROOT
from core.cache import cached_run

CACHE = os.path.join(HERE, ".cache")
N, SIGMA, T, K, STRIDE = 40, 0.01, 800.0, 40000, 5
E_LIST = [0.6, 1.0, 1.5, 3 ** 0.5, 2.0, 2.5, 3.0, 4.0]
GAMMAS = [0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0]
BETAS = [0.85, 1.0, 1.6, 2.0]
SEEDS = list(range(16))
VERSION = 2


def derive(res):
    S = res["traj"].astype(np.float64); A = res["A"]
    fin = S[-1]
    fate = np.linalg.norm(fin[:, None, :] - A[None], axis=2).argmin(1)
    groups = [np.where(fate == g)[0] for g in range(3)]
    sizes = np.array([len(g) for g in groups])
    n = S.shape[0]
    cen = np.full((n, 3, 3), np.nan)
    for g in range(3):
        if sizes[g] > 0:
            cen[:, g] = S[:, groups[g], :].mean(1)
    sep = np.stack([np.linalg.norm(cen[:, i] - cen[:, j], axis=1) for i, j in [(0, 1), (0, 2), (1, 2)]], 1)
    return dict(t=(res["steps"] * res["dt"]).astype(np.float32), z=S[:, :, 2].mean(1).astype(np.float32),
                stdx=S[:, :, 0].std(1).astype(np.float32), stdy=S[:, :, 1].std(1).astype(np.float32),
                stdz=S[:, :, 2].std(1).astype(np.float32), sep=sep.astype(np.float32), sizes=sizes,
                fate=fate, final=fin.astype(np.float32), T_run=res["T_run"],
                dmin=np.linalg.norm(fin[:, None, :] - A[None], axis=2).min(1).astype(np.float32))


def job(args):
    e, g, b, h, s, tag, early = args
    params = dict(kind="sim3d", v=VERSION, e=e, gamma=g, beta=b, h=h, seed=s, N=N, T=T, K=K, sigma=SIGMA,
                  stride=STRIDE, early=early)
    def comp():
        return derive(simulate(e, g, b, h, s, N=N, T=T, K=K, sigma=SIGMA, out_stride=STRIDE, early_stop=early))
    r = cached_run(CACHE, "sim", params, comp)
    return (e, g, b, h, s), r


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "main"
    h = 3.5 if tag == "h35" else 2.5
    early = tag != "full"
    if tag == "h35":
        es, gs, bs = [1.0, 3 ** 0.5, 3.0], [1.3, 2.0, 3.0], [1.6, 2.0]
    else:
        es, gs, bs = E_LIST, GAMMAS, BETAS
    jobs = [(e, g, b, h, s, tag, early) for e in es for g in gs for b in bs for s in SEEDS]
    print(f"{len(jobs)} runs", flush=True); t0 = time.time()
    with Pool(10) as p:
        for i, _ in enumerate(p.imap_unordered(job, jobs, chunksize=4)):
            if i % 100 == 0:
                print(f"{i}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
