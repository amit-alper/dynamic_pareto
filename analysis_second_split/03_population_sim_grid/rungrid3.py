"""Location pass (record v3): same simulations as rungrid.py (bit-identical, same seeds/params) but the derived record
also keeps the population-mean position (x,y,z) and the three final-fate group centroids at every stored step, so
the 3-D LOCATION of each split can be computed. Jobs run in reverse order so this can run next to rungrid.py.
Usage: python rungrid3.py [main|h35] [n_workers]"""
import os, sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simcore import simulate
import rungrid as rg
from core.cache import cached_run

VERSION3 = 3


def derive3(res):
    d = rg.derive(res)
    S = res["traj"].astype(np.float64)
    A = res["A"]; fate = d["fate"]
    cen = np.full((S.shape[0], 3, 3), np.nan)
    for g in range(3):
        idx = np.where(fate == g)[0]
        if len(idx): cen[:, g] = S[:, idx, :].mean(1)
    d["pos"] = S.mean(1).astype(np.float32)          # (n,3) population mean position
    d["cen"] = cen.astype(np.float32)                # (n,3,3) group centroids
    d["A"] = A; d["src"] = res["src"]
    return d


def params3(e, g, b, h, s, early=True):
    return dict(kind="sim3d", v=VERSION3, e=e, gamma=g, beta=b, h=h, seed=s, N=rg.N, T=rg.T, K=rg.K,
                sigma=rg.SIGMA, stride=rg.STRIDE, early=early)


def job3(args):
    e, g, b, h, s = args
    def comp():
        return derive3(simulate(e, g, b, h, s, N=rg.N, T=rg.T, K=rg.K, sigma=rg.SIGMA, out_stride=rg.STRIDE, early_stop=True))
    return args, cached_run(rg.CACHE, "sim", params3(e, g, b, h, s), comp)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "main"
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    h = {"h35": 3.5, "h135": 1.35}.get(tag, 2.5)
    if tag == "h35": es, gs, bs = [1.0, 3 ** 0.5, 3.0], [1.3, 2.0, 3.0], [1.6, 2.0]
    elif tag == "h135": es, gs, bs = [1.0, 1.5, 3 ** 0.5, 2.0, 3.0, 4.0], [1.3, 1.6, 2.0, 2.5, 3.0], [1.0, 1.6, 2.0]
    else: es, gs, bs = rg.E_LIST, rg.GAMMAS, rg.BETAS
    jobs = [(e, g, b, h, s) for e in es for g in gs for b in bs for s in rg.SEEDS][::-1]
    print(len(jobs), "runs", flush=True); t0 = time.time()
    with Pool(nw) as p:
        for i, _ in enumerate(p.imap_unordered(job3, jobs, chunksize=2)):
            if i % 100 == 0: print(f"{i}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
