"""Map over the position of the third target a3=(ax,e): a1=(-1,0), a2=(1,0), source at the centroid, gamma=1.3, beta=1.6."""
import os, sys, pickle
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from general import *
from sourcemap import A3, peeled, GAMMA, BETA, H, SIGMA, NS
HERE = os.path.dirname(os.path.abspath(__file__))
AXS, ES = [0, 0.5, 1.0, 1.5, 2.0], [1.0, 1.5, 2.0, 3.0]
AXF, EF = np.linspace(0, 2.5, 11), np.linspace(0.6, 3.5, 12)
def geo(ax, e):
    A = A3([[-1, 0], [1, 0], [ax, e]]); return A, np.array([*A[:, :2].mean(0), -H])
def one(a):
    A, S = geo(*a); return pred(A, S, GAMMA, BETA, SIGMA)
def main():
    with Pool(12) as pool:
        fine = np.array(pool.map(one, [(a, e) for a in AXF for e in EF])).reshape(len(AXF), len(EF), 3)
        sims = []
        for ax in AXS:
            for e in ES:
                A, S = geo(ax, e); runs = pool.map(sim_one, [(A, S, GAMMA, BETA, SIGMA, s) for s in range(NS)])
                z1 = np.nanmedian([r["z1"] for r in runs]); z2 = np.nanmedian([r["z2"] for r in runs]); th = circ_mean_axis(np.array([r["th"] for r in runs]))
                p = pred(A, S, GAMMA, BETA, SIGMA)
                d = dict(ax=ax, e=e, s1=z1, s2=z2, sp=peeled(A, th) if not np.isnan(th) else -1, p1=p[0], p2=p[1], pp=peeled(A, p[2]) if not np.isnan(p[2]) else -1, shares=np.mean([r["shares"] for r in runs], 0))
                sims.append(d); print(f"a3=({ax},{e}) sim z1={z1:.2f} z2={z2:.2f} peel=a{d['sp']+1} | pred z1={d['p1']:.2f} z2={d['p2']:.2f} peel=a{d['pp']+1} | shares={np.round(d['shares'],2)}", flush=True)
    pickle.dump(dict(AXF=AXF, EF=EF, fine=fine, sims=sims), open(os.path.join(HERE, "targetmap.pkl"), "wb"))
if __name__ == "__main__": main()
