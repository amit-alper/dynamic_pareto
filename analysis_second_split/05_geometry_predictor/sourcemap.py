import os, sys, pickle
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from general import *
HERE = os.path.dirname(os.path.abspath(__file__))
GAMMA, BETA, H, SIGMA, NS = 1.3, 1.6, 1.35, 0.02, 12
SHAPES = {"symmetric, e=2": [[-1, 0], [1, 0], [0, 2]], "skewed base (a2 at 0.5)": [[-1, 0], [0.5, 0], [0, 2]], "flat, e=1.2": [[-1, 0], [1, 0], [0, 1.2]]}
W_SIM = [(1/3, 1/3, 1/3), (.7, .15, .15), (.15, .7, .15), (.15, .15, .7), (.45, .45, .1), (.45, .1, .45), (.1, .45, .45), (.5, .25, .25), (.25, .5, .25), (.25, .25, .5)]


def A3(t): return np.array([[x, y, 0] for x, y in t], float)
def peeled(A, theta):
    u = np.array([np.cos(theta), np.sin(theta)]); p = A[:, :2] @ u
    sc = [abs(p[i] - np.delete(p, i).mean()) for i in range(3)]; return int(np.argmax(sc))
def grid_w(n=14):
    return [(i / n, j / n, 1 - i / n - j / n) for i in range(1, n) for j in range(1, n - i) if 1 - i / n - j / n > 1e-9]


def one(args):
    A, w = args; S = np.array([*(np.array(w) @ A[:, :2]), -H]); p = pred(A, S, GAMMA, BETA, SIGMA); return S, p
def main():
    out = {}
    with Pool(12) as pool:
        for name, t in SHAPES.items():
            A = A3(t)
            fine = pool.map(one, [(A, w) for w in grid_w()])
            sims = []
            for w in W_SIM:
                S = np.array([*(np.array(w) @ A[:, :2]), -H])
                runs = pool.map(sim_one, [(A, S, GAMMA, BETA, SIGMA, s) for s in range(NS)])
                z1 = np.nanmedian([r["z1"] for r in runs]); z2 = np.nanmedian([r["z2"] for r in runs]); th = circ_mean_axis(np.array([r["th"] for r in runs]))
                p = pred(A, S, GAMMA, BETA, SIGMA)
                sims.append(dict(w=w, S=S, s1=z1, s2=z2, sp=peeled(A, th) if not np.isnan(th) else -1, p1=p[0], p2=p[1], pp=peeled(A, p[2]) if not np.isnan(p[2]) else -1, shares=np.mean([r["shares"] for r in runs], 0)))
                d = sims[-1]; print(f"{name:26s} w={np.round(w,2)} sim z1={d['s1']:.2f} z2={d['s2']:.2f} peel=a{d['sp']+1} | pred z1={d['p1']:.2f} z2={d['p2']:.2f} peel=a{d['pp']+1} | shares={np.round(d['shares'],2)}", flush=True)
            out[name] = dict(A=A, fine=[(S, p) for S, p in fine], sims=sims)
    pickle.dump(out, open(os.path.join(HERE, "sourcemap.pkl"), "wb"))
if __name__ == "__main__": main()
