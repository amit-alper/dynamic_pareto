import pickle, numpy as np, sys
from multiprocessing import Pool
from predictor2 import predict2
from sweeps import SWEEPS

def job(a):
    g,b,name,i,v,fshare,e_fn = a
    return None
def one(args):
    (g, b, name, v, fs) = args
    A, S = SWEEPS[name][2](v)
    out = {}
    for lab, f in (("true", fs), ("c0.2", 0.2), ("c0.3", 0.3)):
        out[lab] = predict2(A, S, g, b, float(np.clip(f, 0.05, 0.6)))["x"]
    return out
if __name__ == "__main__":
    tasks, meta = [], []
    for g, b in ((1.3, 1.6), (1.8, 1.4)):
        d = pickle.load(open(f"sweeps_g{g}_b{b}.pkl", "rb"))
        for name in d:
            for r in d[name][1]:
                tasks.append((g, b, name, r["v"], float(r["shares"][2]))); meta.append((g, b, name, r))
    with Pool(8) as p: res = p.map(one, tasks)
    pickle.dump((meta, res), open("eval2.pkl", "wb"))
    for lab in ("true", "c0.2", "c0.3"):
        err_old, err_new, n = [], [], 0
        for (g, b, name, r), o in zip(meta, res):
            if not np.isfinite(r["sim_x"]) or not np.isfinite(r["pred_x"]): continue
            apex_first = np.isfinite(r["sim_y"]) and r["sim_y"] < r["sim_x"] - 0.02
            if not apex_first: continue
            new = o[lab] if o[lab] is not None else r["pred_x"]
            err_old.append(r["pred_x"] - r["sim_x"]); err_new.append(new - r["sim_x"])
        print(lab, "n", len(err_old), "RMSE old %.3f new %.3f   bias old %+.3f new %+.3f" % (np.sqrt(np.mean(np.square(err_old))), np.sqrt(np.mean(np.square(err_new))), np.mean(err_old), np.mean(err_new)))
