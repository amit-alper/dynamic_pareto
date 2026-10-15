import pickle, numpy as np
from multiprocessing import Pool
from predictor3 import predict3
from sweeps import SWEEPS
def one(a):
    g,b,name,v,fs=a; A,S=SWEEPS[name][2](v)
    return predict3(A,S,g,b,f=float(np.clip(fs,.05,.6)))
if __name__=="__main__":
    tasks,meta=[],[]
    for g,b in ((1.3,1.6),(1.8,1.4)):
        d=pickle.load(open(f"sweeps_g{g}_b{b}.pkl","rb"))
        for name in d:
            for r in d[name][1]:
                tasks.append((g,b,name,r["v"],float(r["shares"][2]))); meta.append((g,b,name,r))
    with Pool(8) as p: res=p.map(one,tasks)
    pickle.dump((meta,res),open("eval3.pkl","wb"))
    for g,b in ((1.3,1.6),(1.8,1.4)):
        print("gamma",g,"beta",b)
        for (gg,bb,name,r),o in zip(meta,res):
            if (gg,bb)!=(g,b) or name!="elongation e/b": continue
            fmt=lambda v: "  nan " if v is None or not np.isfinite(v) else f"{v:+.3f}"
            print(f"  e={r['v']:.2f} first={o['first']}  z1 {fmt(o['z1'])} z2 {fmt(o['z2'])} | sim_x {fmt(r['sim_x'])} sim_y {fmt(r['sim_y'])} | old pred_x {fmt(r['pred_x'])} pred_y {fmt(r['pred_y'])}")
    # error stats on the base-first cases: second split is apex, compare to sim_y
    for lab in ("old","new"):
        e=[]
        for (g,b,name,r),o in zip(meta,res):
            if o["first"]!="x" or not np.isfinite(r["sim_y"]): continue
            p = r["pred_y"] if lab=="old" else (o["z2"] if o["z2"] is not None else 0.0)
            if lab=="old" and not np.isfinite(p): p=0.0
            e.append(p-r["sim_y"])
        print(lab,"n",len(e),"rms %.3f bias %+.3f"%(np.sqrt(np.mean(np.square(e))),np.mean(e)))
