import pickle, numpy as np, itertools
rows=[]   # (name, A, z1, z2, shares)
t=pickle.load(open("targetmap.pkl","rb"))
for s in t["sims"]:
    A=np.array([[-1,0],[1,0],[s["ax"],s["e"]]],float); rows.append(("target",A,s["s1"],s["s2"],np.array(s["shares"])))
u=pickle.load(open("unequal_g1.3_b1.6.pkl","rb"))
for name,(lab,rr) in u.items():
    if "x offset" in name: continue
    for r in rr: rows.append((name[:6],np.asarray(r["A"])[:,:2],r["s1"],r["s2"],np.array(r["shares"])))
sw=pickle.load(open("sweeps_g1.3_b1.6.pkl","rb"))
for nm,fn in (("elong",lambda v:np.array([[-1,0],[1,0],[0,v]],float)),("apexsh",lambda v:np.array([[-1,0],[1,0],[v,2.0]],float))):
    key="elongation e/b" if nm=="elong" else "apex shifted sideways (x of a3)"
    for r in sw[key][1]:
        z=sorted([float(r["sim_x"]),float(r["sim_y"])]); rows.append((nm,fn(r["v"]),z[0],z[1],np.array(r["shares"])))
def feats(A):
    c=A.mean(0); r=np.linalg.norm(A-c,axis=1)
    d=np.sort([np.linalg.norm(A[0]-A[1]),np.linalg.norm(A[0]-A[2]),np.linalg.norm(A[1]-A[2])])
    return dict(ds=d[0],dm=d[1],dl=d[2],rmax=r.max(),rmin=r.min(),area=0.5*abs(np.cross(A[1]-A[0],A[2]-A[0])),per=d.sum(),
                gap_sm=d[1]-d[0],ratio=d[0]/d[1],rms=np.sqrt((r**2).mean()))
F=[feats(r[1]) for r in rows]; z1=np.array([r[2] for r in rows]); z2=np.array([r[3] for r in rows])
print("n",len(rows))
keys=list(F[0])
for tgt,nm in ((z1,"z1 (first split)"),(z2,"z2 (second split)"),(z2-z1,"gap z2-z1")):
    print(nm)
    for k in keys:
        x=np.array([f[k] for f in F]); 
        if np.isnan(tgt).any(): m=~np.isnan(tgt)
        else: m=np.ones(len(x),bool)
        c=np.corrcoef(x[m],tgt[m])[0,1]; print("   %-7s r=%+.2f"%(k,c))
    # best 2-feature linear fit
    best=[]
    m=~np.isnan(tgt)
    for a,b in itertools.combinations(keys,2):
        X=np.column_stack([[f[a] for f in F],[f[b] for f in F],np.ones(len(F))])[m]
        co,res,_,_=np.linalg.lstsq(X,tgt[m],rcond=None); pred=X@co
        r2=1-((tgt[m]-pred)**2).sum()/((tgt[m]-tgt[m].mean())**2).sum(); best.append((r2,a,b))
    best.sort(reverse=True); print("   best pairs:",[(round(r,3),a,b) for r,a,b in best[:4]])
