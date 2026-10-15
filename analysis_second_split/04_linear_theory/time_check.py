import sys, io, contextlib, numpy as np
from th import *
from two_stage import tau_path, stage_V
PERP=os.path.join(ROOT,"paper figures","3 tragets source perpendicular"); sys.path.insert(0,PERP)
from perpendicular_common import run_perp
from validate import cached
N=40;SIG=0.01
print(" g   b    e  | theory t_x  t_y | sim t_x  t_y | sim mean z at t_x | coherent z at sim t_x | theory z_x")
for g,b,e in [(1.3,1.6,2.5),(1.3,1.6,3.0),(1.3,1.6,1.0),(2.0,1.6,1.0),(2.0,1.5,1.0),(2.5,1.0,1.0),(2.5,0.85,1.0),(2.0,1.0,1.0),(2.0,0.85,1.0)]:
    tau,P=tau_path(e,g,b,2.5,0.004,24.0); m=mode_lambdas(P,e,g,b,1.0); out=[]
    for lam in (m["lx"],m["ly"]):
        V=stage_V(tau,lam,N*SIG**2); ok=np.sqrt(V)>=0.15; out.append(N*tau[np.argmax(ok)] if ok.any() else np.nan)
    tx=[];ty=[];zs=[]
    for s in range(8):
        if not cached(s,g,b,e,2.5): continue
        with contextlib.redirect_stdout(io.StringIO()): tr=run_perp(s,g,b,e,1.0,2.5)
        t=tr[::10]; dt=10*0.02
        for sd,L in ((t[:,:,0].std(1),tx),(t[:,:,1].std(1),ty)):
            ok=sd>0.15; L.append(np.argmax(ok)*dt if ok.any() else np.nan)
        ok=t[:,:,0].std(1)>0.15
        if ok.any(): zs.append(t[np.argmax(ok),:,2].mean())
    tc=np.nanmean(tx); i=np.argmin(abs(N*tau-tc)); 
    print(f"{g:3.1f} {b:4.2f} {e:3.1f} | {out[0]:8.2f} {out[1]:8.2f} | {np.nanmean(tx):6.2f} {np.nanmean(ty):6.2f} | {np.mean(zs):+.3f} | {P[i,2]:+.3f} | ")
