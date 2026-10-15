import sys, io, contextlib, numpy as np
from two_stage import *
PERP=os.path.join(ROOT,"paper figures","3 tragets source perpendicular"); sys.path.insert(0,PERP)
from perpendicular_common import run_perp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
G,B,H=1.3,1.6,2.5; N=40; SIG=0.01; S=10
# dtau convergence
for dt in (0.02,0.005,0.001):
    tau,P=tau_path(2.5,G,B,H,dt,40.0); m=mode_lambdas(P,2.5,G,B,1.0); V=stage_V(tau,m["lperp"],N*SIG**2); i=np.argmax(np.sqrt(V)>=0.15); print("dtau",dt,"z1",P[i,2])
fig,axs=plt.subplots(2,3,figsize=(15,8))
for col,e in enumerate([1.0,2.5,3.0]):
    tau,P=tau_path(e,G,B,H,0.005,40.0); m=mode_lambdas(P,e,G,B,1.0); s2=N*SIG**2
    Vx=stage_V(tau,m["lx"],s2); Vy=stage_V(tau,m["lperp"],s2)
    sx=[];sy=[];zs=[]
    for s in range(8):
        with contextlib.redirect_stdout(io.StringIO()): tr=run_perp(s,G,B,e,1.0,H)
        t=tr[::S]; zs.append(t[:,:,2].mean(1)); sx.append(t[:,:,0].std(1)); sy.append(t[:,:,1].std(1))
    for row,(nm,sims,Vt) in enumerate([("std x",sx,Vx),("std y",sy,Vy)]):
        ax=axs[row,col]
        for z_,s_ in zip(zs,sims): ax.semilogy(z_,s_,color="0.6",lw=0.8)
        ax.semilogy(P[:,2],np.sqrt(Vt),"r",lw=2,label="linear theory, coherent path")
        ax.axhline(0.15,color="k",ls=":"); ax.set_xlim(-2.5,0); ax.set_ylim(5e-3,1.5); ax.set_title(f"e/b={e}: {nm} vs mean height"); ax.set_xlabel("z")
        if col==0 and row==0: ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("std_vs_z_theory_vs_sim.png",dpi=130)
print("saved")
