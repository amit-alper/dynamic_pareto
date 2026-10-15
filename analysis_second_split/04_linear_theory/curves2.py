import sys, io, contextlib, numpy as np
from two_stage import *
PERP=os.path.join(ROOT,"paper figures","3 tragets source perpendicular"); sys.path.insert(0,PERP)
from perpendicular_common import run_perp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
G,B,H=1.3,1.6,2.5; N=40; SIG=0.01; S=10
fig,axs=plt.subplots(1,3,figsize=(15,4.5))
for ax,e in zip(axs,[2.0,2.5,3.0]):
    sx=[];zs=[];fr=[]
    from perpendicular_common import classify, archetypes3
    for s in range(8):
        with contextlib.redirect_stdout(io.StringIO()): tr=run_perp(s,G,B,e,1.0,H)
        t=tr[::S]; zs.append(t[:,:,2].mean(1)); sx.append(t[:,:,0].std(1)); fr.append(np.mean(classify(tr[-1],archetypes3(e))==2))
    fbar=float(np.mean(fr)); 
    for z_,s_ in zip(zs,sx): ax.semilogy(z_,s_,color="0.7",lw=0.8)
    o=two_stage(e,G,B,fbar); tau,P=tau_path(e,G,B); m=mode_lambdas(P,e,G,B,1.0); Vx=stage_V(tau,m["lx"],N*SIG**2)
    ax.semilogy(P[:,2],np.sqrt(Vx),"r",lw=2,label="coherent-path linear theory")
    ax.semilogy(o["zbar"],o["sx"],"b",lw=2,label=f"two-cluster (f={fbar:.2f} = observed apex fraction)")
    ax.axhline(0.15,color="k",ls=":"); ax.axvline(o["z1"],color="g",ls="--",label=f"split 1 (y-mode, thr .15): z={o['z1']:.2f}")
    ax.set_xlim(-1.5,0); ax.set_ylim(1e-2,1.2); ax.set_title(f"e/b={e}: std x vs mean height"); ax.set_xlabel("z"); ax.legend(fontsize=7)
    print(f"e={e}: f_obs={fbar:.2f} z1={o['z1']:.3f} z2_coh={o['z2_coh']:.3f} z2_2cl={o['z2']:.3f}  (sim std_x crossing see step2)")
fig.tight_layout(); fig.savefig("std_x_two_cluster.png",dpi=130)
