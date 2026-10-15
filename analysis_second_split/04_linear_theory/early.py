import sys, io, contextlib, numpy as np
from th import *
from two_stage import tau_path, stage_V
PERP=os.path.join(ROOT,"paper figures","3 tragets source perpendicular"); sys.path.insert(0,PERP)
from perpendicular_common import run_perp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,3,figsize=(15,4.3))
for ax,(g,b,e) in zip(axs,[(2.0,1.6,1.0),(2.0,1.0,1.0),(1.3,1.6,1.0)]):
    tau,P=tau_path(e,g,b,2.5,0.004,24.0); m=mode_lambdas(P,e,g,b,1.0); V=stage_V(tau,m["lx"],40*1e-4)
    for s in range(8):
        with contextlib.redirect_stdout(io.StringIO()): tr=run_perp(s,g,b,e,1.0,2.5)
        t=tr[::10]; ax.semilogy(t[:,:,2].mean(1),t[:,:,0].std(1),color="0.6",lw=.8)
        if s==0:
            print(g,b,e,"seed0 first 5 mean z:",t[:5,:,2].mean(1),"\n   z at steps 0,50,100,500,1000 (x10):",[round(t[i,:,2].mean(),3) for i in (0,50,100,500,1000)],"std_x:",[round(t[i,:,0].std(),4) for i in (0,50,100,500,1000)])
    ax.semilogy(P[:,2],np.sqrt(V),"r",lw=2); ax.axhline(.15,color="k",ls=":"); ax.set_xlim(-2.6,0); ax.set_ylim(1e-3,1.5); ax.set_title(f"gamma={g} beta={b} e={e}: std_x vs z (red=linear theory)")
fig.tight_layout(); fig.savefig("early_regime_check.png",dpi=110)
