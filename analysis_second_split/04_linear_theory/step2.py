import sys, io, contextlib, numpy as np
from th import *
PERP=os.path.join(ROOT,"paper figures","3 tragets source perpendicular"); sys.path.insert(0,PERP)
from perpendicular_common import run_perp, classify
G,B,H=1.3,1.6,2.5; N=40; SIG=0.01
def theory_modes(e,G,B,thr=0.15,T=800.0,K=40000):
    t,P=coherent_path(e,G,B,N=N,h=H,T=T,K=K)
    m=mode_lambdas(P,e,G,B,N)
    out={}
    for k in ("lx","ly","lperp"):
        lam=m[k]; i0=first_lasting_positive(lam)
        V,i=variance_cross(t,lam,SIG,thr)
        out[k]=dict(z_onset=None if i0 is None else P[i0,2], z_vis=None if i is None else P[i,2], t_vis=None if i is None else t[i])
    return out,t,P,m
print("theory (coherent path, N=40, sigma=0.01), gamma=1.3 beta=1.6")
res={}
for e in [0.6,1.0,1.73205,2.0,2.5,3.0]:
    for thr in (0.15,0.094):
        o,_,_,_=theory_modes(e,G,B,thr)
        f=lambda d:"  none " if d is None else f"{d:+.3f}"
        print(f"e={e:5.2f} thr={thr}: x: onset {f(o['lx']['z_onset'])} vis {f(o['lx']['z_vis'])} | y: onset {f(o['ly']['z_onset'])} vis {f(o['ly']['z_vis'])} | perp: onset {f(o['lperp']['z_onset'])} vis {f(o['lperp']['z_vis'])}")
# simulation (cached) with the SAME metrics: first time std over cells of x (resp y) >= thr; mean height
S=10
print("\nsimulation (cached 8 seeds): height at first crossing of std_x / std_y >= 0.15, and group-centroid split heights (delta 0.15)")
for e in [0.6,1.0,1.73,2.0,2.5,3.0]:
    A=arche(e if e!=1.73 else 1.73)
    zx=[];zy=[];z1=[];z2=[]
    for s in range(8):
        with contextlib.redirect_stdout(io.StringIO()): tr=run_perp(s,G,B,e,1.0,H)
        t=tr[::S]; sx=t[:,:,0].std(1); sy=t[:,:,1].std(1); zz=t[:,:,2].mean(1)
        zx.append(zz[np.argmax(sx>0.15)] if (sx>0.15).any() else np.nan)
        zy.append(zz[np.argmax(sy>0.15)] if (sy>0.15).any() else np.nan)
        f=classify(tr[-1],A); g=[np.where(f==k)[0] for k in range(3)]
        cen=np.stack([t[:,i,:].mean(1) for i in g],1); tt=[]
        for (i,j) in [(0,1),(0,2),(1,2)]:
            d=np.linalg.norm(cen[:,i]-cen[:,j],axis=1); tt.append(np.argmax(d>0.15))
        z1.append(zz[min(tt)]); z2.append(zz[max(tt)])
    print(f"e={e:5.2f}: std_x cross z={np.nanmean(zx):+.2f}±{np.nanstd(zx):.2f} | std_y cross z={np.nanmean(zy):+.2f}±{np.nanstd(zy):.2f} | centroid split1 z={np.mean(z1):+.2f} split2 z={np.mean(z2):+.2f}")
