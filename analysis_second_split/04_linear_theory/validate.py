import sys, io, contextlib, numpy as np, csv
from predict import predict
from th import *
PERP=os.path.join(ROOT,"paper figures","3 tragets source perpendicular"); sys.path.insert(0,PERP)
from perpendicular_common import run_perp
from core.cache import _params_hash
S=10; T,K=800.0,40000
CD=os.path.join(PERP,".cache")
def cached(seed,g,b,e,h):
    r_stop=0.05 if g<1 else None
    p=dict(kind="perp3d",seed=seed,gamma=g,beta=b,e=e,b=1.0,h=h,N=40,T=T,K=K,sigma=0.01,r_stop=r_stop)
    return os.path.exists(os.path.join(CD,f"traj_{_params_hash(p)}.pkl"))
def sim_cross(g,b,e,h=2.5,seeds=range(8)):
    zx=[];zy=[];yx=[];yy=[]
    for s in seeds:
        if not cached(s,g,b,e,h): continue
        with contextlib.redirect_stdout(io.StringIO()): tr=run_perp(s,g,b,e,1.0,h)
        t=tr[::S]; m=t.mean(1); sx=t[:,:,0].std(1); sy=t[:,:,1].std(1)
        for sd,zl,yl in ((sx,zx,yx),(sy,zy,yy)):
            ok=sd>0.15
            if ok.any(): i=np.argmax(ok); zl.append(m[i,2]); yl.append(m[i,1])
    f=lambda a:(np.mean(a),np.std(a)) if len(a) else (np.nan,np.nan)
    return f(zx),f(zy),f(yx),f(yy),len(zx)
rows=[]
combos=[(1.3,1.6,e) for e in (0.6,1.0,1.73,2.0,2.5,3.0)]+[(g,b,1.0) for b,g in [(0.7,1.3),(0.85,1.0),(0.85,1.3),(0.85,1.5),(0.85,2.0),(0.85,2.5),(1.0,0.8),(1.0,1.0),(1.0,1.3),(1.0,1.5),(1.0,2.0),(1.0,2.5),(1.5,2.0),(1.6,0.8),(1.6,2.0)]]
print("g    b    e     | pred z_x  z_y   z2 | sim std_x-cross  std_y-cross  (n)")
for g,b,e in combos:
    p=predict(e,g,b)
    (zx,sx_),(zy,sy_),(yx,_),(yy,_),n=sim_cross(g,b,e)
    fmt=lambda v:"  none " if v is None else f"{v:+.3f}"
    print(f"{g:3.1f} {b:4.2f} {e:4.2f} | {fmt(p['z_vis_x'])} {fmt(p['z_vis_y'])} {fmt(p['z2_pred'])} | {zx:+.3f}±{sx_:.3f}  {zy:+.3f}±{sy_:.3f} ({n})")
    rows.append((g,b,e,p['z_vis_x'],p['z_vis_y'],p['z2_pred'],zx,sx_,zy,sy_,yx,yy,n))
np.save("validation_rows.npy",np.array(rows,dtype=object),allow_pickle=True)
