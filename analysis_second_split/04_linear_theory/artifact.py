import sys, io, contextlib, numpy as np
from th import *
PERP=os.path.join(ROOT,"paper figures","3 tragets source perpendicular"); sys.path.insert(0,PERP)
from perpendicular_common import run_perp, classify, archetypes3
G,B,H=1.3,1.6,2.5; S=10
print("At the first time any pair of final-fate group centroids is >0.15 apart: std of x, std of y over all cells, and the mean height")
for e in [0.6,1.0,1.73,2.0,2.5,3.0]:
    R=[]
    for s in range(8):
        with contextlib.redirect_stdout(io.StringIO()): tr=run_perp(s,G,B,e,1.0,H)
        A=archetypes3(e); f=classify(tr[-1],A); g=[np.where(f==k)[0] for k in range(3)]
        t=tr[::S]; cen=np.stack([t[:,i,:].mean(1) for i in g],1)
        i1=min(np.argmax(np.linalg.norm(cen[:,i]-cen[:,j],axis=1)>0.15) for i,j in [(0,1),(0,2),(1,2)])
        R.append((t[i1,:,0].std(),t[i1,:,1].std(),t[i1,:,2].mean()))
    R=np.array(R); print(f"e={e}: std_x={R[:,0].mean():.3f} std_y={R[:,1].mean():.3f} z={R[:,2].mean():+.2f}")
# null: no instability at all -> pure diffusion cloud (gamma=2? use gaussian cloud) selection effect of grouping by a random label correlated with offset
rng=np.random.default_rng(0)
for s_ in (0.04,0.07,0.1):
    X=rng.normal(0,s_,size=(4000,40,2)); 
    # group by sign of x (2 groups) and by 3 groups from angle
    sep=[]
    for k in range(400):
        x=X[k]; lab=(np.digitize(np.arctan2(x[:,1],x[:,0]),[-np.pi/3, np.pi/3])) 
        c=[x[lab==j].mean(0) for j in range(3) if (lab==j).sum()>0]
        sep.append(max(np.linalg.norm(a-b) for a in c for b in c))
    print(f"gaussian cloud std {s_}: max centroid separation of 3 angular groups = {np.mean(sep):.3f}")
