import time, sys, numpy as np, io, contextlib
sys.path.insert(0,'.')
from simcore import *
from core.gradients import tissue_gradient
# 1. gradient identity vs core where unclamped
rng=np.random.default_rng(1); A=archetypes3(2.5)
for g,b in [(1.0,1.0),(1.3,1.6),(2.0,2.0),(3.0,1.6)]:
    X=A[0]*0.3+rng.normal(size=(40,3))*0.5+np.array([0,0,-1.0])
    a=grad_stable(X,A,g,b); c=tissue_gradient(X,A,gamma=g,mode="exp",log=True,beta=b)
    print("max|diff| core vs stable",g,b,np.abs(a-c).max())
# 2. timing
t=time.time(); r=simulate(3.0,1.3,1.6,2.5,0); print("run time",time.time()-t,"K_run",r['K_run'])
# 3. compare to cached run
sys.path.insert(0,'../../paper figures/3 tragets source perpendicular')
from perpendicular_common import run_perp
with contextlib.redirect_stdout(io.StringIO()): tr=run_perp(0,1.3,1.6,3.0,1.0,2.5)
n=min(len(r['traj']),len(tr[::5]))
d=np.abs(r['traj'][:n]-tr[::5][:n]).max(axis=(1,2))
print("max dev vs cached run at steps 0,1000,2000,3000:", d[[0,200,400,600]], " final:",d[-1])
