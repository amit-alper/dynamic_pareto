import numpy as np
from th import *
from two_stage import tau_path, stage_V
# 1. decomposition identity lambda_n = sum_i c_i [rho_i cos^2 - sin^2]
rng=np.random.default_rng(1); mx=0
for _ in range(500):
    e=rng.uniform(0.4,4); g=rng.uniform(0.6,4); b=rng.uniform(0.3,2.5); Y=rng.uniform(0,3); z=-rng.uniform(0.05,2.5)
    x=np.array([0,Y,z]); A=arche(e); r=x-A; d=np.linalg.norm(r,axis=1); c=b*g*d**(g-2); rho=b*g*d**g-g+1
    for n in (np.array([1.,0,0]),np.array([0,1.,0])):
        cos2=((r@n)/d)**2; lam=(c*(rho*cos2-(1-cos2))).sum()
        lx,ly,_=lam_xy(Y,z,e,g,b,1.0); mx=max(mx,abs(lam-(lx if n[0]==1 else ly)))
print("decomposition identity max error:",mx)
# 2. location rule  Gamma(z1)=0.5 ln(1+2 lam x_p^2/(N sigma^2))
N=40;SIG=0.01;THR=0.15;S2=N*SIG**2
print(" e  gamma beta  mode  z1     Gamma(z1)  rule     lam(z1)")
for (G,B) in [(1.3,1.6),(2.0,1.6),(1.3,1.0),(3.0,1.6)]:
    for e in [0.6,1.0,2.0,2.5,3.0]:
        tau,P=tau_path(e,G,B,2.5,0.004,24.0); m=mode_lambdas(P,e,G,B,1.0)
        for k,lam in (("x",m["lx"]),("y",m["ly"])):
            V=stage_V(tau,lam,S2); ok=np.sqrt(V)>=THR
            if not ok.any() or P[np.argmax(ok),2]>-0.03: continue
            i=np.argmax(ok); Gam=np.sum(lam[:i])*0.004; rule=0.5*np.log(1+2*lam[i]*THR**2/S2)
            print(f"{e:4.1f} {G:4.1f} {B:4.1f}   {k}   {P[i,2]:+.3f}   {Gam:6.2f}    {rule:6.2f}   {lam[i]:.2f}")
