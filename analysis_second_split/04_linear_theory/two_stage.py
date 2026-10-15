"""Two-stage prediction: stage 1 = linear y-mode (apex) on the coherent path; stage 2 = x-mode of the base cluster
after an apex cluster of weight f has left (internal operator of core/cluster_ansatz)."""
import numpy as np
from scipy.integrate import solve_ivp
from th import *
from core import cluster_ansatz as ca

def tau_path(e,G,B,h=2.5,dtau=0.005,Tau=40.0):
    """coherent path in N=1 time units (real time = N*tau), x=0"""
    A=arche(e); x=source(e,h); n=int(Tau/dtau)
    P=np.empty((n+1,3)); P[0]=x
    for k in range(n):
        r=x[None,:]-A; d=np.maximum(np.linalg.norm(r,axis=1),1e-12)
        x=x-dtau*((G*B)*(d**(G-2))[:,None]*r).sum(0); P[k+1]=x
    return np.arange(n+1)*dtau,P

def stage_V(tau,lam,s2,V0=0.0):
    dt=tau[1]-tau[0]; V=np.empty(len(tau)); V[0]=V0
    for k in range(len(tau)-1):
        l=lam[k]; a=np.exp(2*l*dt); b=s2*(a-1)/(2*l) if abs(l)>1e-12 else s2*dt
        V[k+1]=V[k]*a+b
    return V

def two_stage(e,G,B,f,N=40,sigma=0.01,thr=0.15,h=2.5,dtau=0.005,Tau=40.0,detail=False):
    A=arche(e); s2=N*sigma**2
    tau,P=tau_path(e,G,B,h,dtau,Tau)
    m=mode_lambdas(P,e,G,B,1.0)
    Vy=stage_V(tau,m["lperp"],s2); Vx=stage_V(tau,m["lx"],s2)
    sy=np.sqrt(Vy); i1=np.argmax(sy>=thr) if (sy>=thr).any() else None
    sxc=np.sqrt(Vx); i2c=np.argmax(sxc>=thr) if (sxc>=thr).any() else None
    out=dict(z1=None if i1 is None else P[i1,2], z2_coh=None if i2c is None else P[i2c,2])
    if i1 is None: return out
    # split into two clusters with weights (f,1-f), separation along the unstable y-direction
    vv=m["v"][i1]; nv=np.hypot(vv[1],vv[2]); ny,nz=vv[2]/nv,-vv[1]/nv; sgn=1.0 if ny>0 else -1.0
    dirn=np.array([0,sgn*ny,sgn*nz]); Delta=thr/np.sqrt(f*(1-f))
    Xc=P[i1]+(1-f)*Delta*dirn; Xb=P[i1]-f*Delta*dirn
    X0=np.stack([Xc,Xb]); w=np.array([f,1-f])
    def rhs(t,y): return ca.velocity(y.reshape(2,3),w,A,G,B).ravel()
    ts=tau[i1:]; dtau_=tau[1]-tau[0]
    Xl=[X0]; x_=X0.copy()
    for k in range(len(ts)-1):
        f1=ca.velocity(x_,w,A,G,B); f2=ca.velocity(x_+0.5*dtau_*f1,w,A,G,B)
        f3=ca.velocity(x_+0.5*dtau_*f2,w,A,G,B); f4=ca.velocity(x_+dtau_*f3,w,A,G,B)
        x_=x_+dtau_/6*(f1+2*f2+2*f3+f4); Xl.append(x_.copy())
        if (x_[:,2]*w).sum()>-0.01: break
    Xs=np.array(Xl); ts=ts[:len(Xs)]
    lamC=np.empty(len(ts)); lamB=np.empty(len(ts))
    for k in range(len(ts)):
        H=ca.internal_operator(Xs[k],w,A,G,B); lamC[k]=H[0,0,0]; lamB[k]=H[1,0,0]
    VB=stage_V(ts,lamB,s2,Vx[i1]); VC=stage_V(ts,lamC,s2,Vx[i1])
    sx=np.sqrt(f*VC+(1-f)*VB); j=np.argmax(sx>=thr) if (sx>=thr).any() else None
    zbar=f*Xs[:,0,2]+(1-f)*Xs[:,1,2]
    out.update(z2=None if j is None else zbar[j], lamB_at_split=lamB[0], lamX_coh_at_split=m["lx"][i1],
               lamC_at_split=lamC[0], sx=sx, zC=Xs[:,0,2],zB=Xs[:,1,2],lamB=lamB,lamC=lamC,zbar=zbar,tau2=ts)
    return out

if __name__=="__main__":
    print("e   f   z1(lin y) | z2 coherent-x | z2 two-cluster | lam_x coh -> base cluster at split")
    for e in [2.0,2.5,3.0,4.0]:
        for f in [0.1,0.2,0.3,0.5]:
            o=two_stage(e,1.3,1.6,f)
            fmt=lambda v:" none" if v is None else f"{v:+.3f}"
            print(f"{e:.1f} {f:.1f}  {fmt(o['z1'])} | {fmt(o['z2_coh'])} | {fmt(o.get('z2'))} | {o.get('lamX_coh_at_split',float('nan')):.4f} -> {o.get('lamB_at_split',float('nan')):.4f}  (apex cl. {o.get('lamC_at_split',float('nan')):.4f})")
