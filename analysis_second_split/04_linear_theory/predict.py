"""Prediction grid: heights of split 1 and split 2 (std-based detection, thr=0.15, sigma=0.01, N=40, h=2.5).
Stage 1: linear growth of the more unstable in-plane mode on the coherent path.
Stage 2: two-cluster correction (core.cluster_ansatz) for the second mode, apex fraction f (y-first) or 1/2-1/2 (x-first)."""
import sys, csv, numpy as np
from th import *
from two_stage import tau_path, stage_V
from core import cluster_ansatz as ca
N=40; SIG=0.01; THR=0.15; H=2.5; S2=N*SIG**2; TOL=0.02

def rk4_clusters(X0,w,A,G,B,tau,i1,stop_z=-0.01):
    dt=tau[1]-tau[0]; Xl=[X0]; x_=X0.copy()
    for k in range(i1,len(tau)-1):
        f1=ca.velocity(x_,w,A,G,B); f2=ca.velocity(x_+0.5*dt*f1,w,A,G,B)
        f3=ca.velocity(x_+0.5*dt*f2,w,A,G,B); f4=ca.velocity(x_+dt*f3,w,A,G,B)
        x_=x_+dt/6*(f1+2*f2+2*f3+f4); Xl.append(x_.copy())
        if (x_[:,2]*w).sum()>stop_z: break
    return np.array(Xl)

def _flag(out):
    g=out["Gamma_zz_at_split1"]
    out["validity"]="ok" if g<2 else ("marginal" if g<4 else "height-dispersion-dominated")
    if g>=2: out["notes"]=out.get("notes","")+f" | Gamma_zz(split1)={g:.1f}: height-dispersion mode competes; simulated split is lower (earlier in z) than predicted"

def predict(e,G,B,f=0.2,dtau=0.004,Tau=24.0):
    tau,P=tau_path(e,G,B,H,dtau,Tau); m=mode_lambdas(P,e,G,B,1.0)
    z=P[:,2]; out=dict(e=e,gamma=G,beta=B,notes="")
    for k,lam in (("x",m["lx"]),("y",m["ly"])):
        i0=first_lasting_positive(lam); out["z_onset_"+k]=None if i0 is None else z[i0]
        V=stage_V(tau,lam,S2); ok=np.sqrt(V)>=THR; ii=np.argmax(ok) if ok.any() else None
        out["z_vis_"+k]=None if ii is None else z[ii]; out["_i_"+k]=ii; out["_V_"+k]=V
    zx,zy=out["z_vis_x"],out["z_vis_y"]
    if zx is None and zy is None: first="none"
    elif zy is None: first="x"
    elif zx is None: first="y"
    elif abs(zx-zy)<TOL: first="simultaneous"
    else: first="y" if zy<zx else "x"
    out["order_first_mode"]=first; out["z1_pred"]=None; out["z2_pred"]=None
    if first=="none": out["notes"]="no mode reaches threshold before the plane"; out["validity"]="n/a"; return out
    if first=="simultaneous":
        out["z1_pred"]=min(zx,zy); out["z2_pred"]=max(zx,zy); out["notes"]="modes within 0.02: single 3-way event"
        ia,ib=sorted([out["_i_x"],out["_i_y"]]); out["y1_pred"]=P[ia,1]; out["y2_pred"]=P[ib,1]; out["x1_pred"]=0.0; out["x2_pred"]=0.0
        out["Gamma_at_split1"]=float(np.sum(m["ly"][:ia])*dtau); out["t1_pred"]=N*tau[ia]; out["t2_pred"]=N*tau[ib]
        out["Gamma_zz_at_split1"]=float(np.sum(m["lz"][:ia])*dtau); _flag(out); return out
    i1=out["_i_"+first]; out["z1_pred"]=z[i1]; out["y1_pred"]=P[i1,1]; out["x1_pred"]=0.0
    lam1=m["ly"] if first=="y" else m["lx"]; out["Gamma_at_split1"]=float(np.sum(lam1[:i1])*dtau)
    out["t1_pred"]=N*tau[i1]; out["Gamma_zz_at_split1"]=float(np.sum(m["lz"][:i1])*dtau)
    other="x" if first=="y" else "y"
    out["z2_coherent_other"]=out["z_vis_"+other]
    # stage 2
    if first=="y":
        vv=m["v"][i1]; nv=np.hypot(vv[1],vv[2]); ny,nz=vv[2]/nv,-vv[1]/nv; s=1.0 if ny>0 else -1.0
        dirn=np.array([0,s*ny,s*nz]); Delta=THR/np.sqrt(f*(1-f)); w=np.array([f,1-f])
        X0=np.stack([P[i1]+(1-f)*Delta*dirn,P[i1]-f*Delta*dirn])
        Xs=rk4_clusters(X0,w,arche(e),G,B,tau,i1)
        lam=np.array([ca.internal_operator(x_,w,arche(e),G,B)[:, 0,0] for x_ in Xs])      # (k,2) x-growth of apex, base cluster
        Vx0=out["_V_x"][i1]; VC=stage_V(tau[i1:i1+len(Xs)],lam[:,0],S2,Vx0); VB=stage_V(tau[i1:i1+len(Xs)],lam[:,1],S2,Vx0)
        sx=np.sqrt(f*VC+(1-f)*VB); zb=f*Xs[:,0,2]+(1-f)*Xs[:,1,2]; yb=f*Xs[:,0,1]+(1-f)*Xs[:,1,1]
        out["lam_x_coh_at_split1"]=m["lx"][i1]; out["lam_x_base_cluster_at_split1"]=lam[0,1]
    else:
        # x-first: left/right clusters, weights 1/2, then y-growth inside a side cluster
        Delta=2*THR; w=np.array([0.5,0.5]); x0=P[i1].copy()
        X0=np.stack([x0+np.array([Delta/2,0,0]),x0-np.array([Delta/2,0,0])])
        Xs=rk4_clusters(X0,w,arche(e),G,B,tau,i1)
        lam=np.array([ca.internal_operator(x_,w,arche(e),G,B)[:, 1,1] for x_ in Xs])
        Vy0=out["_V_y"][i1]; VL=stage_V(tau[i1:i1+len(Xs)],lam[:,0],S2,Vy0)
        sx=np.sqrt(VL); zb=0.5*(Xs[:,0,2]+Xs[:,1,2]); yb=0.5*(Xs[:,0,1]+Xs[:,1,1])
        out["lam_y_coh_at_split1"]=m["ly"][i1]; out["lam_y_side_cluster_at_split1"]=lam[0,0]
    ok=sx>=THR; j=np.argmax(ok) if ok.any() else None
    out["t2_pred"]=None if j is None else N*tau[i1+j]
    out["z2_pred"]=None if j is None else float(zb[j]); out["y2_pred"]=None if j is None else float(yb[j]); out["x2_pred"]=0.0 if j is not None else None
    _flag(out)
    if j is None: out["notes"]="second mode does not reach threshold before plane (two-cluster)"
    if out["z1_pred"] is not None and out["z1_pred"]>-0.03: out["notes"]+=" | split 1 within 0.03 of the plane (unresolved before arrival)"
    if out["z2_pred"] is not None and out["z2_pred"]>-0.03: out["notes"]+=" | split 2 within 0.03 of the plane"
    return out

if __name__=="__main__":
    Es=[0.6,1.0,1.5,np.sqrt(3),2.0,2.5,3.0,4.0]; Gs=[1.0,1.3,2.0,3.0]; Bs=[1.0,1.6,2.0]
    rows=[]; n=0
    for G in Gs:
        for B in Bs:
            for e in Es:
                o=predict(e,G,B); rows.append(o); n+=1
                print(n,e,G,B,o["order_first_mode"],o["z1_pred"],o["z2_pred"],flush=True)
    keys=["e","gamma","beta","order_first_mode","x1_pred","y1_pred","z1_pred","x2_pred","y2_pred","z2_pred","Gamma_at_split1","Gamma_zz_at_split1","validity","t1_pred","t2_pred","z_onset_x","z_onset_y","z_vis_x","z_vis_y","notes"]
    with open("predictions.csv","w",newline="") as fh:
        wr=csv.writer(fh); wr.writerow(keys)
        for o in rows: wr.writerow([("" if o.get(k) is None else (f"{o[k]:.4f}" if isinstance(o[k],float) else o[k])) for k in keys])
    print("wrote predictions.csv")
