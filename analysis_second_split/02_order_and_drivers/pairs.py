import os, sys, io, contextlib
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.abspath(os.path.join(HERE,"..",".."))
PERP=os.path.join(ROOT,"paper figures","3 tragets source perpendicular")
sys.path.insert(0,ROOT); sys.path.insert(0,PERP)
from perpendicular_common import archetypes3, run_perp, classify
GAMMA,BETA,H=1.3,1.6,2.5; T,K=800.0,40000; S=10
for delta in (0.10,0.15,0.25):
    print(f"\n######## delta={delta}")
    for e in (0.6,1.0,1.73,2.0,2.5,3.0):
        A=archetypes3(e); gaps=[]; first=[]; sm=[]
        for s in range(8):
            with contextlib.redirect_stdout(io.StringIO()): tr=run_perp(s,GAMMA,BETA,e,1.0,H)
            f=classify(tr[-1],A); g=[np.where(f==k)[0] for k in range(3)]
            t=tr[::S]; cen=np.stack([t[:,i,:].mean(1) for i in g],1)
            tt={}
            for (i,j) in [(0,1),(0,2),(1,2)]:
                d=np.linalg.norm(cen[:,i]-cen[:,j],axis=1); tt[(i,j)]=np.argmax(d>delta)*S*T/K
            v=sorted(tt.values()); gaps.append(v[2]-v[0]); first.append(v[0])
            # which pair is LAST to separate (the 'second split' pair)
            last=max(tt,key=tt.get); sm.append("AB" if last==(0,1) else "AC" if last==(0,2) else "BC")
            if delta==0.15 and e in (0.6,1.0,1.73,2.5) and s<3: print(f"  e={e} seed{s} pair times AB={tt[(0,1)]:.1f} AC={tt[(0,2)]:.1f} BC={tt[(1,2)]:.1f}")
        print(f"e/b={e}: t1 mean={np.mean(first):.1f}  gap t2-t1 mean={np.mean(gaps):.1f} (min {min(gaps):.1f}, max {max(gaps):.1f})  last-to-separate pair: {dict((k,sm.count(k)) for k in set(sm))}")
