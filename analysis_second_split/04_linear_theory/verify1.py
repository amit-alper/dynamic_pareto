import numpy as np
from th import *
rng=np.random.default_rng(0)
mx=0
for _ in range(2000):
    e=rng.uniform(0.3,4); g=rng.uniform(0.6,4); b=rng.uniform(0.3,2.5); Y=rng.uniform(-0.5,3); z=-rng.uniform(0.02,2.5); N=rng.choice([1,40])
    A=arche(e); L=transverse_operator(np.array([0,Y,z]),A,g,b,N)
    lx,ly,lyz=lam_xy(Y,z,e,g,b,N)
    mx=max(mx,abs(L[0,0]-lx),abs(L[1,1]-ly),abs(L[1,2]-lyz),abs(L[0,1]),abs(L[0,2]),abs((ly-lx)-lam_diff(Y,z,e,g,b,N)))
    assert abs(L[0,1])<1e-9 and abs(L[0,2])<1e-9
print("max |closed form - transverse_operator| over 2000 random points:",mx)
# equilateral degeneracy
e=np.sqrt(3); worst=0
for _ in range(2000):
    g=rng.uniform(0.6,4); b=rng.uniform(0.3,2.5); z=-rng.uniform(0.0,3)
    worst=max(worst,abs(lam_diff(e/3,z,e,g,b)))
print("equilateral |lambda_y-lambda_x| (Y=e/3) max over random (gamma,beta,z):",worst)
# drift of coherent path
for e in [0.6,1.0,np.sqrt(3),2.5,3.0]:
    for g in [1.3,2.0,3.0]:
        t,P=coherent_path(e,g,1.6,T=200,K=10000)
        k=np.argmin(abs(P[:,2]+0.5))
        print(f"e={e:.2f} gamma={g}: Y(z=-2.5)={P[0,1]:.3f} Y(z=-0.5)={P[k,1]:.3f}  Y at end={P[-1,1]:.3f} (e/3={e/3:.3f}) maxx={abs(P[:,0]).max():.1e}")
