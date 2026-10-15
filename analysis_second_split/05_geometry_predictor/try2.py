import pickle, numpy as np, sys
from predictor import predict
from predictor2 import predict2
from sweeps import geom
G,B=1.3,1.6
rows=pickle.load(open("sweeps_g1.3_b1.6.pkl","rb"))["elongation e/b"][1]
print("e   sim_x  pred_x(old)  pred_x(2cl, f=a3 share)  f=0.5*share  sim_y pred_y")
for r in rows:
    A,S=geom(e=r["v"]); f=float(r["shares"][2])
    o=predict2(A,S,G,B,f)
    print(f"{r['v']:.2f} {r['sim_x']:+.3f} {r['pred_x']:+.3f}  {o['x'] if o['x'] is None else round(o['x'],3)} (f={f:.2f})  sim_y {r['sim_y']:+.3f} pred_y {r['pred_y']:+.3f} -> {o['y']}", flush=True)
