"""Does early stopping (settled population) change fates or events relative to the full T=800 run?"""
import sys, numpy as np
sys.path.insert(0,'.')
import rungrid as rg, events as ev
cases=[(1.0,1.3,1.6),(3.0,1.3,1.6),(3**0.5,2.0,1.0),(3.0,3.0,2.0),(0.6,1.0,1.0),(2.5,2.0,0.85),(4.0,1.6,1.6),(1.5,0.8,1.6)]
bad=0
for e,g,b in cases:
    for s in range(3):
        _,a=rg.job((e,g,b,2.5,s,"main",True)); _,f=rg.job((e,g,b,2.5,s,"full",False))
        same_fate=np.array_equal(a["fate"],f["fate"])
        ea=ev.events(a,0.15); ef=ev.events(f,0.15)
        dt=None if not(ea and ef) else (abs(ea["t1"]-ef["t1"]),abs(ea["t2"]-ef["t2"]))
        print(f"e={e:.2f} g={g} b={b} s={s}: T_run early={a['T_run']:.0f} full={f['T_run']:.0f} fates_same={same_fate} dEvents={dt} z_end={f['z'][-1]:.3f}")
        bad+= (not same_fate)
print("fate mismatches:",bad)
