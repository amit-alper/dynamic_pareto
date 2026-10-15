import os, sys, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
G, B = sys.argv[1], sys.argv[2]
res = pickle.load(open(os.path.join(HERE, f"sweeps_g{G}_b{B}.pkl"), "rb"))
fig, axs = plt.subplots(2, 3, figsize=(15, 8), gridspec_kw=dict(height_ratios=[1.5, 1]), sharex="col")
CA, CB = "#2e9e4f", "#7b3294"
for c, (name, (lab, rows)) in enumerate(res.items()):
    v = np.array([r["v"] for r in rows]); ax = axs[0, c]
    for key, col, nm in (("y", CA, "apex split"), ("x", CB, "base-pair split")):
        ax.plot(v, [r["sim_" + key] for r in rows], "o", color=col, ms=8, label=f"{nm}: simulation")
        ax.plot(v, [r["pred_" + key] for r in rows], "-", color=col, lw=2, label=f"{nm}: geometry-only prediction")
    ax.axhline(0, color="0.6", lw=0.8); ax.set_title(name, fontsize=13); ax.set_ylabel("split height z  (lower = earlier)" if c == 0 else "")
    if c == 0: ax.legend(fontsize=8.5, loc="lower left")
    ax2 = axs[1, c]; sc = np.array([r["score"] for r in rows])
    first_sim = np.array([r["sim_y"] - r["sim_x"] for r in rows])      # <0: apex deeper = earlier
    ax2.plot(v, -first_sim, "o", color="k", label=r"simulation: $z_x - z_y$  (>0 apex first)")
    ax2.plot(v, [r["pred_x"] - r["pred_y"] if np.isfinite(r["pred_y"]) else np.nan for r in rows], "-", color="k", lw=1.5, alpha=.6, label="prediction")
    ax2.axhline(0, color="r", lw=1); ax2.set_xlabel(lab); ax2.set_ylabel(r"order gap  $z_x-z_y$" if c == 0 else "")
    ax3 = ax2.twinx(); ax3.plot(v, sc, "s--", color="tab:orange", ms=5, lw=1); ax3.axhline(0, color="tab:orange", lw=.5, alpha=.5)
    ax3.set_ylabel(r"static score $\lambda_y-\lambda_x$ at source" if c == 2 else "", color="tab:orange")
    if c == 0: ax2.legend(fontsize=8.5, loc="upper left")
fig.suptitle(f"Geometry alone predicts when each split happens   (γ={G}, β={B}, N=40, h=1.35)", fontsize=15)
fig.tight_layout(); out = os.path.join(HERE, f"geometry_predicts_splits_g{G}_b{B}.png"); fig.savefig(out, dpi=110); print(out)
