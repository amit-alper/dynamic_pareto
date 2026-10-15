import os, pickle, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
res = pickle.load(open(os.path.join(HERE, "unequal_g1.3_b1.6.pkl"), "rb"))
fig, axs = plt.subplots(2, 4, figsize=(18, 7.5), gridspec_kw=dict(height_ratios=[1.6, 1]))
for c, (name, (lab, rows)) in enumerate(res.items()):
    v = [r["v"] for r in rows]; ax = axs[0, c]
    for k, col, nm in (("1", "#d95f02", "first split"), ("2", "#1b6ca8", "second split")):
        ax.plot(v, [r["s" + k] for r in rows], "o", color=col, ms=8, label=nm + ": simulation")
        ax.plot(v, [r["p" + k] for r in rows], "-", color=col, lw=2, label=nm + ": geometry-only")
    ax.axhline(0, color="0.6", lw=.8); ax.set_title(name, fontsize=10.5); ax.set_ylim(-0.55, 0.03)
    if c == 0: ax.set_ylabel("split height z (lower = earlier)"); ax.legend(fontsize=8, loc="lower right")
    b = axs[1, c]; b.plot(v, [r["sth"] for r in rows], "ko", label="simulation"); b.plot(v, [r["pth"] for r in rows], "k-", alpha=.6, label="prediction")
    b.set_xlabel(lab); b.set_ylim(-100, 100)
    if c == 0: b.set_ylabel("first-split axis angle (deg)\n90 = apex direction, 0 = base-pair direction"); b.legend(fontsize=8)
fig.suptitle("Unequal base pairing and in-plane source position  (γ=1.3, β=1.6)", fontsize=14)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "unequal_base_and_source_position.png"), dpi=105)
