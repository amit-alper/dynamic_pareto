import os, pickle, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); CA, CB = "#2e9e4f", "#7b3294"
fig, axs = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
for ax, (g, b) in zip(axs, (("1.3", "1.6"), ("1.8", "1.4"))):
    rows = pickle.load(open(os.path.join(HERE, f"sweeps_g{g}_b{b}.pkl"), "rb"))["elongation e/b"][1]
    e = np.array([r["v"] for r in rows]); sa = np.array([r["sim_y"] for r in rows]); sb = np.array([r["sim_x"] for r in rows])
    pa = np.array([r["pred_y"] for r in rows]); pb = np.array([r["pred_x"] for r in rows])
    ax.axvspan(0.5, 3 ** .5, color="#7b3294", alpha=.07); ax.axvspan(3 ** .5, 4.1, color="#2e9e4f", alpha=.07)
    ax.axvline(3 ** .5, color="k", ls="--", lw=1)
    ax.plot(e, sa, "o", color=CA, ms=8, label="apex split (a₃ peels off): simulation"); ax.plot(e, pa, "-", color=CA, lw=2, label="apex split: prediction from geometry")
    ax.plot(e, sb, "s", color=CB, ms=8, label="base-pair split (a₁|a₂): simulation"); ax.plot(e, pb, "-", color=CB, lw=2, label="base-pair split: prediction from geometry")
    ax.set_ylim(-1.0, 0.05); ax.set_xlim(0.5, 4.1); ax.set_xlabel("elongation  e/b"); ax.axhline(0, color="0.5", lw=.8)
    ax.text(1.1, -0.93, "base pair\nsplits first", ha="center", color=CB, fontsize=12, weight="bold")
    ax.text(2.9, -0.93, "apex splits first", ha="center", color=CA, fontsize=12, weight="bold")
    ax.text(3 ** .5 + .03, 0.02, "e/b = √3", fontsize=10); ax.set_title(f"γ={g}, β={b}", fontsize=13)
    i = -1; ax.annotate("", xy=(4.0, sa[i]), xytext=(4.0, sb[i]), arrowprops=dict(arrowstyle="<->", color="k")); ax.text(3.93, 0.5 * (sa[i] + sb[i]), "gap", ha="right", fontsize=10)
axs[0].set_ylabel("height z of the split   (lower = earlier in the descent)"); axs[0].legend(fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 0.93))
fig.suptitle("Elongating the triangle (e/b): the apex split gets steadily earlier, the base-pair split barely moves, and the order flips at e/b = √3", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "eb_summary_clean.png"), dpi=110)
