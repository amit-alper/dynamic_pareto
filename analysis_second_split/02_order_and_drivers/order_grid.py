"""Which decision comes first across the gamma x e/b grid (cached runs only).
Split 1 = first pair of final-fate groups whose centroids separate by DELTA; 'lone leaf' = the leaf
that separates first (C = apex a3, A/B = base-pair member). Reuses split_events from second_split.py."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import second_split as ss
from perpendicular_common import archetypes3, run_perp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

GAMMAS = [1.0, 1.3, 1.6, 2.0]; EB = [0.6, 1.0, 1.73, 2.0, 2.5, 3.0, 4.0]; BETA, H = 1.6, 2.5
apex = np.full((4, 7), np.nan); n_ok = np.zeros((4, 7), int); z1 = apex.copy(); z2 = apex.copy(); gap = apex.copy()
for i, g in enumerate(GAMMAS):
    for j, e in enumerate(EB):
        A = archetypes3(e); rs = []
        for s in range(8):
            r = ss.split_events(run_perp(s, g, BETA, e, 1.0, H), A)
            if r: rs.append(r)
        n_ok[i, j] = len(rs)
        if rs:
            apex[i, j] = np.mean([r["lone"] == 2 for r in rs])
            z1[i, j] = np.median([r["z1"] for r in rs]); z2[i, j] = np.median([r["z2"] for r in rs])
            gap[i, j] = np.median([r["t2"] - r["t1"] for r in rs])
np.set_printoptions(precision=2, suppress=True, linewidth=150)
for nm, M in [("P(apex splits first)", apex), ("usable seeds /8", n_ok), ("median z at split 1", z1),
              ("median z at split 2", z2), ("median t2-t1", gap)]:
    print(f"\n{nm}  rows gamma={GAMMAS}, cols e/b={EB}\n{M}")
fig, axs = plt.subplots(1, 3, figsize=(17, 4.2))
for ax, M, t, cm, kw in [(axs[0], apex, "P(apex a3 separates first)", "RdBu_r", dict(vmin=0, vmax=1)),
                         (axs[1], z1 - z2, "z(split 1) - z(split 2)", "viridis", {}),
                         (axs[2], gap, "median time gap t2 - t1", "viridis", {})]:
    im = ax.imshow(M, cmap=cm, aspect="auto", origin="lower", **kw); fig.colorbar(im, ax=ax)
    ax.set_xticks(range(7), [str(e) for e in EB]); ax.set_yticks(range(4), [str(g) for g in GAMMAS])
    ax.set_xlabel("e/b"); ax.set_ylabel("gamma"); ax.set_title(t)
    for i in range(4):
        for j in range(7):
            if np.isfinite(M[i, j]): ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8, color="w" if cm=="viridis" else "k")
            else: ax.text(j, i, "n/a", ha="center", va="center", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "order_grid.png"), dpi=130)
