import os, pickle, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.interpolate import griddata
HERE = os.path.dirname(os.path.abspath(__file__)); TC = ["#1f77b4", "#d62728", "#2ca02c"]
# ---- source-position map
sm = pickle.load(open(os.path.join(HERE, "sourcemap.pkl"), "rb"))
fig, axs = plt.subplots(2, 3, figsize=(16, 9.5))
for c, (name, d) in enumerate(sm.items()):
    A = d["A"]; P = np.array([[S[0], S[1], p[0], p[1]] for S, p in d["fine"]])
    for r, (k, ttl) in enumerate(((2, "first split height z1"), (3, "second split height z2"))):
        ax = axs[r, c]; ok = np.isfinite(P[:, k]); tri = Polygon(A[:, :2], closed=True, fill=False, ec="k", lw=1.5)
        sc = ax.tricontourf(P[ok, 0], P[ok, 1], P[ok, k], 14, cmap="viridis", vmin=(-0.5 if r == 0 else -0.25), vmax=0)
        if c == 2: fig.colorbar(sc, ax=axs[r], shrink=.8, label="predicted z (geometry only)")
        ax.add_patch(tri); ax.scatter(A[:, 0], A[:, 1], c=TC, s=160, edgecolor="k", zorder=5)
        for s in d["sims"]:
            v = s["s1" if r == 0 else "s2"]; pk = s["sp"]
            ax.scatter(s["S"][0], s["S"][1], s=260, facecolor="white", edgecolor=TC[pk] if (r == 0 and pk >= 0) else "k", lw=3, zorder=6)
            ax.text(s["S"][0], s["S"][1], f"{v:.2f}", ha="center", va="center", fontsize=7.5, zorder=7)
        ax.set_aspect("equal"); ax.set_title(f"{name}: {ttl}" if r == 0 else ttl, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
        if r == 0: ax.set_xlabel("circles: simulation z1, ring colour = target that peels first (sim)", fontsize=8)
fig.suptitle("Source position inside the triangle (γ=1.3, β=1.6): background = geometry-only prediction, circles = simulation", fontsize=13)
fig.savefig(os.path.join(HERE, "map_source_position.png"), dpi=100, bbox_inches="tight")
# ---- target-position map
tm = pickle.load(open(os.path.join(HERE, "targetmap.pkl"), "rb"))
fig, axs = plt.subplots(1, 3, figsize=(17, 5.8)); AX, E = np.meshgrid(tm["AXF"], tm["EF"], indexing="ij")
for k, (idx, ttl) in enumerate(((0, "first split height z1"), (1, "second split height z2"))):
    ax = axs[k]; Z = tm["fine"][:, :, idx]; cs = ax.contourf(AX, E, np.where(np.isfinite(Z), Z, np.nan), 14, cmap="viridis", vmin=-0.8, vmax=0)
    for s in tm["sims"]:
        v = s["s1" if idx == 0 else "s2"]
        ax.scatter(s["ax"], s["e"], s=420, facecolor="white", edgecolor=TC[s["sp"]] if (idx == 0 and s["sp"] >= 0) else "k", lw=3, zorder=6)
        ax.text(s["ax"], s["e"], f"{v:.2f}", ha="center", va="center", fontsize=8, zorder=7)
    ax.axhline(0, color="k"); ax.set_xlabel("x of apex a3  (a1=(-1,0), a2=(1,0))"); ax.set_ylabel("height e of apex"); ax.set_title(ttl + "\n(ring colour = target peeling first, sim)" if idx == 0 else ttl)
    plt.colorbar(cs, ax=ax, shrink=.8)
ax = axs[2]; s1 = np.array([s["s1"] for s in tm["sims"]]); p1 = np.array([s["p1"] for s in tm["sims"]]); s2 = np.array([s["s2"] for s in tm["sims"]]); p2 = np.array([s["p2"] for s in tm["sims"]])
ax.plot([-.8, 0], [-.8, 0], "k-", lw=.8); ax.plot(p1, s1, "o", color="#d95f02", label="first split"); ax.plot(p2, s2, "s", color="#1b6ca8", label="second split")
ax.set_xlabel("geometry-only prediction z"); ax.set_ylabel("simulation z"); ax.legend(); ax.set_title("prediction vs simulation, all grid points")
fig.suptitle("Position of the third target (source at centroid, γ=1.3, β=1.6)", fontsize=13)
fig.savefig(os.path.join(HERE, "map_target_position.png"), dpi=100, bbox_inches="tight")
