"""Path-comparison grids across e/b and gamma. Each panel shows ONE representative seed: the seed whose (z1, z2) is
closest (Euclidean) to the median (z1, z2) over the valid seeds of that (e/b, gamma, beta) cell. Full trajectories are
re-run (bit-reproducible, same seeds/params as the grid) and cached in .cache_traj.
Run: python trajfigs.py"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rungrid as rg, events as ev
from simcore import simulate, HERE
from core.cache import cached_run

H = 2.5; EQ = 3 ** 0.5; DELTA = 0.15
FCOL = {0: "#2c7fb8", 1: "#d7301f", 2: "#31a354"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
TRAJ_CACHE = os.path.join(HERE, ".cache_traj")
VALID = ("simultaneous", "apex-first", "base-peel", "mixed")


def get_traj(e, g, b, s):
    params = dict(kind="traj", e=e, gamma=g, beta=b, h=H, seed=s, N=rg.N, T=rg.T, K=rg.K, sigma=rg.SIGMA, stride=rg.STRIDE)
    def comp():
        r = simulate(e, g, b, H, s, N=rg.N, T=rg.T, K=rg.K, sigma=rg.SIGMA, out_stride=rg.STRIDE, early_stop=True)
        return dict(traj=r["traj"], A=r["A"], dt=r["dt"], steps=r["steps"])
    return cached_run(TRAJ_CACHE, "traj", params, comp)


def representative(e, g, b):
    """(seed, event dict or None) -- seed with (z1,z2) closest to the cell median."""
    recs = [(s, ev.load_run(e, g, b, H, s)) for s in rg.SEEDS]
    evs = [(s, r, ev.events(r, DELTA)) for s, r in recs if r is not None]
    ok = [(s, r, x) for s, r, x in evs if x]
    if len(ok) < 8:
        return 0, None
    Z = np.array([[x["z1"], x["z2"]] for _, _, x in ok]); med = np.median(Z, axis=0)
    k = int(np.argmin(np.linalg.norm(Z - med, axis=1)))
    return ok[k][0], ok[k][2]


def panel_data(e, g, b, rows):
    typ = [r for r in rows if abs(r["e"] - e) < 1e-9 and abs(r["gamma"] - g) < 1e-9 and abs(r["beta"] - b) < 1e-9][0]["type"]
    s, x = representative(e, g, b)
    tr = get_traj(e, g, b, s)
    S = tr["traj"].astype(float); A = tr["A"]
    fate = np.linalg.norm(S[-1][:, None, :] - A[None], axis=2).argmin(1)
    return dict(S=S, A=A, fate=fate, ev=x, seed=s, type=typ, e=e, g=g, b=b)


def draw(ax, d, view):
    S, A = d["S"], d["A"]; ia, ib = {"side": (1, 2), "top": (0, 1)}[view]
    for c in range(S.shape[1]):
        ax.plot(S[:, c, ia], S[:, c, ib], color=FCOL[d["fate"][c]], lw=0.6, alpha=0.75)
    m = S.mean(1); ax.plot(m[:, ia], m[:, ib], color="k", lw=1.6)
    for k in range(3):
        ax.plot(A[k, ia], A[k, ib], marker="*", ms=11, color=FCOL[k], mec="k", zorder=6)
    if view == "side":
        ax.axhline(0, color="0.35", lw=1.0, ls="-"); ax.set_xlim(-0.35, 3.45); ax.set_ylim(-H * 1.03, 0.3)
        ax.set_xlabel("y (toward the apex)"); ax.set_ylabel("z (plane at 0)")
    else:
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.35, 3.45); ax.set_xlabel("x (base pair)"); ax.set_ylabel("y (toward the apex)")
        ax.set_aspect("equal", adjustable="box")
    x = d["ev"]
    if x and d["type"] in VALID:
        loc = x["loc"]
        for nm, mk in (("1", "o"), ("2", "s")):
            p = (loc["y" + nm], -loc["dist_to_plane" + nm]) if view == "side" else (loc["x" + nm], loc["y" + nm])
            ax.plot(*p, marker=mk, ms=9, mfc="white", mec="k", mew=1.8, zorder=8)
        sub = f"z1={x['z1']:.2f}, z2={x['z2']:.2f}"
    else:
        sub = d["type"]
    ax.set_title(f"e/b={d['e']:.2f}  γ={d['g']}  β={d['b']}\n{sub}  (seed {d['seed']})", fontsize=8.5)


def legend(fig, view):
    h = [Line2D([], [], color=FCOL[0], label="cells ending at $a_1$"), Line2D([], [], color=FCOL[1], label="cells ending at $a_2$"),
         Line2D([], [], color=FCOL[2], label="cells ending at apex $a_3$"), Line2D([], [], color="k", lw=1.6, label="mean path"),
         Line2D([], [], marker="o", color="k", mfc="white", mew=1.8, ls="", label="split 1"),
         Line2D([], [], marker="s", color="k", mfc="white", mew=1.8, ls="", label="split 2")]
    fig.legend(handles=h, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.012))


def grid(rows, es, gs, b, view, fname):
    fig, axs = plt.subplots(len(es), len(gs), figsize=(3.3 * len(gs), (3.0 if view == "side" else 3.6) * len(es)), squeeze=False)
    for i, e in enumerate(es):
        for j, g in enumerate(gs):
            draw(axs[i, j], panel_data(e, g, b, rows), view)
            if j > 0: axs[i, j].set_ylabel("")
            if i < len(es) - 1: axs[i, j].set_xlabel("")
    legend(fig, view)
    fig.suptitle(f"{'Side view (y vs z)' if view == 'side' else 'Top view (x vs y)'}, beta={b}: representative seed per panel "
                 "(the seed whose (z1,z2) is closest to the cell median); identical axes in all panels", y=1.0, fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(os.path.join(HERE, fname), dpi=160, facecolor="white"); plt.close(fig); print("saved", fname)


def grid3d(rows, es, gs, b, fname):
    fig = plt.figure(figsize=(4.6 * len(es), 4.4 * len(gs)))
    for i, g in enumerate(gs):
        for j, e in enumerate(es):
            ax = fig.add_subplot(len(gs), len(es), i * len(es) + j + 1, projection="3d")
            d = panel_data(e, g, b, rows); S, A = d["S"], d["A"]
            for c in range(S.shape[1]):
                ax.plot(S[:, c, 0], S[:, c, 1], S[:, c, 2], color=FCOL[d["fate"][c]], lw=0.5, alpha=0.7)
            m = S.mean(1); ax.plot(m[:, 0], m[:, 1], m[:, 2], color="k", lw=1.6)
            ax.add_collection3d(Poly3DCollection([A], facecolor="#d9d9d9", edgecolor="0.4", alpha=0.35))
            for k in range(3):
                ax.plot([A[k, 0]], [A[k, 1]], [0], marker="*", ms=11, color=FCOL[k], mec="k")
            x = d["ev"]
            if x and d["type"] in VALID:
                for nm, mk in (("1", "o"), ("2", "s")):
                    l = x["loc"]; ax.plot([l["x" + nm]], [l["y" + nm]], [-l["dist_to_plane" + nm]], marker=mk, ms=8, mfc="white", mec="k", mew=1.6)
                sub = f"z1={x['z1']:.2f}, z2={x['z2']:.2f}"
            else:
                sub = d["type"]
            ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.35, 3.45); ax.set_zlim(-H * 1.03, 0.3)
            ax.set_box_aspect((2.6, 3.8, 2.6)); ax.view_init(elev=18, azim=-62)
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
            ax.set_title(f"e/b={e:.2f}  γ={g}  β={b}\n{sub}  (seed {d['seed']})", fontsize=9)
    legend(fig, "3d")
    fig.suptitle("3D paths, representative seed per panel; identical axes", y=0.995, fontsize=10)
    fig.savefig(os.path.join(HERE, fname), dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig); print("saved", fname)


if __name__ == "__main__":
    rows = ev.aggregate(h=H, delta=DELTA)
    GS = [1.0, 1.3, 2.0, 3.0]
    ES = [0.6, 1.0, EQ, 2.0, 3.0]
    for view in ("side", "top"):
        grid(rows, ES, GS, 1.6, view, f"fig9_paths_{view}_beta1.6.png")
        for b in (1.0, 2.0):
            grid(rows, [EQ, 3.0], GS, b, view, f"fig9_paths_{view}_beta{b}.png")
    grid3d(rows, [1.0, EQ, 3.0], [1.3, 2.0], 1.6, "fig10_paths_3d_beta1.6.png")
