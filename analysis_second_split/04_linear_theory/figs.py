import sys, io, contextlib, numpy as np
from th import *
from two_stage import tau_path
from predict import predict
PERP = os.path.join(ROOT, "paper figures", "3 tragets source perpendicular"); sys.path.insert(0, PERP)
from perpendicular_common import run_perp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
N = 40; S = 10; h = 2.5
cm = plt.get_cmap("viridis")
E_CACHE = {1.73205: 1.73}      # the cached sweep used e/b=1.73


def sim(e, g, b, sd):
    with contextlib.redirect_stdout(io.StringIO()):
        return run_perp(sd, g, b, E_CACHE.get(round(e, 5), e), 1.0, h)


def Lam(e, g, b):
    tau, P = tau_path(e, g, b, h, 0.004, 24.0); m = mode_lambdas(P, e, g, b, 1.0)
    de2, d32, ce, c3, se, s3 = pieces(P[:, 1], P[:, 2], e, g, b, 1.0); T_ = 2 * ce + c3
    return P, m["lx"] / T_, m["ly"] / T_


# ---------------- Fig 1: growth rates per unit log-height and order map -----------------------------
fig, axs = plt.subplots(2, 3, figsize=(16, 8.5))
ax = axs[0, 0]
for e, c in zip([0.6, 1.0, 1.73205, 2.5, 3.0, 4.0], cm(np.linspace(0, .9, 6))):
    P, Lx, Ly = Lam(e, 1.3, 1.6); ax.plot(P[:, 2], Ly, color=c, label=f"e/b={e:.2f}"); ax.plot(P[:, 2], Lx, color=c, ls="--", lw=1)
ax.axhline(0, color="k", lw=.6); ax.set_xlim(-2.5, 0); ax.set_ylim(-2, 4); ax.set_xlabel("z"); ax.set_ylabel(r"$\Lambda=\lambda/\tau$ (growth per unit log-height)")
ax.set_title(r"$\gamma$=1.3, $\beta$=1.6: solid $\Lambda_y$ (apex mode), dashed $\Lambda_x$ (base-pair mode)", fontsize=9); ax.legend(fontsize=7, ncol=2)
ax = axs[0, 1]
for g, c in zip([1.0, 1.3, 2.0, 3.0], cm(np.linspace(0, .9, 4))):
    P, Lx, Ly = Lam(2.5, g, 1.6); ax.plot(P[:, 2], Ly, color=c, label=f"$\\gamma$={g}"); ax.plot(P[:, 2], Lx, color=c, ls="--", lw=1)
ax.axhline(0, color="k", lw=.6); ax.set_xlim(-2.5, 0); ax.set_ylim(-2, 6); ax.set_xlabel("z"); ax.set_title(r"e/b=2.5, $\beta$=1.6: solid $\Lambda_y$, dashed $\Lambda_x$", fontsize=9); ax.legend(fontsize=8)
ax = axs[0, 2]
for b, c in zip([1.0, 1.6, 2.0, 3.0], cm(np.linspace(0, .9, 4))):
    P, Lx, Ly = Lam(2.5, 1.3, b); ax.plot(P[:, 2], Ly, color=c, label=f"$\\beta$={b}"); ax.plot(P[:, 2], Lx, color=c, ls="--", lw=1)
ax.axhline(0, color="k", lw=.6); ax.set_xlim(-2.5, 0); ax.set_ylim(-2, 6); ax.set_xlabel("z"); ax.set_title(r"e/b=2.5, $\gamma$=1.3: solid $\Lambda_y$, dashed $\Lambda_x$", fontsize=9); ax.legend(fontsize=8)
es = np.linspace(0.3, 4.5, 200); zs = -np.linspace(0.02, 2.5, 200)
for ax, (g, b) in zip(axs[1], [(1.3, 1.6), (2.0, 1.6), (3.0, 1.0)]):
    E, Z = np.meshgrid(es, zs); Y = E / 3
    de2, d32, ce, c3, se, s3 = pieces(Y, Z, E, g, b, 1.0); tau_ = 2 * ce + c3
    D = lam_diff(Y, Z, E, g, b, 1.0) / tau_
    lim = np.nanpercentile(abs(D), 95)
    im = ax.pcolormesh(E, Z, D, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), shading="auto")
    ax.contour(E, Z, D, [0], colors="k", linewidths=1.5); ax.axvline(np.sqrt(3), color="g", ls="--", lw=1); ax.set_xlabel("e/b"); ax.set_ylabel("z")
    ax.set_title(f"$\\gamma$={g}, $\\beta$={b}: $\\Lambda_y-\\Lambda_x$ (red: apex mode first)\ngreen: $e=\\sqrt{{3}}$; black: zero set", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=.04)
fig.tight_layout(); fig.savefig("fig1_growth_rates_and_order_map.png", dpi=130); plt.close(fig)

# ---------------- Fig 2: side view with predicted split points --------------------------------------
fig, axs = plt.subplots(1, 5, figsize=(20, 4.6), sharey=True)
for ax, e in zip(axs, [1.0, 1.73205, 2.5, 3.0, 4.0]):
    g, b = 1.3, 1.6; tau, P = tau_path(e, g, b, h, 0.004, 24.0); p = predict(e, g, b)
    ax.plot(P[:, 1], P[:, 2], "k", lw=2, label="coherent path (x=0)")
    cached = e <= 3.0
    if cached:
        for dim, mk, col, nm in ((1, "^", "C3", "sim: std_y=0.15"), (0, "s", "C0", "sim: std_x=0.15")):
            pts = []
            for sd in range(8):
                tr = sim(e, g, b, sd); t = tr[::S]
                if dim == 1 and sd < 8:
                    mm = t.mean(1); ax.plot(mm[:, 1], mm[:, 2], color="0.8", lw=.6)
                ok = t[:, :, dim].std(1) > 0.15
                if ok.any(): i = np.argmax(ok); pts.append(t[i].mean(0)[[1, 2]])
            pts = np.array(pts)
            if len(pts): ax.scatter(pts[:, 0], pts[:, 1], marker=mk, c=col, s=18, alpha=.7, label=nm)
    if p["z1_pred"] is not None: ax.scatter([p["y1_pred"]], [p["z1_pred"]], marker="*", s=260, c="gold", edgecolors="k", zorder=5, label=f"predicted split 1 ({p['order_first_mode']}-mode)")
    if p["z2_pred"] is not None: ax.scatter([p["y2_pred"]], [p["z2_pred"]], marker="P", s=160, c="orange", edgecolors="k", zorder=5, label="predicted split 2")
    ax.scatter([0], [0], marker="D", c="tab:blue", s=60, zorder=6, label="base pair (x=+-1)"); ax.scatter([e], [0], marker="D", c="tab:green", s=60, zorder=6, label="apex a3")
    ax.axhline(0, color="k", lw=.5); ax.set_title(f"e/b={e:.2f}  (first: {p['order_first_mode']})"); ax.set_xlabel("y (toward apex)"); ax.set_xlim(-0.3, e + 0.3)
axs[0].set_ylabel("z (height below the plane)"); axs[0].legend(fontsize=6.5, loc="lower left")
fig.suptitle(r"Side view (y-z), $\gamma$=1.3, $\beta$=1.6: coherent path, predicted split points and simulated std-crossings (e/b=4 not simulated)", y=1.02)
fig.tight_layout(); fig.savefig("fig2_split_locations_side_view.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# ---------------- Fig 3: z, y of splits vs e/b -------------------------------------------------------
Es = np.linspace(0.6, 4.0, 29)


def curve(g, b):
    out = {k: [] for k in ("z1", "z2", "y1", "y2", "ok")}
    for e in Es:
        p = predict(e, g, b)
        for k, kk in (("z1", "z1_pred"), ("z2", "z2_pred"), ("y1", "y1_pred"), ("y2", "y2_pred")):
            out[k].append(np.nan if p.get(kk) is None else p[kk])
        out["ok"].append(p["validity"])
    return {k: np.array(v) for k, v in out.items()}


fig, axs = plt.subplots(2, 2, figsize=(13, 9))
for col, (sweep, vals, fixed) in enumerate([("gamma", [1.0, 1.3, 2.0, 3.0], 1.6), ("beta", [1.0, 1.6, 2.0], 1.3)]):
    for k, v in enumerate(vals):
        g, b = (v, fixed) if sweep == "gamma" else (fixed, v); c_ = curve(g, b); c = cm(k / len(vals))
        bad = c_["ok"] == "height-dispersion-dominated"
        for ax, (a, a2) in zip(axs[:, col], [(c_["z1"], c_["z2"]), (c_["y1"], c_["y2"])]):
            ax.plot(Es, np.where(bad, np.nan, a), color=c, lw=2, label=f"{sweep}={v}"); ax.plot(Es, np.where(bad, a, np.nan), color=c, lw=1, ls=":")
            ax.plot(Es, np.where(bad, np.nan, a2), color=c, lw=2, ls="--"); ax.plot(Es, np.where(bad, a2, np.nan), color=c, lw=1, ls=":")
    for r in (0, 1): axs[r, col].axvline(np.sqrt(3), color="g", lw=.8)
    axs[0, col].set_ylabel("z at split (solid: split 1, dashed: split 2)"); axs[1, col].set_ylabel("y of population mean at split"); axs[1, col].set_xlabel("e/b")
    axs[0, col].set_title(("sweep $\\gamma$ at $\\beta$=1.6" if sweep == "gamma" else "sweep $\\beta$ at $\\gamma$=1.3") + " (dotted: height-dispersion regime, unreliable)", fontsize=9); axs[0, col].legend(fontsize=8, loc="lower left")
for col in (0, 1):
    for e in (0.6, 1.0, 1.73205, 2.0, 2.5, 3.0):
        R = {0: ([], []), 1: ([], [])}
        for sd in range(8):
            t = sim(e, 1.3, 1.6, sd)[::S]
            for dim in (0, 1):
                ok = t[:, :, dim].std(1) > 0.15
                if ok.any(): i = np.argmax(ok); R[dim][0].append(t[i, :, 2].mean()); R[dim][1].append(t[i, :, 1].mean())
        first = 1 if e >= 1.8 else 0       # apex (y) first for elongated, base-pair (x) first for flat
        for r, idx in ((0, 0), (1, 1)):
            axs[r, col].errorbar([e], [np.mean(R[first][idx])], [np.std(R[first][idx])], fmt="k^", ms=5, capsize=2, label="sim, first mode (std=0.15)" if (e == 0.6 and col == 0 and r == 0) else None)
            axs[r, col].errorbar([e], [np.mean(R[1 - first][idx])], [np.std(R[1 - first][idx])], fmt="ks", mfc="none", ms=5, capsize=2, label="sim, other mode" if (e == 0.6 and col == 0 and r == 0) else None)
axs[0, 0].legend(fontsize=7, ncol=2, loc="lower left")
fig.tight_layout(); fig.savefig("fig3_split_heights_vs_e.png", dpi=130); plt.close(fig)

# ---------------- Fig 4: (gamma,beta) maps ---------------------------------------------------------
Gs = np.linspace(1.0, 3.0, 11); Bs = np.linspace(0.8, 2.4, 11)
fig, axs = plt.subplots(2, 3, figsize=(16, 8.5))
for r, e in enumerate([1.0, 3.0]):
    Z1 = np.full((len(Bs), len(Gs)), np.nan); Z2 = Z1.copy(); GZ = Z1.copy()
    for i, b in enumerate(Bs):
        for j, g in enumerate(Gs):
            p = predict(e, g, b)
            if p["z1_pred"] is not None: Z1[i, j] = p["z1_pred"]
            if p["z2_pred"] is not None: Z2[i, j] = p["z2_pred"]
            if "Gamma_zz_at_split1" in p: GZ[i, j] = p["Gamma_zz_at_split1"]
    for ax, Z, t in zip(axs[r], [Z1, Z2, GZ], ["z of split 1", "z of split 2", "Gamma_zz at split 1 (white contour = 2; <2 ok)"]):
        im = ax.pcolormesh(Gs, Bs, Z, shading="nearest", cmap="magma" if t[0] == "G" else "viridis"); fig.colorbar(im, ax=ax)
        ax.set_xlabel("gamma"); ax.set_ylabel("beta"); ax.set_title(f"e/b={e}: {t}", fontsize=9)
        if t[0] == "G": ax.contour(Gs, Bs, np.nan_to_num(Z, nan=0), [2], colors="w", linewidths=1.5)
fig.tight_layout(); fig.savefig("fig4_gamma_beta_maps.png", dpi=130); plt.close(fig)
print("figures done")
