"""All figures + results csv from the cached simulations.
Usage: python figs.py [h]      h = source depth, 2.5 (main grid, default) or 1.35 (high-gamma grid, prefix h1.35_)"""
import os, sys, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rungrid as rg, events as ev
from simcore import simulate, HERE

EQ = 3 ** 0.5
OUT = HERE
GCOL = {0.8: "#7a0177", 1.0: "#c51b8a", 1.3: "#f768a1", 1.6: "#fdae6b", 2.0: "#41ab5d", 2.5: "#1d91c0", 3.0: "#253494"}
TYPES = ["simultaneous", "apex-first", "base-peel", "mixed", "no split", "arrested", "unresolved"]
TYPE_COL = {"simultaneous": "#bdbdbd", "apex-first": "#1b9e77", "base-peel": "#d95f02", "mixed": "#7570b3",
            "no split": "#fff7bc", "arrested": "#fcbba1", "unresolved": "#ffffff"}
VALID = ("simultaneous", "apex-first", "base-peel", "mixed")
PCOL = ["#7570b3", "#1b9e77", "#d95f02"]            # pair colours A-B, A-C, B-C
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
CFG = dict(H=2.5, pre="", E=rg.E_LIST, G=rg.GAMMAS, B=rg.BETAS)


def save(fig, name):
    name = CFG["pre"] + name
    fig.savefig(os.path.join(OUT, name), dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig); print("saved", name)


KEYS = ["e", "gamma", "beta", "h", "delta", "n_runs", "n_valid", "type", "frac_apexfirst", "p_apex", "p_base", "gap_ratio", "ctrl_gap", "floor",
        "z1_mean", "z1_sd", "z2_mean", "z2_sd", "t1_mean", "t1_sd", "t2_mean", "t2_sd", "gap_mean", "gap_sd", "gap_median",
        "z_stdx_mean", "frac_stdx", "z_stdy_mean", "frac_stdy", "frac_apex", "min_final_sep", "z_end", "T_run",
        "x1", "x2", "y1", "y1_sd", "y2", "y2_sd", "d_apex1", "d_apex1_sd", "d_apex2", "d_apex2_sd", "d_base1", "d_base1_sd",
        "d_base2", "d_base2_sd", "dist_to_plane1", "dist_to_plane2", "disp1", "disp1_sd", "disp2", "disp2_sd"]


def write_csv(rows, name):
    with open(os.path.join(OUT, CFG["pre"] + name), "w", newline="") as f:
        w = csv.writer(f); w.writerow(KEYS)
        for r in rows:
            out = []
            for k in KEYS:
                v = r.get(k, "")
                if isinstance(v, (float, np.floating)):
                    v = "" if np.isnan(v) else f"{v:.4f}"
                out.append(v)
            w.writerow(out)


def by(rows, **kw):
    return [r for r in rows if all(abs(r[k] - v) < 1e-9 if isinstance(v, float) else r[k] == v for k, v in kw.items())]


def has_stats(r):
    return "t1_mean" in r


def split_rows(rows, **kw):
    """(solid, faint): rows with valid structure type; rows with statistics but arrested / no split."""
    rr = [r for r in by(rows, **kw) if has_stats(r)]
    return [r for r in rr if r["type"] in VALID], [r for r in rr if r["type"] not in VALID]


# ---------------------------------------------------------------- generic curve panels
def curves(rows, xkey, key1, key2, ylabel, fname, title, logy=False, curve_by="gamma", panel_by="beta", extra=None):
    panels = CFG["B"] if panel_by == "beta" else CFG["G"]
    curve_vals = CFG["G"] if curve_by == "gamma" else CFG["E"]
    ecol = dict(zip(CFG["E"], plt.cm.viridis(np.linspace(0.05, 0.9, len(CFG["E"])))))
    col = (lambda v: GCOL[v]) if curve_by == "gamma" else (lambda v: ecol[v])
    fig, axs = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4.4), sharey=True)
    for ax, pv in zip(np.atleast_1d(axs), panels):
        for cv in curve_vals:
            kw = {curve_by: cv, panel_by: pv}
            sol, fa = split_rows(rows, **kw)
            for rr, style in ((sol, "solid"),):
                rr = sorted(rr, key=lambda r: r[xkey])
                if not rr: continue
                x = [r[xkey] for r in rr]
                a = 1.0 if style == "solid" else 0.30
                ls1, ls2 = ("-", "--") if style == "solid" else (":", ":")
                ax.errorbar(x, [r[key1 + "_mean"] for r in rr], [r[key1 + "_sd"] for r in rr] if style == "solid" else None, color=col(cv), marker="o", ms=4, lw=1.6, ls=ls1, capsize=2, alpha=a)
                if key2:
                    ax.errorbar(x, [r[key2 + "_mean"] for r in rr], [r[key2 + "_sd"] for r in rr] if style == "solid" else None, color=col(cv), marker="^", ms=5, lw=1.2, ls=ls2, mfc="white", capsize=2, alpha=a)
        if xkey == "e": ax.axvline(EQ, color="0.6", lw=0.8, ls=":")
        if extra: extra(ax)
        if logy: ax.set_yscale("log")
        ax.set_title(rf"$\beta={pv}$" if panel_by == "beta" else rf"$\gamma={pv}$"); ax.set_xlabel("e/b" if xkey == "e" else r"$\gamma$")
    np.atleast_1d(axs)[0].set_ylabel(ylabel)
    if curve_by == "gamma": hnd = [Line2D([], [], color=GCOL[g], lw=2, label=rf"$\gamma={g}$") for g in CFG["G"]]
    else: hnd = [Line2D([], [], color=ecol[e], lw=2, label=f"e/b={e:.2f}") for e in CFG["E"]]
    if key2: hnd += [Line2D([], [], color="k", marker="o", ls="-", label="split 1 (first pair)"), Line2D([], [], color="k", marker="^", ls="--", mfc="white", label="split 2 (last pair)")]
    np.atleast_1d(axs)[-1].legend(handles=hnd, fontsize=7, frameon=False, loc="best")
    fig.suptitle(title, y=1.02, fontsize=10)
    save(fig, fname)


def fig_all_curves(rows):
    H = CFG["H"]
    src = lambda ax: ax.axhline(-H, color="0.8", lw=0.8)
    curves(rows, "e", "z1", "z2", "height $z$ of the population at the split", "fig1_heights_vs_eb.png",
           "HEIGHT of decision 1 (solid) and decision 2 (dashed) against e/b (dotted vertical: equilateral; grey line: source; arrested / no-split cells omitted)", extra=src)
    curves(rows, "gamma", "z1", "z2", "height $z$ at the split", "fig1b_heights_vs_gamma.png",
           r"HEIGHT of decision 1 (solid) and 2 (dashed) against $\gamma$, one curve per e/b", curve_by="e", extra=src)
    curves(rows, "e", "t1", "t2", "time of the split (time units)", "fig1t_times_vs_eb.png",
           "TIME of decision 1 (solid) and decision 2 (dashed) against e/b", logy=True)
    curves(rows, "gamma", "t1", "t2", "time of the split (time units)", "fig1tb_times_vs_gamma.png",
           r"TIME of decision 1 (solid) and 2 (dashed) against $\gamma$, one curve per e/b", logy=True, curve_by="e")
    curves(rows, "e", "gap", None, "time between split 1 and split 2", "fig2_gap_vs_eb.png",
           "GAP between the two decisions against e/b (mean $\\pm$ sd over seeds)")
    fig, axs = plt.subplots(2, len(CFG["B"]), figsize=(4.3 * len(CFG["B"]), 7.2), sharex=True)
    for j, b in enumerate(CFG["B"]):
        for g in CFG["G"]:
            sol, fa = split_rows(rows, gamma=g, beta=b)
            for rr, a, ls in ((sol, 1.0, "-"),):
                rr = sorted(rr, key=lambda r: r["e"])
                if not rr: continue
                axs[0, j].plot([r["e"] for r in rr], [r["gap_ratio"] for r in rr], color=GCOL[g], marker="o", ms=4, lw=1.5, ls=ls, alpha=a)
                axs[1, j].plot([r["e"] for r in rr], [r["frac_apexfirst"] for r in rr], color=GCOL[g], marker="o", ms=4, lw=1.5, ls=ls, alpha=a)
        axs[0, j].axhline(ev.GAP_RATIO, color="k", lw=0.7, ls="--"); axs[0, j].axhline(1, color="0.6", lw=0.6, ls=":")
        axs[1, j].axhline(1 / 3, color="k", lw=0.7, ls="--"); axs[1, j].set_ylim(-0.03, 1.05)
        for ax in axs[:, j]: ax.axvline(EQ, color="0.6", lw=0.8, ls=":")
        axs[0, j].set_title(rf"$\beta={b}$"); axs[1, j].set_xlabel("e/b")
    axs[0, 0].set_ylabel("gap / gap of the equilateral control\n(dashed: significance threshold)")
    axs[1, 0].set_ylabel("fraction of seeds in which the apex group\nseparates first (A-B last); dashed: chance 1/3")
    axs[0, -1].legend(handles=[Line2D([], [], color=GCOL[g], lw=2, label=rf"$\gamma={g}$") for g in CFG["G"]], fontsize=7, frameon=False)
    fig.suptitle("Evidence for an ordered pair of decisions: size of the gap relative to the equilateral control (top) and consistency of the order (bottom)", y=0.995, fontsize=10)
    save(fig, "fig2b_order_evidence.png")


# ---------------------------------------------------------------- phase map and time maps
def fig_phase(rows):
    E, G, B = CFG["E"], CFG["G"], CFG["B"]
    cm = ListedColormap([TYPE_COL[t] for t in TYPES])
    fig, axs = plt.subplots(1, len(B), figsize=(4.3 * len(B), 4.2), sharey=True)
    for ax, b in zip(np.atleast_1d(axs), B):
        M = np.full((len(G), len(E)), float(TYPES.index("unresolved")))
        for i, g in enumerate(G):
            for j, e in enumerate(E):
                r = by(rows, e=e, gamma=g, beta=b)
                if r: M[i, j] = TYPES.index(r[0]["type"])
        ax.imshow(M, cmap=cm, vmin=0, vmax=len(TYPES) - 1, origin="lower", aspect="auto")
        for i, g in enumerate(G):
            for j, e in enumerate(E):
                r = by(rows, e=e, gamma=g, beta=b)
                if r and r[0]["type"] in VALID:
                    ax.text(j, i, f"{r[0]['gap_mean']:.0f}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(len(E))); ax.set_xticklabels([f"{e:.2f}" for e in E], rotation=60, fontsize=8)
        ax.set_yticks(range(len(G))); ax.set_yticklabels(G); ax.set_xlabel("e/b"); ax.set_title(rf"$\beta={b}$")
    np.atleast_1d(axs)[0].set_ylabel(r"$\gamma$")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=TYPE_COL[t], ec="0.5") for t in TYPES], labels=TYPES, loc="lower center",
               ncol=len(TYPES), frameon=False, bbox_to_anchor=(0.5, -0.08), fontsize=8)
    fig.suptitle(f"Order of the decisions (source depth h={CFG['H']}); number = mean gap, time units", y=1.02)
    save(fig, "fig3_phase_map.png")


def fig_time_maps(rows):
    E, G, B = CFG["E"], CFG["G"], CFG["B"]
    items = [("t1_mean", "time of split 1", True), ("t2_mean", "time of split 2", True), ("gap_mean", "gap (time units)", True),
             ("z1_mean", "height $z_1$ of split 1", False), ("z2_mean", "height $z_2$ of split 2", False)]
    fig, axs = plt.subplots(len(items), len(B), figsize=(4.0 * len(B), 2.9 * len(items)), squeeze=False)
    for r_, (key, lab, lg) in enumerate(items):
        allv = [r[key] for r in rows if has_stats(r) and r["type"] in VALID and np.isfinite(r[key]) and (r[key] > 0 or not lg)]
        lo, hi = (min(allv), max(allv)) if allv else (1, 10)
        for c_, b in enumerate(B):
            ax = axs[r_, c_]; M = np.full((len(G), len(E)), np.nan)
            for i, g in enumerate(G):
                for j, e in enumerate(E):
                    r = by(rows, e=e, gamma=g, beta=b)
                    if r and has_stats(r[0]) and r[0]["type"] in VALID: M[i, j] = r[0][key]
            kw = dict(norm=LogNorm(vmin=max(lo, 1e-2), vmax=hi)) if lg else dict(vmin=lo, vmax=hi)
            ax.imshow(np.ma.masked_invalid(M), cmap="viridis_r" if key.startswith("z") else "magma_r", origin="lower", aspect="auto", **kw)
            for i in range(len(G)):
                for j in range(len(E)):
                    if np.isfinite(M[i, j]):
                        dark = (M[i, j] < np.sqrt(lo * hi)) if lg else (M[i, j] > (lo + hi) / 2)
                        ax.text(j, i, f"{M[i, j]:.0f}" if lg else f"{M[i, j]:.1f}", ha="center", va="center", fontsize=6.5, color="w" if dark else "k")
                    else:
                        ax.text(j, i, "·", ha="center", va="center", color="0.7", fontsize=8)
            ax.set_xticks(range(len(E))); ax.set_xticklabels([f"{e:.2f}" for e in E], rotation=60, fontsize=7)
            ax.set_yticks(range(len(G))); ax.set_yticklabels(G, fontsize=8)
            if r_ == 0: ax.set_title(rf"$\beta={b}$")
            if c_ == 0: ax.set_ylabel(lab + r"   ($\gamma$)", fontsize=8.5)
            if r_ == len(items) - 1: ax.set_xlabel("e/b")
    fig.suptitle(f"When and where the two decisions happen (h={CFG['H']}; dots: arrested / no split / unresolved)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save(fig, "fig3b_time_height_maps.png")


# ---------------------------------------------------------------- timelines and mode spreads
def example_grid(rows, which, fname, ylabel, title, es=None, gs=None, b=1.6):
    H = CFG["H"]; es = es or [1.0, EQ, 3.0]; gs = gs or [g for g in (1.3, 3.0) if g in CFG["G"]]
    fig, axs = plt.subplots(len(gs), len(es), figsize=(4.4 * len(es), 3.4 * len(gs)), sharex=True, squeeze=False)
    for i, g in enumerate(gs):
        for j, e in enumerate(es):
            ax = axs[i, j]; recs = ev.load_cell(e, g, b, H)
            for r in recs:
                if which == "sep":
                    for p in range(3): ax.plot(r["z"], r["sep"][:, p], color=PCOL[p], lw=0.5, alpha=0.3)
                else:
                    ax.plot(r["z"], r["stdx"], color=PCOL[2], lw=0.5, alpha=0.3); ax.plot(r["z"], r["stdy"], color=PCOL[1], lw=0.5, alpha=0.3)
            ax.axhline(0.15, color="k", lw=0.6, ls=":")
            rr = by(rows, e=e, gamma=g, beta=b)[0]
            if has_stats(rr):
                for zz, sd, ls in ((rr["z1_mean"], rr["z1_sd"], "--"), (rr["z2_mean"], rr["z2_sd"], "-")):
                    ax.axvspan(zz - sd, zz + sd, color="0.88", zorder=0); ax.axvline(zz, color="k", ls=ls, lw=1)
            ax.set_ylim(0, 1.0 if which == "sep" else 0.8); ax.set_xlim(-H, 0.02)
            lab = " (equilateral)" if abs(e - EQ) < 1e-6 else ""
            ax.set_title(rf"$e/b={e:.2f}${lab}, $\gamma={g}$, $\beta={b}$" + f"\n{rr['type']}", fontsize=9)
            if i == len(gs) - 1: ax.set_xlabel("height $z$ of the population")
            if j == 0: ax.set_ylabel(ylabel)
    if which == "sep": hnd = [Line2D([], [], color=c, label=l) for c, l in zip(PCOL, ["A-B (base pair)", "A-C", "B-C"])]
    else: hnd = [Line2D([], [], color=PCOL[2], label="std of $x$ (base-pair direction)"), Line2D([], [], color=PCOL[1], label="std of $y$ (toward the apex)")]
    hnd += [Line2D([], [], color="k", ls="--", label="mean $z$, split 1"), Line2D([], [], color="k", label="mean $z$, split 2")]
    fig.legend(handles=hnd, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(title, y=1.0)
    save(fig, fname)


# ---------------------------------------------------------------- side views and location summary
def mean_path(recs, tmax=80.0, n=800):
    tt = np.linspace(0, tmax, n)
    return np.stack([np.stack([np.interp(tt, r["t"], r["pos"][:, k]) for k in range(3)], 1) for r in recs]).mean(0)


def fig_sideview(rows, panels, fname, title, gammas=(1.3, 2.0, 3.0)):
    H = CFG["H"]; gammas = [g for g in gammas if g in CFG["G"]]
    fig, axs = plt.subplots(1, len(panels), figsize=(3.9 * len(panels), 4.6), sharey=True, squeeze=False)
    for ax, (e, b) in zip(axs[0], panels):
        for gi, g in enumerate(gammas):
            recs = ev.load_cell(e, g, b, H)
            if not recs: continue
            P = mean_path(recs); ax.plot(P[:, 1], P[:, 2], color=GCOL[g], lw=1.6)
            rr = by(rows, e=e, gamma=g, beta=b)[0]
            if has_stats(rr):
                faint = rr["type"] not in VALID
                ax.plot(rr["y1"], rr["z1_mean"], "o", color=GCOL[g], mec="k", ms=7, zorder=5, alpha=0.4 if faint else 1)
                ax.plot(rr["y2"], rr["z2_mean"], "^", color=GCOL[g], mec="k", ms=8, zorder=5, alpha=0.4 if faint else 1)
            if rr["type"] not in VALID:
                ax.text(0.03, 0.04 + 0.07 * gi, rf"$\gamma={g}$: {rr['type']}", transform=ax.transAxes, fontsize=7, color=GCOL[g])
        ax.axhline(0, color="0.4", lw=1.0); ax.axvline(e / 3, color="0.8", lw=0.8, ls=":")
        ax.plot([0, e], [0, 0], "*", color="k", ms=10)
        ax.text(0, 0.05, "base", ha="center", fontsize=7); ax.text(e, 0.05, "apex", ha="center", fontsize=7)
        ax.set_xlabel("$y$ (toward the apex)"); ax.set_title(rf"$e/b={e:.2f}$, $\beta={b}$", fontsize=10); ax.set_ylim(-H * 1.02, 0.25)
    axs[0, 0].set_ylabel("height $z$ (archetype plane at 0)")
    hnd = [Line2D([], [], color=GCOL[g], lw=2, label=rf"$\gamma={g}$") for g in gammas]
    hnd += [Line2D([], [], color="k", marker="o", ls="", label="split 1"), Line2D([], [], color="k", marker="^", ls="", label="split 2")]
    fig.legend(handles=hnd, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(title, y=1.0)
    save(fig, fname)


def fig_location(rows, b=1.6):
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.0))
    spec = [("disp", "in-plane displacement of the mean\nfrom the centroid line"), ("d_apex", "distance of the population mean to the apex $a_3$"),
            ("d_base", "distance to the nearest base archetype"), ("dist_to_plane", "distance to the archetype plane $|z|$")]
    for ax, (k, lab) in zip(axs, spec):
        for g in CFG["G"]:
            sol, _ = split_rows(rows, gamma=g, beta=b); rr = sorted(sol, key=lambda r: r["e"])
            if not rr: continue
            x = [r["e"] for r in rr]
            ax.errorbar(x, [r[k + "1"] for r in rr], [r.get(k + "1_sd", 0) or 0 for r in rr], color=GCOL[g], marker="o", ms=4, lw=1.5, capsize=2)
            ax.errorbar(x, [r[k + "2"] for r in rr], [r.get(k + "2_sd", 0) or 0 for r in rr], color=GCOL[g], marker="^", ms=5, lw=1.1, ls="--", mfc="white", capsize=2)
        ax.axvline(EQ, color="0.6", lw=0.8, ls=":"); ax.set_xlabel("e/b"); ax.set_ylabel(lab, fontsize=9)
    axs[-1].legend(handles=[Line2D([], [], color=GCOL[g], lw=2, label=rf"$\gamma={g}$") for g in CFG["G"]] + [
        Line2D([], [], color="k", marker="o", label="split 1"), Line2D([], [], color="k", marker="^", ls="--", mfc="white", label="split 2")], fontsize=7, frameon=False)
    fig.suptitle(rf"Where the splits happen in space ($\beta={b}$)", y=1.03)
    save(fig, "fig8_location_summary.png")


def run(h):
    if abs(h - 1.35) < 1e-9:
        CFG.update(H=1.35, pre="h1.35_", E=[1.0, 1.5, EQ, 2.0, 3.0, 4.0], G=[1.3, 1.6, 2.0, 2.5, 3.0], B=[1.0, 1.6, 2.0])
    rows = ev.aggregate(h=CFG["H"], delta=0.15, e_list=CFG["E"], gammas=CFG["G"], betas=CFG["B"])
    write_csv(rows, "results.csv")
    for d in (0.10, 0.25):
        write_csv(ev.aggregate(h=CFG["H"], delta=d, e_list=CFG["E"], gammas=CFG["G"], betas=CFG["B"]), f"results_delta{d:.2f}.csv")
    fig_all_curves(rows); fig_phase(rows); fig_time_maps(rows)
    example_grid(rows, "sep", "fig4_timelines_pair_separation.png", "distance between group centroids",
                 "Pair-separation timelines (thin: 16 seeds; grey bands: mean $\\pm$ sd of the split heights)")
    example_grid(rows, "std", "fig5_mode_spreads.png", "spread of cells in $x$ and $y$",
                 "In-plane spreads against height (observables of the $x$ and $y$ linear modes)")
    fig_sideview(rows, [(1.0, 1.6), (EQ, 1.6), (2.5 if 2.5 in CFG["E"] else 3.0, 1.6), (4.0, 1.6)], "fig7_sideview_vs_eb_beta1.6.png",
                 r"Side view: mean path and the location of split 1 (circle) and split 2 (triangle), $\beta=1.6$")
    fig_sideview(rows, [(3.0, 1.0), (3.0, 1.6), (3.0, 2.0)], "fig7b_sideview_vs_beta_eb3.png", r"Side view, $e/b=3$: dependence on $\beta$")
    fig_sideview(rows, [(EQ, 1.0), (EQ, 1.6), (EQ, 2.0)], "fig7c_sideview_vs_beta_equilateral.png", r"Side view, equilateral triangle: dependence on $\beta$")
    fig_location(rows)
    return rows


if __name__ == "__main__":
    run(float(sys.argv[1]) if len(sys.argv) > 1 else 2.5)
