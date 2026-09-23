"""
Phase portraits for the perpendicular geometry, extending Section 12 beyond the
zero-crossing z_c into the "unstable everywhere" (z_c = 0) regime.

Motivating question: gamma_effect_beta_gt1_equilateral.png shows the split
visually creeping toward the source as gamma grows even *after* z_c has
already hit 0 for every larger gamma in the panel (so the sign-based z_c
story alone can't explain it -- the whole axis is transversally unstable in
all those panels). Panels (a)/(b) below make the extra mechanism explicit:
lambda_x(z) itself keeps growing pointwise in gamma (and in beta) even where
its sign no longer changes.

Panel (d), the actual phase portrait, does NOT use min_z lambda_x(z) (a v1
of this script did -- it mixes two incomparable regimes on one linear scale:
"how deep is the still-trapped part of the path" and "how strong is the
instability", and washes out the boundary between gamma/beta combinations
that look very different in simulation). Instead it uses the properly
time-integrated accumulated growth along the descent,

    Gamma(z0; gamma, beta) := integral_{-h}^{z0} lambda_x(z) / zdot(z) dz,

where zdot(z) is the actual axial velocity (derived from the same tissue
gradient as lambda_x -- see the appendix of
perpendicular_instability_strengthening.tex). Gamma is the log of the
amplification factor a small symmetry-breaking seed sigma has accumulated
by height z0: x(z0) ~ sigma * exp(Gamma(z0)). This is genuinely the
"accumulated growth" quantity (not just the worst instantaneous rate),
it is smooth and monotone in both gamma and beta with no saturation, and its
sign has a clean meaning (net decay vs. net growth over the whole path) that
survives past the classical z_c=0 boundary. We plot it on a signed-log
(symlog) color scale so the wide dynamic range near the boundary doesn't get
washed out the way a linear scale does, and overlay two contours:
  - z_c = 0 (black): the classical local-sign boundary from Section 12.
  - Gamma = ln(X_PARTIAL/sigma) (white dashed): the height/parameter
    combination at which the seed noise has actually grown enough to be
    classified "partial" by the project's own thresholds
    (perpendicular_common.X_PARTIAL, SIGMA_DEFAULT) by the time the
    population reaches the plane -- the real "visible by arrival" boundary,
    strictly inside the z_c=0 boundary.

Run from anywhere:
    python "paper figures/3 tragets source perpendicular/phase_portrait.py"
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

from perpendicular_common import (
    lambda_x, z_critical, d_e, d_3, HERE, X_PARTIAL, SIGMA_DEFAULT,
)

E = np.sqrt(3.0)   # equilateral, matches gamma_effect_beta_*_equilateral.png
B = 1.0
H_GT1 = 1.35        # source depth used in the beta>1 figure
H_LT1 = 2.10        # source depth used in the beta<1 figure

VISIBLE_THRESHOLD = np.log(X_PARTIAL / SIGMA_DEFAULT)  # Gamma level = "visible by arrival"


def z_dot(z, gamma, beta, e=E, b=B, N=1.0):
    """Axial velocity along the centroid line (derived alongside lambda_x;
    see Appendix A of the companion tex note). All three archetypes lie in
    the z=0 plane, so the z-component of the tissue gradient at (0,e/3,z) is
    just -beta*gamma*z * [2*d_e(z)^(gamma-2) + d_3(z)^(gamma-2)] / N."""
    de, d3 = d_e(z, e, b), d_3(z, e)
    return -(beta * gamma / N) * z * (2.0 * de ** (gamma - 2.0) + d3 ** (gamma - 2.0))


def Gamma_total(gamma, beta, e=E, b=B, h=H_GT1, n=4000):
    """Accumulated log-growth of the transverse seed from the source to
    (just short of) the plane: integral of lambda_x(z)/zdot(z) dz. Positive
    => the seed has net-amplified by arrival (visible split some time before
    the plane); negative => it has net-decayed (still coherent)."""
    zs = np.linspace(-h, -1e-3, n)
    integrand = lambda_x(zs, gamma, beta, e, b) / z_dot(zs, gamma, beta, e, b)
    return np.trapezoid(integrand, zs)


def panel_lambda_curves(ax, beta, gammas, h, e=E, b=B, title=""):
    zs = np.linspace(-h, -1e-3, 600)
    cmap = plt.cm.viridis(np.linspace(0.05, 0.9, len(gammas)))
    for gamma, c in zip(gammas, cmap):
        vals = lambda_x(zs, gamma, beta, e, b)
        zc = z_critical(gamma, beta, e, b)
        lab = rf"$\gamma$={gamma:g}"
        if np.isfinite(zc) and zc != 0.0:
            lab += rf"  ($z_c$={zc:.2f})"
        elif zc == 0.0:
            lab += "  (unstable everywhere)"
        else:
            lab += "  (trapped)"
        ax.plot(zs, vals, color=c, lw=1.8, label=lab)
    ax.axhline(0, color="0.3", lw=0.8, ls="--")
    ax.axvline(-h, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("z  (source at $-h$, plane at 0)")
    ax.set_ylabel(r"$\lambda_x(z)$")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(-h * 1.02, 0)


def panel_gamma_vs_gamma(ax, betas, gammas, h=H_GT1, e=E, b=B):
    """Accumulated growth Gamma_total vs the shape exponent gamma, for a
    few bandwidths -- the correctly-integrated counterpart of the old
    "instability floor" panel: smooth, monotone, and signed."""
    for beta in betas:
        vals = np.array([Gamma_total(g, beta, e, b, h) for g in gammas])
        ax.plot(gammas, vals, marker="o", ms=3.5, lw=1.5,
                 label=rf"$\beta$={beta:g}" + (r"  ($\beta$>1)" if beta > 1 else
                                                (r"  ($\beta$<1)" if beta < 1 else "")))
    ax.axhline(0.0, color="0.4", lw=0.8, ls="--", label=r"$\Gamma=0$ (net decay/growth)")
    ax.axhline(VISIBLE_THRESHOLD, color="0.4", lw=0.8, ls=":",
               label=rf"$\Gamma=\ln(X_{{\rm partial}}/\sigma)\approx{VISIBLE_THRESHOLD:.2g}$ (visible by arrival)")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\Gamma_{\rm total}(\gamma,\beta)$  (accumulated growth, symlog)")
    ax.set_title("Accumulated growth keeps climbing in $\\gamma$\n"
                 "(smooth and monotone -- no saturation at the $z_c=0$ crossing)", fontsize=11)
    ax.legend(fontsize=7.5, loc="upper left")


def panel_phase_portrait(ax, gamma_range, beta_range, h, e=E, b=B, n=140, title=""):
    gammas = np.linspace(*gamma_range, n)
    betas = np.linspace(*beta_range, n)
    Z = np.zeros((n, n))
    Zc = np.zeros((n, n))
    for i, beta in enumerate(betas):
        for j, gamma in enumerate(gammas):
            Z[i, j] = Gamma_total(gamma, beta, e, b, h, n=800)
            Zc[i, j] = z_critical(gamma, beta, e, b)
    vmax = np.percentile(np.abs(Z), 98)
    norm = SymLogNorm(linthresh=1.0, vmin=-vmax, vmax=vmax, base=10)
    im = ax.pcolormesh(gammas, betas, Z, cmap="RdBu_r", norm=norm, shading="auto")

    cs1 = ax.contour(gammas, betas, Zc, levels=[0.0], colors="k", linewidths=1.6)
    ax.clabel(cs1, fmt={0.0: r"$z_c=0$ (local)"}, fontsize=8)
    cs2 = ax.contour(gammas, betas, Z, levels=[VISIBLE_THRESHOLD], colors="white",
                      linewidths=1.8, linestyles="--")
    ax.clabel(cs2, fmt={VISIBLE_THRESHOLD: "visible by arrival"}, fontsize=8)

    ax.set_xlabel(r"$\gamma$ (shape exponent)")
    ax.set_ylabel(r"$\beta$ (bandwidth)")
    ax.set_title(title, fontsize=11)
    return im


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))

    panel_lambda_curves(axes[0, 0], beta=1.60, gammas=[1.2, 2.0, 2.5, 3.0], h=H_GT1,
                         title=r"(a) $\lambda_x(z)$ vs $z$, $\beta$=1.60 ($\beta$>1), equilateral"
                               "\ncurves keep rising well past $z_c\\to0$")
    panel_lambda_curves(axes[0, 1], beta=0.85, gammas=[0.5, 1.2, 1.5, 1.9, 2.0, 2.5], h=H_LT1,
                         title=r"(b) $\lambda_x(z)$ vs $z$, $\beta$=0.85 ($\beta$<1), equilateral"
                               "\ntrapped $\\to$ late split $\\to$ unstable everywhere")

    panel_gamma_vs_gamma(axes[1, 0], betas=[0.85, 1.0, 1.6], gammas=np.linspace(0.4, 3.2, 60),
                          h=2.5)

    im = panel_phase_portrait(axes[1, 1], gamma_range=(0.4, 3.2), beta_range=(0.4, 2.2),
                               h=1.6, title="(d) Phase portrait: accumulated growth $\\Gamma_{\\rm total}$\n"
                                            r"in ($\gamma$,$\beta$), equilateral, $h$=1.6")
    cbar = fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar.set_label(r"$\Gamma_{\rm total}$ (symlog)  blue=net decay (trapped), red=net growth (split)")

    fig.suptitle("Perpendicular geometry: instability doesn't just turn on, it strengthens "
                  r"($e=\sqrt{3}\,b$, equilateral)", fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = f"{HERE}/phase_portrait.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
