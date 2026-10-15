# Theory of the two-stage split (non-equilateral perpendicular three-target geometry)

Geometry: a1=(-1,0,0), a2=(1,0,0), a3=(0,e,0) (b=1), source (0,e/3,-h), h=2.5. N=40, sigma=0.01, detection = std over cells of
one coordinate reaches x_p=0.15. Modes: **x** = base-pair direction, **y** = toward the apex. Everything below is in `th.py`,
`predict.py`, `two_stage.py`; all closed forms were checked against `core.stability.transverse_operator` (max error 3e-11).

Status labels: **[exact]** proven/algebraic and checked numerically, **[num]** numerical (linear theory integrated along the
coherent path), **[approx]** approximation, **[conj]** conjecture.

## 1. Exact results

Notation at a point (0,Y,z): d_e^2=1+Y^2+z^2 (distance to a1,a2), d_3^2=(e-Y)^2+z^2, c_i=(beta*gamma/N) d_i^(gamma-2),
s_i=beta*gamma*d_i^gamma-gamma+2, rho_i=s_i-1 (radial curvature factor of the kernel), tau=2c_e+c_3.

* **[exact]** The coherent path stays on x=0 for all e. Its y-coordinate drifts away from e/3 unless e=sqrt(3) or gamma=2
  (drift of Y is <=0.1 down to z=-0.5 for gamma=1.3, beta=1.6; it matters near the plane, y_pred columns include it).
* **[exact]** The in-plane block of Lambda_split is diagonal in (x,y); yz is coupled, xz and xy vanish:
  lam_x = 2c_e s_e/d_e^2 - tau,
  lam_y = 2c_e s_e Y^2/d_e^2 + c_3 s_3 (e-Y)^2/d_3^2 - tau,
  lam_y - lam_x = 2c_e s_e (Y^2-1)/d_e^2 + c_3 s_3 (e-Y)^2/d_3^2.
* **[exact] Mode decomposition (the mechanism).** For any unit direction n,
  lam_n = sum_i c_i [ rho_i cos^2(theta_i) - sin^2(theta_i) ], theta_i = angle between n and the direction to archetype i.
  A target aligned with the mode contributes rho_i (positive when the kernel is in its convex tail, beta*gamma*d^gamma > gamma-1);
  a target orthogonal to the mode contributes -c_i (a pure stabilising tax). The x-mode sees only the base pair (the apex is
  orthogonal and only taxes it); the y-mode sees the apex head-on and the base pair at angle Y/d_e. Far targets are in the convex tail
  (rho_3 ~ (2e/3)^gamma), so the apex mode strengthens with e while the base-pair mode barely changes.
* **[exact] Order flip at e=sqrt(3).** At Y=e/3, d_e=d_3 and the weights cancel exactly: lam_y-lam_x = 0 for every z, gamma, beta
  at e=sqrt(3) (equilateral; verified to 2e-12). Far field (|z| large, s>0): sign(lam_y-lam_x)=sign(e^2-3), so apex-mode first for
  e>sqrt(3) (elongated), base-pair mode first for e<sqrt(3) (flat). At gamma=2 this is exact at all heights:
  Lambda_y - Lambda_x = 4*beta*(e^2-3)/9 where Lambda=lam/tau. For gamma!=2 the magnitude depends on z, but on the line Y=e/3 the zero
  set is exactly e=sqrt(3) at every z (the only zero in e in [0.3,4.5] for the three (gamma,beta) maps of fig1, bottom row). The actual path
  has Y!=e/3 for gamma!=2 (drift), which shifts the flip slightly near the plane; not quantified.
* **[exact] Height form.** dz/dt=-tau z on the axis, so the log-amplification accumulated down to height z is
  Gamma(z)=int_{|z|}^{h} Lambda(z') dz'/z', Lambda=lam/tau (independent of N and speed; generalises the two-target dGamma/dy=phi/(u|y|)).
* **[exact, gamma=2] closed form.** Lambda_x=(4beta-3)/3 (independent of e!), Lambda_y=4*beta*e^2/9-1, both constant in z, and
  z_k = -h exp(-Gamma_k*/Lambda_k), Gamma_k* = 0.5 ln(1+12*beta*Lambda_k*x_p^2/(N sigma^2)).
  Checks: beta=1.6, e=2.5: Lambda_y=3.44, z1=-1.06 (numerical -1.02; Euler step error ~4%); x-mode z=-0.298 (numerical -0.279).
  The x-mode needs beta>3/4, the apex mode beta*e^2>9/4.

## 2. What sets the location of decision 1 [num + exact structure]

z1 is the height where the *faster* mode has accumulated Gamma_mode(z1)=Gamma*, Gamma*~0.5 ln(1+2*lam*x_p^2/(N sigma^2)) = 1.2-3 (only
logarithmic in noise, N, x_p). So location = geometry of Lambda(z) (section 1), not noise:
* y-first (e>sqrt(3)): |z1| grows ~linearly in (e-sqrt(3)) (gamma=1.3, beta=1.6: z1=-0.18,-0.42,-0.66,-1.05 at e=2,2.5,3,4; slope ~0.45)
  because rho_3 ~ (2e/3)^gamma raises Lambda_y; it saturates at the source depth h (e>=4 at gamma>=2 is clipped by h=2.5).
* x-first (e<sqrt(3)): z1 is essentially independent of e (-0.006,-0.049,-0.075,-0.081 for e=0.6,1,1.5,1.73; exact independence at gamma=2),
  and z1 -> 0 (split only at the plane) for beta<=beta_c-like values (see predictions.csv: gamma<=1.3, beta<=1 gives 'none'/at-plane).
* Larger gamma and beta both push |z1| out (more convex tail, larger rho): beta 1.0->1.6->2.0 at gamma=1.3, e=3: -0.36,-0.66,-0.80.
* y1 (population mean y at the split) is the coherent-path y: ~e/3 shifted by drift toward the plane value; sim agrees to <0.02
  (fig3 bottom row, fig2). x1=0 by symmetry.

## 3. What sets the location of decision 2 [num, two-cluster approximation]

After the apex group (weight f) leaves, the base cluster's x-growth is the internal operator H_c of `core/cluster_ansatz`.
Two effects raise it relative to the coherent value: (i) the apex task is now served by the apex cluster, so the stabilising apex tax
-c_3 is shielded (w_3=p_3(x_B)/S_3 -> small); (ii) the base-pair tasks are carried by only (1-f) of the cells, rate x 1/(1-f).
At gamma=2 [exact, positions on x=0]: lam_x^B = (8 beta^2-4 beta)/(1-f) - 2 beta w_3 (N=1 units) vs coherent 8beta^2-6beta.
Numerically (gamma=1.3, beta=1.6, e=2.5, f=0.2) lam_x rises 2.38 -> 3.37 (+41%) at the moment of split 1.
Consequence: z2 lies below the coherent x-mode height (-0.09): z2 = -0.098, -0.139, -0.175, -0.21 for e=2,2.5,3,4, and it moves
deeper with e because split 1 happens earlier (the x-variance has had longer) and the apex cluster is farther (better shielding).
Sensitivity to the assumed apex fraction f in [0.1,0.5]: |dz2| <= 0.03 (f is not predicted; observed f=0.28,0.20,0.15 at e=2,2.5,3).
y2 ~ y of the base cluster (~0.76 at e=2.5), i.e. within 0.02 of the simulated value.
For e<sqrt(3) (x first) the second stage is the y-mode inside the side clusters; the model gives it stable or at the plane
(lam_y inside a side cluster <0 at e=1; at e=1.5 the crossing is at the plane, z2=-0.02), so **no resolved second split** exists there.

## 4. Where the theory is verified, and where it is not

Cached simulations (gamma=1.3, beta=1.6, 8 seeds, std=0.15 detection), predicted vs simulated height z (mean +- sd):
| e/b | split 1 pred | sim | split 2 pred | sim |
|---|---|---|---|---|
| 1.0 | -0.049 (x) | -0.067+-0.017 | at plane | -0.019+-0.006 (y) |
| 1.73 | -0.081 | -0.13+-0.03 | -0.081 | -0.13+-0.03 |
| 2.0 | -0.182 (y) | -0.237+-0.039 | -0.098 | -0.160+-0.041 |
| 2.5 | -0.420 (y) | -0.503+-0.063 | -0.139 | -0.239+-0.058 |
| 3.0 | -0.660 (y) | -0.746+-0.078 | -0.175 | -0.342+-0.075 |
In time (not height) the y-mode agrees within 7% (16.6 vs 17.6, 13.5 vs 14.4 time units, e=2.5, 3.0); the x-mode is 25-35% earlier in the sim at e=2.5-3 (22.9 vs 30.2, 21.2 vs 32.2), consistent with the post-split speed-up of section 3. Order, e-trend, y positions: correct.
Quantitative bias: predicted z1 is 0.05-0.09 shallower than simulated, z2 0.05-0.17 shallower (grows with e). The two-cluster
correction explains about half of the x shift (coherent -0.085 -> -0.14 vs sim -0.24 at e=2.5); the rest is **unexplained** [conj: the
apex group keeps growing after the split (f not constant) and cells shed in a continuing sequence, not one two-cluster event].

**The apparent "failure" at e<=1 is not a failure of the linear theory.** In the std metric the linear theory is right (fig `std_vs_z_theory_vs_sim.png`):
x-split at -0.05 vs -0.07 sim. The earlier "split at z=-0.9" came from the group-centroid criterion, which fires when std is only 0.03:
at that moment centroids of final-fate groups of 4-8 cells are 0.15 apart purely by selection on noise (`artifact.py`). Grouping by final
fate over-reports early splits.

**Hard validity limit: the height-dispersion mode.** lam_zz (cells spreading in height) is positive far from the targets and can be
much larger than lam_x,y (gamma=2, beta=1.6: lam_zz=182 vs 10.9 in N=1 units at the source). Define Gamma_zz(z1)=int lam_zz dt up to the
predicted split (column `Gamma_zz_at_split1`; validity ok <2, marginal 2-4, dominated >=4). Cases: gamma=1.3, beta=1.6: Gamma_zz~1-1.2, prediction
good. gamma=2, beta>=1.5 (Gamma_zz=7-8) or gamma=2.5, beta>=0.85: the simulated std_x crosses at z=-0.46...-1.7 although the linear
theory in *time* is within a factor 2 (sim t=3.7-4.2 vs 9-10); the cloud spreads in height first (std_z=0.39 at t=2, std_x=0.02), the
mean height stalls (-1.3 at t=10 vs coherent -0.23) and the coherent-path height is meaningless. In this regime predictions in
predictions.csv are tagged `height-dispersion-dominated`; use them only as lower bounds on the lateral amplification, not as locations.
This regime, not the lateral instability, controls the decision for gamma>~2 at beta>~1.5 [num; mechanism of the lag is a conjecture].

## 5. Not done / open
* A closed form for z2 (only the structure at gamma=2); f is an input, not predicted.
* The 3D coupling of the y-mode with the height mode near the plane (lam_yz) is ignored in the variance ODE (e_y direction used; the
  direction perpendicular to the velocity gives z1 within 0.01-0.02).
* The x-first (flat) case: second stage predicted stable/at-plane; not confirmed by a std-resolved simulation (centroid-based splits there are the selection artifact).
* Predictions use h=2.5; for |z1| within ~0.5 of h (e>=4, gamma>=2) the result is clipped by the source depth.

## Files
`th.py` (closed forms, coherent path), `predict.py` -> `predictions.csv` (96 rows: e x gamma x beta; columns x/y/z of both splits,
times, Gamma, Gamma_zz, validity, notes), `two_stage.py` (two-cluster stage), `figs.py` -> `fig1_growth_rates_and_order_map.png`,
`fig2_split_locations_side_view.png`, `fig3_split_heights_vs_e.png`, `fig4_gamma_beta_maps.png`; checks: `verify1.py`, `check_rules.py`,
`step2.py`, `curves.py`, `curves2.py`, `artifact.py`, `validate.py`, `early.py`, `time_check.py`; figures `std_vs_z_theory_vs_sim.png`,
`std_x_two_cluster.png`, `early_regime_check.png`.
