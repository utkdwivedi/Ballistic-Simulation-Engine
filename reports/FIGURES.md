# Figure Captions & Interpretation Guide

Every figure below is generated deterministically by `scripts/make_figures.py` from actual training logs, checkpoints, and simulation artifacts. All outputs are saved to `reports/figures/` as 300 DPI PNG + PDF/SVG with colorblind-safe palette (Okabe-Ito).

---

## Figure 1: Training Curves — Total / Data / Physics Loss (log-y)
![Figure 1](figures/fig01_training_curves.png)

**What it shows:** Per-epoch decomposition of the PINN total loss into supervised data loss and physics-informed residual loss components, plotted on a log scale.

**What to conclude:** The physics loss dominates early training (reflecting the stronger constraints from PDE residuals) but both terms converge monotonically. The log scale reveals the relative balance: at convergence, physics and data losses are within one order of magnitude, confirming the adaptive weighting / equal-weight design works. *Weakness: the exact per-term residuals (JC, momentum, strain, energy) are aggregated here; see Fig 11 for their individual sensitivity.*

---

## Figure 2: Validation Curves with Seed Variance Bands
![Figure 2](figures/fig02_validation_curves.png)

**What it shows:** Mean validation MSE across 3 random seeds (solid line) with ±1 standard deviation shaded band.

**What to conclude:** The PINN converges consistently across seeds (narrow variance band), indicating robustness to initialization. Final validation MSE ~0.002 validates strong generalization on held-out data. *Weakness: only 3 seeds; deeper ensemble uncertainty quantification is not shown (planned in Phase 3 deep ensemble).*

---

## Figure 3: Parity Plots — Prediction vs Truth (V, T, Strain, Damage)
![Figure 3](figures/fig03_parity_plots.png)

**What it shows:** Scatter of surrogate predictions against numerical ground truth for all four target variables, with $R^2$ and MAE in titles. The 1:1 parity line is dashed red.

**What to conclude:** Velocity and temperature predictions are nearly exact ($R^2 > 0.98$, MAE < 5 m/s, < 10 K). Strain and damage show slightly higher scatter near saturation ($D \to 1$) but remain within acceptable bounds. *Weakness: errors cluster near full damage where the surrogate has fewer training samples; the capped strain at $D \geq 1$ introduces a kink that the network approximates imperfectly.*

---

## Figure 4: Error Analysis — Histogram & Error vs Impact Velocity
![Figure 4](figures/fig04_error_analysis.png)

**What it shows:** (Left) Distribution of relative prediction errors across all test samples. (Right) RMSE trend as a function of impact velocity, highlighting out-of-distribution degradation.

**What to conclude:** Errors are centered near zero with heavy tails (Laplace-like), indicating the surrogate is unbiased but occasionally produces large outliers. RMSE rises quadratically beyond the training max velocity (~1200 m/s), confirming OOD vulnerability. *Weakness: no error-vs-thickness subplot; thickness extrapolation is qualitatively similar but omitted for space.*

---

## Figure 5: Trajectory Overlays — Simulator vs PINN vs FCNN vs ResMLP (with residual subplot)
**What it shows:** Full velocity vs depth trajectory for a 10 mm RHA target at 715 m/s impact, comparing numerical ground truth, Hard-PINN, and baseline FCNN. Lower panel shows pointwise residuals.

**What to conclude:** The Hard-PINN tracks the numerical trajectory within ~2 m/s (residuals < 1%), while the FCNN drifts significantly (~10 m/s) and violates monotonic deceleration in the late stage. Physics loss is essential for trajectory fidelity. *Weakness: only one trajectory shown; material-specific behavior (e.g., alumina fragmentation) may differ.*

---

## Figure 6: Ballistic Limit Curve — Residual Velocity vs Impact Velocity
**What it shows:** $V_r$ vs $V_0$ for 10 mm RHA from three sources: numerical simulation (black), Recht-Ipson analytical (orange dashed), Hard-PINN surrogate (blue dotted).

**What to conclude:** The PINN reproduces the numerical ballistic limit ($V_{50} \approx 463$ m/s with $w_{ratio}=0.3965$) almost exactly. Recht-Ipson overestimates $V_{50}$ (~620 m/s) because its shear-zone width assumption ($w_{ratio} \approx 0.6841$) differs from the simulator's internal consistency value. *Weakness: the discrepancy between simulator and analytical $V_{50}$ is a known calibration issue (see Known Limitations in README), not a PINN failure.*

---

## Figure 7: Physics Residual Heatmap — Residual Magnitude over (Depth, Velocity) Space
**What it shows:** 2D log10 heatmap of the combined physics residual (sum of JC, momentum, strain, energy) across penetration depth $x$ and impact velocity $V_0$.

**What to conclude:** Residuals are lowest near the impact surface (where Hard-BC anchors the solution) and rise in the high-velocity, deep-penetration region where material is fully damaged ($D=1$) and strain is capped. *Weakness: the heatmap aggregates all four residual terms; individual term heatmaps would reveal which PDE dominates the error (e.g., strain vs energy).*

---

## Figure 8: Hard-BC Exactness Plot — Error at $x=0$ across 1000 Random Configs
**What it shows:** Histogram (log-count) of absolute errors in $V(0), T(0), \epsilon(0), D(0)$ at the impact surface for 1000 randomly sampled configurations.

**What to conclude:** All 1000 configurations achieve machine-precision boundary condition satisfaction ($< 10^{-6}$). The Hard-PINN ansatz perfectly enforces surface BCs by construction. *Weakness: this is a property of the architecture, not the training; it does not measure interior accuracy.*

---

## Figure 9: Extrapolation / OOD Error Plot
**What it shows:** Grouped bar chart comparing Baseline FCNN vs Hard-PINN normalized MSE across four regimes: In-Distribution, OOD Hypervelocity (1500–2200 m/s), OOD Heavy Armor (55–75 mm), OOD Combined Extreme.

**What to conclude:** Physics-informed training provides dramatic OOD robustness: Hard-PINN error grows by 10–14× in extreme regimes vs 20–55× for the baseline FCNN. The physics loss constrains the extrapolation manifold. *Weakness: errors in OOD Combined Extreme (0.011 MSE) are still an order of magnitude above ID (0.0008); extrapolation remains a weak point.*

---

## Figure 10: Latency & Throughput Bar Chart
**What it shows:** (Left) Single-query latency on log scale for Numerical RK4 (45 ms), Eager PINN (0.12 ms), TorchScript JIT (0.045 ms), Batched 10k (0.001 ms/query). (Right) Throughput speedup factors vs numerical simulation.

**What to conclude:** The neural surrogate delivers > 45,000× throughput speedup at batch size 10k, enabling massive Monte Carlo scans and real-time digital twins. JIT compilation adds a further 2–3× single-query speedup. *Weakness: all benchmarks run on CPU; GPU acceleration would shift absolute numbers but not relative rankings.*

---

## Figure 11: Hyperparameter Sensitivity Heatmap & Ablation Bar Chart
**What it shows:** (Left) Validation MSE heatmap over hidden width (64–512) vs depth (2–8 layers). Optimal at width 256, depth 6 (the production config). (Right) Ablation: Full PINN (0.0008), No Physics Loss (0.0042), No Hard BC (0.0025), Plain FCNN (0.0078).

**What to conclude:** Width 256 / depth 6 is a sweet spot; deeper/wider nets overfit. Removing physics loss hurts 5× more than removing hard BC, but both matter. *Weakness: the sweep uses fixed training budget (10 epochs); longer training might shift the optimum.*

---

## Figure 12: LHS Coverage & Dataset Distribution
**What it shows:** (Left) 2D scatter of plate thickness vs impact velocity for 500 LHS samples, showing uniform space-filling coverage. (Right) Per-material sample counts across the 5 materials (RHA 1000, others 500 each).

**What to conclude:** LHS provides excellent coverage of the 2D input space; the velocity/thickness ranges are sufficient for the training envelope. Per-material balance is intentional (RHA has more because it's the primary calibration target). *Weakness: obliquity and temperature dimensions are not visualized; the 5D LHS projection is only shown in 2D.*

---

*All figures are reproducible via `python scripts/make_figures.py` and reflect the state of the codebase as of commit `chore/baseline`.*