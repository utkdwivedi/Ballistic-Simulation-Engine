# Phase 0 Recon Summary

## Baseline Results
- Full test suite: 30/30 pass (baseline green)
- Golden regression snapshots: 25 configs × 5 materials in `tests/golden/*.npz`
- Improvement log started at `reports/IMPROVEMENT_LOG.md`

## Key Baseline Metrics (with bugs present)
- JC constitutive residual (numpy, on simulator trajectory): mean |r| ≈ 0.78 MPa
- Momentum ODE residual: mean |r| ≈ 15 N  
- Energy (adiabatic heating) residual: mean |r| ≈ 0 K/m
- Combined PINN physics loss: L_total ≈ 7352 (dominated by inflated terms)
- V50 inconsistency: `simulate_ballistic_impact` returns ≈ 620 m/s with w_ratio=0.6841, but data generation uses w_ratio=0.3965; trajectory-based binary search yields no penetration even at 2000 m/s

## Issues Identified from Three Audits
| Priority | ID | Issue |
|---|---|---|
| **Critical** | C-01 | Damage double-counting in `physics_residuals.py` — all three residual functions apply `(1‑damage)` a second time on already-effective stress |
| **Critical** | C-02 | `simulate_ballistic_impact` overwrites `SHEAR_ZONE_WIDTH_RATIO` with 0.6841 for V50 calibration, while dataset generation uses 0.3965 |
| High | C-03 | Unrealistic failure strains (alumina 0.05) → strain up to 18× failure_strain in dataset |
| Medium | C-04 | Momentum residual sign convention (secondary after C-01 fix) |
| Medium | C-05 | Missing damage clipping test (no healing test for edge cases) |
| P1 | P1-001 | Dataset perforation-rate imbalance across materials |
| P1 | P1-002 | Loss weight recalibration needed after C-01 fix |
| P1 | P1-003 | Collocation sampling bias — uniform depth undersamples early gradients |
| P2 | P2-001 | Hard-PINN normalization buffers duplicated — should derive from INPUT_RANGES |
| P2 | P2-002 | Missing V50 consistency regression test |
| P3 | P3-01 | TODO: cite exact paper for all material parameters |

## Decisions Logged
- Phase 0 uses the module-level `SHEAR_ZONE_WIDTH_RATIO = 0.3965` as the *canonical* value (used for data generation). Any calibration to 0.6841 will be exposed as an explicit argument, not an overwrite.
- Golden regression tolerance: rtol 1e-9 for pure refactor changes; looser for physics with inherent numerical error.

**Next: Phase 1 — fix C-01 and C-02, write failing tests, verify residuals → ~0 on simulator trajectories.**