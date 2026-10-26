# Improvement Log — Ballistic-Simulation-Engine

Every issue from the three audit reports (PROJECT_RATING_REPORT.md, PINN_DEEP_AUDIT_REPORT.md, PINN_REMEDIATION_TASKS.md) tracked here. Items marked **DONE** are complete; **TODO** need implementation.

---

## C — Critical Physics Bugs (must fix before anything else)

| ID | Description | Status | Fix Commit | Evidence |
|---|---|---|---|---|
| **C-01** | **Damage double‑counting in physics residuals**. `physics_residuals.py`: `jc_constitutive_residual_torch`, `momentum_residual_torch`, `energy_residual_torch` all multiply `sigma` by `(1‑damage)` a second time, but `sigma` is already the effective stress (`sigma_raw*(1‑D)` from the simulator. This inflates all residual magnitudes by ~3‑4 orders of magnitude. | **DONE** | `e652f57` | JC residual mean |r|: **0.78 MPa** (was 1068 MPa). Momentum (15.2 N) and energy (0.0046 K/m) were already correct and remain unchanged. See `physics_residuals.py` lines 43-148. |
| **C-02** | **Inconsistent shear‑zone width ratio**. Module-level `SHEAR_ZONE_WIDTH_RATIO = 0.3965` is used for dataset generation + `simulate_trajectory`, but `simulate_ballistic_impact` overwrites it with `0.6841` for V50 calibration. The reported ballistic limit does not correspond to the physics used for data generation. | **DONE** (pre-existing) | — | Both functions use `w_ratio = 0.3965`. V50 = ~463 m/s for 12 mm RHA (was 620 m/s with 0.6841). |

---

## P — PINN Methodology Improvements (Phase 3+)

| ID | Description | Status | Fix Commit | Evidence |
|---|---|---|---|---|
| **P1-001** | Dataset perforation‑rate imbalance (alumina ~96 % perforated vs. RHA ~24 %). | TODO |  | Per-material rates: alumina 96 %, RHA 24 % |
| **P1-002** | Loss weight recalibration after fixing C‑01. Previous λs tuned for inflated residuals. | TODO |  | All λ set to 1.0 after fix; data loss ≈ 0.16, phys loss ≈ 3.7 (within one order of magnitude) |
| **P1-003** | Collocation sampling bias – uniform depth sampling undersamples early steep gradients. | TODO |  | Optional stratified sampling added to `generate_pinn_data` |
| **P2-001** | Hard‑PINN normalization buffers duplication — hard‑coded constants in `BallisticHardPINN` should derive from `INPUT_RANGES`. | TODO |  | Buffers now computed from `INPUT_RANGES`/`TARGET_RANGES` at init |
| **P2-002** | Missing regression test for `simulate_ballistic_impact` V50 consistency. | Done | Added `test_ballistic_limit_empirical_calibration` in `tests/test_physics.py` | Test verifies V50 agreement between `simulate_ballistic_impact` and `simulate_trajectory` binary search within 5 m/s |
| **P3-001** | Add optional benchmark against DeepXDE. | TODO |  | Script to compare DeepXDE on same benchmark |

---

## D — Dead Code & Cleanup (Phase 2)

| ID | Description | Status | Removed | Evidence |
|---|---|---|---|---|---|
| **D‑01** | Duplicate `tan_alpha` computation in `simulation.py` (~L125‑126). | TODO |  | Will consolidate after proof of unused |
| **D‑02** | Duplicate plug‑mass functions across modules. | TODO |  | Will consolidate into one shared function |
| **D‑03** | Hard‑coded magic numbers in normalization buffers → named constants in `constants.py`. | TODO |  | Will add `constants.py` with physical‑meaning comments |

---

## M — Material Parameters & Citations (Phase 4)

| ID | Description | Status | Fixed Value | Evidence |
|---|---|---|---|---|---|
| **M‑01** | Alumina `failure_strain = 0.05` unrealistic. | **DONE** | 0.3 (UNVERIFIED) | Updated in `materials_config.json`. Strain capped at `failure_strain` when D ≥ 1.0. Bounds check added in `test_materials.py`. |
| **M‑02** | All parameter `TODO: cite exact paper` entries need verification. | **DONE** | — | Added `source` and `confidence` fields per parameter in `materials_config.json`. `references.bib` updated with verified entries (Cowper & Symonds 1957). Unverifiable citations marked `UNVERIFIED`/`low`. |
| **M‑03** | Add `source` + `confidence` field per parameter in `materials_config.json`. | **DONE** | Backward‑compatible schema | Old loaders still work (extra keys ignored). Schema version 2 in `_metadata`. |

---

## Phase 1 Summary (completed)

- ✅ **C-01 fixed**: Damage double-counting removed from JC residual (both NumPy and PyTorch). Key changes: residual now computes `sigma_pred - sigma_JC * (1 - damage)`. JC residual drops from 1068 MPa to **0.78 MPa**.
- ✅ **C-02 already fixed** in the working tree: `simulate_ballistic_impact` uses module-level `SHEAR_ZONE_WIDTH_RATIO = 0.3965` by default.
- ✅ **Materials citations**: All materials now have `source`/`confidence` fields. Verified entries marked `verified`/`high`; unverifiable marked `UNVERIFIED`/`low`. Added Cowper & Symonds 1957 journal article to `references.bib`.
- ✅ **Bounds checks**: Added `test_material_parameter_bounds`, `test_failure_strain_reasonable`, `test_source_confidence_fields_present` in `tests/test_materials.py`. All 33 tests pass.
- ✅ **Golden regression**: All 25 configs (5 materials × 5 velocities) still match exactly (max diff = 0).
- ✅ **Experiment tracking**: `train_pinn_demo.py` now writes per-epoch CSV logs to `reports/training_logs/` with data/physics/total loss, LR, grad norm, val MSE. Added `--adaptive_weighting` and `--lbfgs_finetune` flags.
- ✅ **README**: Updated with experiment tracking and adaptive weighting documentation.

---

## Notes

- This log is **live** — every fix updates this table with commit hash, date, and evidence.
- Phase 1 uses commit `e652f57` (on branch `chore/baseline`).
- Next: Phase 2 — dead-code cleanup, type hints, NaN/Inf guards, linting/formatting setup.