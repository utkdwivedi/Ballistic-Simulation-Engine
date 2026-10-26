# Decisions Log

Every non-obvious decision made during the refactor is recorded here,
with the rationale and the evidence. Decisions are dated and tagged with
the commit that implemented them.

---

## 2026-10-26 — Phase 1: Material Parameters & Citations

### D-001 — JC constitutive residual compares against σ_JC·(1−D), not σ_JC
**Decision:** `jc_constitutive_residual_np/torch` now returns
`sigma_pred − σ_JC(ε, ε̇, T)·(1−D)`.

**Rationale:** Definitive test on a simulator trajectory
(`tests/golden/RHA_steel_200.npz`): the stored `sigma_MPa` is the *raw*
Johnson-Cook stress. Proof:
- `stored == raw_JC·(1−D)` → **False**
- `stored == raw_JC` in undamaged regions → **True**
- The force implied by the velocity trajectory
  `F = m(V²−V'²)/(2Δx)` has `corr(F, σ_raw·perimeter·t/√3) = 1.0000`,
  `mean|ΔF| = 0.00 N`, while `corr(F, σ_eff·…) = 0.9539`,
  `mean|ΔF| = 24,434 N`.

The simulator therefore stores raw stress and applies damage once in the
ODEs. The PINN predicts the *effective* stress (the training target), so
the constitutive residual must compare it against σ_JC·(1−D).

**Evidence:** `physics_residuals.py` (C-01 fix). Before: mean|JC residual|
= 1068 MPa on a 1000-point RHA trajectory. After: **0.78 MPa**.
Momentum (15.2 N) and energy (0.0046 K/m) residuals were already correct
and are unchanged.

**Commit:** `chore/baseline` — C-01 (see `reports/IMPROVEMENT_LOG.md`).

---

### D-002 — Alumina failure_strain = 0.3, documented as UNVERIFIED
**Decision:** Keep `failure_strain = 0.3` for alumina but mark the source
`UNVERIFIED` and add a bounds check in `test_materials.py`.

**Rationale:** The audit flagged the old value (0.05) as unrealistic.
Alumina (Al₂O₃) is brittle: tensile failure strain ≈ 0.001–0.01,
compressive ≈ 0.01–0.05. The value 0.3 is a *shear-plugging* effective
strain, which is a different quantity from the tensile failure strain.
No published JC-in-shear parameter set for alumina could be verified
without internet access, so the value is marked `UNVERIFIED` rather than
presented as measured.

**Evidence:** `materials_config.json`, `tests/test_materials.py`.
Bounds check: `0.01 ≤ failure_strain ≤ 5.0` for all materials.

---

### D-003 — Cowper-Symonds canonical reference is Cowper & Symonds 1957
**Decision:** Add the 1957 *Journal of the Mechanics and Physics of Solids*
paper to `references.bib` as the canonical CS citation; keep the 1952
Brown University report (TR-28) as the earlier technical version.

**Rationale:** The 1952 report is the original derivation, but the
peer-reviewed 1957 journal article is the citable version. Both are
real, verified entries.

**Evidence:** `references.bib` — `cowper1957` added.

---

### D-004 — UHMWPE and copper parameter sources marked UNVERIFIED
**Decision:** The UHMWPE parameters cite Chocron et al. (2008) and
Russell et al. (2013), and the copper JC parameters cite no verified
source. Both are marked `UNVERIFIED` in `materials_config.json`.

**Rationale:** Without internet access the exact DOIs and fitted
parameter sets could not be confirmed. Per the "never fabricate"
directive, unverifiable citations are marked explicitly rather than
presented as verified.

**Evidence:** `materials_config.json` — `source: "UNVERIFIED"`,
`confidence: "low"`.

---

### D-005 — Backward-compatible `source`/`confidence` schema
**Decision:** Add `source` and `confidence` fields to
`materials_config.json` without removing any existing field. The
loader in `stage1_part1_johnson_cook.py` only requires
`{A, B, n, C, m, T_melt, rho, E, Cp}` and ignores extra keys, so old
loaders continue to work.

**Evidence:** `stage1_part1_johnson_cook.py:_load_materials()` —
`_REQUIRED_KEYS` unchanged; extra keys are passed through.