# docs/known_limitations.md

## Known Limitations & Assumptions

- **Material‑Dependent Failure Strain**: The `failure_strain` parameter in `materials_config.json` caps strain at the material‑specific value when damage `D \ge 1.0`.
- **Geometric Strain Rate**: We use the simplified estimate `\dot{\epsilon} = V / (w \sqrt{3})` for the strain‑rate calculation. Future work will improve this for more accurate rate‑dependence.
- **Shear‑Zone Width Ratio**: The simulation‐internal value `SHEAR_ZONE_WIDTH_RATIO = 0.3965` is calibrated against the 12‑mm RHA `V_{50}` target. Using a different `w_ratio` (e.g., 0.6841) will shift ballistic limits—a caution for users porting across geometries.
- **Calibration**: A single‑point calibration (12‑mm RHA) was used. Extrapolation to other materials or plate thicknesses may require additional tuning.
- **Closed‑System**: The solver assumes no mass or energy flux out of the system; surface conditions are hard‑BC enforced.
