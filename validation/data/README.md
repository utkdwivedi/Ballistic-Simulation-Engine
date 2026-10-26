# Experimental Validation Data

This subdirectory is reserved for experimental datasets that will be used to validate the ballistic simulator and the PINN surrogate.  Each experimental record should be written as a single CSV row following the schema below.

| Column | Type | Description |
|--------|------|-------------|
| material | string | Material name as defined in `materials_config.json` |
| thickness_mm | float | Plate thickness in millimetres |
| projectile_mass_g | float | Projectile mass in grams |
| projectile_diameter_mm | float | Projectile diameter in millimetres |
| projectile_shape | string | Projectile shape descriptor (e.g., `cylinder`, `sphere`). |
| v_impact_ms | float | Impact velocity of the projectile in m/s |
| v_residual_ms | float | Measured residual velocity after penetration in m/s |
| source | string | Citation or link to the source of the data |

### Adding new data

1. Create a new CSV file inside `validation/data/` named `<experiment_name>.csv`.
2. Populate it with rows that follow the table above.  A header row is **required**.
3. Commit and push the new file; the validation harness will automatically discover and validate it.

Existing datasets from literature (e.g., RHA tests from Børvik et al. 2003) will be added by external contributors.

If you have data available, please upload the CSVs following this format and I will run the harness.
