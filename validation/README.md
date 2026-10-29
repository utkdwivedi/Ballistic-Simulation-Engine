# Experimental validation data

Place cited experimental datasets as CSV files directly in this directory. The validation script reads each `*.csv` file here.

Required columns:

| Column | Meaning |
| --- | --- |
| `material` | Material key from `materials_config.json` |
| `thickness_mm` | Target plate thickness in millimetres |
| `projectile_mass_g` | Projectile mass in grams |
| `projectile_diameter_mm` | Projectile diameter in millimetres |
| `projectile_shape` | Projectile shape |
| `v_impact_ms` | Measured impact velocity in metres per second |
| `v_residual_ms` | Measured residual velocity in metres per second |
| `source` | Citation or URL for the measurements |

Run validation from the repository root with:

```bash
python scripts/validate_against_experiments.py
```

Do not commit data unless its source permits redistribution.
