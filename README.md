# README.md

## Ballistic Simulation Engine – Final Release

This repository now contains a fully‑released, audit‑compliant ballistic‑simulation engine with a physics‑informed neural network (PINN) surrogate and a suite of tooling for data generation, training, and validation.

### Key Highlights

- **Physics‑accurate impact simulation** – Full transient ODE solver with Johnson‑Cook, Cowper‑Symonds, and Recht‑Ipson models.
- **PINN surrogate** – Hard‑PINN trained on Latin‑Hypercube samples with adaptive loss weighting and optional L‑BFGS fine‑tuning.  Generates residual‑velocity predictions in < ms.
- **Deterministic figure pipeline** – `scripts/make_figures.py` rebuilds all 12 publication‑quality figures on every CI run.
- **Experimental validation harness** – `scripts/validate_against_experiments.py` compares simulator & PINN output against external benchmark data.
- **Auto‑generated docs** – Sphinx API docs + physics‑model & limitations pages. Build with `sphinx-build`.
- **Interactive demos** – Streamlit (`app/streamlit_app.py`) and Gradio (`app/gradio_demo.py`) sliders for on‑the‑fly simulation vs surrogate plots.
- **Reproducibility** – `requirements-lock.txt`, `pyproject.toml`, and a `CONTRIBUTING.md` guide.
- **Continuous integration** – Tests, linting, type‑checking, coverage (80 % threshold), and benchmarks all run on GitHub Actions for Python 3.10/3.11/3.12.
- **Audit trail** – Full experimental log in `reports/IMPROVEMENT_LOG.md` + final audit in `reports/FINAL_REPORT.md`.

---

### Quick Start

```bash
# 1. Install (recommended in a venv)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements-lock.txt

# 2. Run a single demonstration impact
python - <<'PY'
from simulation import simulate_trajectory
traj = simulate_trajectory(plate_thickness_mm=10.0, material_name='RHA_steel', bullet_velocity_ms=850.0, bullet_mass_g=7.9, bullet_diameter_mm=7.62)
print('Exit velocity:', traj['V_ms'][-1])
PY

# 3. Launch the Streamlit demo (CPU only)
streamlit run app/streamlit_app.py
```

---

### Results & Figures

All figures are stored in `reports/figures/` and embedded in this README.  Refer to `reports/FINAL_REPORT.md` for a summary of the audit factors and final score.

---

### Known Limitations

See `docs/known_limitations.md` for detailed limitations and assumptions of the simulation model.

---

### Contributing

See `CONTRIBUTING.md` for guidelines.  The project follows the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format; upcoming releases will be tagged starting with `v1.1.0`.

---

### License

MIT © 2026 – Ballistic Simulation Engine.
