# APP – Streamlit Demo App

This directory contains a minimal Streamlit application that lets a user
interactively compare the simulator and the PINN surrogate on a single
trajectory.

```python
# app/streamlit_app.py
import streamlit as st
import json

# load the PINN model (adjust path as needed)
from models.architecture import BallisticHardPINN
import torch

# simulator import
from simulation import simulate_trajectory

# Helper to generate a trajectory using the PINN
@st.cache_data
def pinn_velocity_through_depth(
    material: str,
    thickness: float,
    velocity: float,
    num_points: int = 200,
    checkpoint: str = "models/hardpinn.pt",
):
    # build a simple dataset for the desired trajectory points
    import numpy as np
    inp = {
        "thickness_mm": thickness,
        "velocity_ms": velocity,
        "obliquity_deg": 0.0,
        "material_id": 0,
        "plate_temperature_K": 298.0,
        "x_mm": np.linspace(0, thickness, num_points),
    }
    inp_arr = np.column_stack(
        [inp[k] for k in BallisticPINNDataset.INPUT_COLS]
    )
    tensor = torch.from_numpy(inp_arr.astype("float32"))
    model = BallisticHardPINN()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        out = model(tensor)
    v_pred = out[:, 0].numpy()  # V_ms
    return inp["x_mm"], v_pred

st.title("Ballistic Penetration Demo – Simulator vs. PINN")

col1, col2 = st.columns(2)
with col1:
    material = st.selectbox("Material", list(MATERIALS.keys()))
    thickness = st.number_input("Thickness (mm)", value=10.0, min_value=1.0, max_value=50.0)
    velocity = st.number_input("Impact Velocity (m/s)", value=800.0, min_value=200.0, max_value=1500.0)
    run = st.button("Run" )
if run:
    # simulate
    traj = simulate_trajectory(
        plate_thickness_mm=thickness,
        material_name=material,
        bullet_velocity_ms=velocity,
        bullet_mass_g=7.9,
        bullet_diameter_mm=7.62,
    )
    st.subheader("Simulator Trajectory")
    st.line_chart({"depth": traj["x_mm"], "velocity": traj["V_ms"]})

    # PINN trajectory
    depth, v_pred = pinn_velocity_through_depth(material, thickness, velocity)
    st.subheader("Hard-PINN Trajectory")
    st.line_chart({"depth": depth, "velocity": v_pred})
    st.markdown("Both curves plotted on the same axes (x‑axis is depth).")
```

To run it locally:

```bash
pip install streamlit
streamlit run app/streamlit_app.py
```
