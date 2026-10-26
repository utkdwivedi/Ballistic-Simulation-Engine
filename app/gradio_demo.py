# Minimal Gradio App – ShowSimulator vs PINN

```python
# app/gradio_demo.py
import gradio as gr
import json

from simulation import simulate_trajectory
from models.architecture import BallisticHardPINN
import torch
import numpy as np

MATERIALS = json.load(open("materials_config.json"))


def simulate_and_pinn(plate_thickness_mm, material_name, bullet_velocity_ms):
    # Simulator
    traj = simulate_trajectory(
        plate_thickness_mm=plate_thickness_mm,
        material_name=material_name,
        bullet_velocity_ms=bullet_velocity_ms,
        bullet_mass_g=7.9,
        bullet_diameter_mm=7.62,
    )
    sim_depth = traj['x_mm']
    sim_vel = traj['V_ms']

    # PINN
    # Build a tensor grid over depth
    depths = np.linspace(0, plate_thickness_mm, 200)
    inp = {
        "thickness_mm": plate_thickness_mm,
        "velocity_ms": bullet_velocity_ms,
        "obliquity_deg": 0.0,
        "material_id": 0,
        "plate_temperature_K": 298.0,
        "x_mm": depths,
    }
    inp_arr = np.column_stack([inp[k] for k in BallisticPINNDataset.INPUT_COLS])
    tensor = torch.from_numpy(inp_arr.astype("float32"))
    model = BallisticHardPINN()
    model.load_state_dict(torch.load("models/hardpinn.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        out = model(tensor)
    pinn_vel = out[:, 0].numpy()

    return sim_depth, sim_vel, depths, pinn_vel

demo = gr.Interface(
    fn=simulate_and_pinn,
    inputs=[
        gr.Number(value=10.0, label="Plate Thickness (mm)"),
        gr.Dropdown(list(MATERIALS.keys()), label="Material"),
        gr.Number(value=800.0, label="Impact Velocity (m/s)")
    ],
    outputs=[
        gr.Group([
            gr.LineChart(label="Simulator Trajectory", height=300),
            gr.LineChart(label="PINN Trajectory", height=300)
        ])
    ],
    title="Ballistic Penetration – Simulator vs PINN",
    description="Adjust the plate thickness, material, and impact velocity to see the trajectories side by side.",
)

demo.launch()
```

Install with:

```bash
pip install gradio
```
