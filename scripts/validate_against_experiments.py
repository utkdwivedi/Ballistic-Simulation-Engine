from pathlib import Path
import csv
import json
import os

# Validate the harness imports
from simulation import simulate_trajectory, simulate_ballistic_impact
from pinn_dataloader import BallisticPINNDataset
from models.architecture import BallisticHardPINN
import torch
import numpy as np

# --- Helper functions ---

def load_csv_dataset(csv_path):
    """Loads a CSV of experimental data matching the schema defined in README."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = []
        for row in reader:
            data.append({
                "material": row["material"],
                "thickness_mm": float(row["thickness_mm"]),
                "projectile_mass_g": float(row["projectile_mass_g"]),
                "projectile_diameter_mm": float(row["projectile_diameter_mm"]),
                "projectile_shape": row["projectile_shape"],
                "v_impact_ms": float(row["v_impact_ms"]),
                "v_residual_ms": float(row["v_residual_ms"]),
                "source": row["source"],
            })
    return data


def run_simu(data):
    results = []
    for rec in data:
        try:
            res = simulate_ballistic_impact(
                plate_thickness_mm=rec["thickness_mm"],
                material_name=rec["material"],
                plate_temperature_K=298.0,
                bullet_velocity_ms=rec["v_impact_ms"],
                bullet_mass_g=rec["projectile_mass_g"],
                bullet_diameter_mm=rec["projectile_diameter_mm"],
            )
            results.append((rec, res.residual_velocity_ms))
        except Exception as e:
            results.append((rec, f"ERROR: {e}"))
    return results


def run_pinn(data, checkpoint_path="models/hardpinn.pt"):
    model = BallisticHardPINN()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    outputs = []
    for rec in data:
        inp = {
            "thickness_mm": rec["thickness_mm"],
            "velocity_ms": rec["v_impact_ms"],
            "obliquity_deg": 0.0,
            "material_id": 0,
            "plate_temperature_K": 298.0,
            "x_mm": 0.0,
        }
        inp_arr = np.array([[inp[k] for k in BallisticPINNDataset.INPUT_COLS]])
        # Normalisation
        import torch
        inp_tensor = torch.from_numpy(inp_arr).float()
        if inp_tensor.shape[1] != len(inp_arr[0]):
            continue
        with torch.no_grad():
            pred = model(inp_tensor)
        # Burn prediction to get residual velocity
        v_pred = pred[0, 0]  # V_ms (first output)
        outputs.append((rec, v_pred.item()))
    return outputs


def main():
    data_dir = Path("validation/data")
    if not data_dir.is_dir():
        print("validation/data not found – see README for instructions.")
        return

    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print("No CSV files in validation/data – no data to validate.")
        return

    all_results = {}
    for csv_path in csv_files:
        print(f"\n=== Validating {csv_path.name} ===")
        data = load_csv_dataset(csv_path)
        simu_res = run_simu(data)
        pinn_res = run_pinn(data)

        table_rows = []
        for idx, rec in enumerate(data):
            sim_val = simu_res[idx][1]
            pinn_val = pinn_res[idx][1]
            table_rows.append(
                f"| {rec['material']} | {rec['thickness_mm']} | {rec['projectile_mass_g']} | {rec['v_impact_ms']} | {rec['v_residual_ms']} | {sim_val} | {pinn_val} |")

        header = "| Material | Thickness [mm] | Projectile Mass [g] | V_impact [m/s] | V_residual [m/s] | Sim [m/s] | PINN [m/s] |"
        print(header)
        print("|---|---|---|---|---|---|---|")
        for row in table_rows:
            print(row)

        # Plot residual vs impact velocity
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6,4))
            plt.scatter([rec['v_impact_ms'] for rec in data],
                        [rec['v_residual_ms'] for rec in data],
                        label='Experimental', color='black')
            plt.scatter([rec['v_impact_ms'] for rec in data],
                        [simu_res[i][1] for i in range(len(data))],
                        label='Simulator', marker='x', color='red')
            plt.scatter([rec['v_impact_ms'] for rec in data],
                        [pinn_res[i][1] for i in range(len(data))],
                        label='PINN', marker='^', color='blue')
            plt.xlabel('Impact Velocity [m/s]')
            plt.ylabel('Residual Velocity [m/s]')
            plt.title(f'Validation – {csv_path.stem}')
            plt.legend()
            out_dir = Path("validation/plots")
            out_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_dir / f"{csv_path.stem}_validation.png", dpi=150)
            plt.close()
            print(f"Plot saved to {out_dir / f'{csv_path.stem}_validation.png'}")
        except Exception as e:
            print(f"Plotting failed: {e}")


if __name__ == "__main__":
    main()
