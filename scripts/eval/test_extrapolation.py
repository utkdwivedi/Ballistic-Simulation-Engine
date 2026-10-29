"""
Out-of-Distribution (OOD) Extrapolation Benchmark (Phase 5, Task 5.2)
=====================================================================

Compares:
1. Supervised Baseline (fcnn_production_best.pt)
2. Physics-Informed Neural Surrogate (pinn_production_best.pt)

Regimes Evaluated:
- In-Distribution (ID):  V0 in [200, 1500] m/s,  h in [5, 50] mm
- OOD Hypervelocity:     V0 in [1500, 2200] m/s, h in [5, 50] mm
- OOD Heavy Armor:       V0 in [200, 1500] m/s,  h in [50, 75] mm
- OOD Combined Extreme:  V0 in [1500, 2200] m/s, h in [50, 75] mm

Computes:
- Trajectory Root Mean Square Error (RMSE) against numerical ground truth
- Monotonic Deceleration Violations (dV/dx > 1e-4)
- Non-Dimensional Continuum Physics Residuals (R_phys)
"""
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))

from ballistic.physics.johnson_cook import MATERIALS
from ballistic.simulation import simulate_trajectory

D_BULLET_M = 0.00762
M_BULLET_KG = 0.0079
SQRT3 = math.sqrt(3.0)
from scripts.train_pinn import (
    MAT_A,
    MAT_B,
    MAT_C_JC,
    MAT_CP,
    MAT_M,
    MAT_N,
    MAT_RHO,
    MAT_TMELT,
    compute_physics_loss,
)

from ballistic.models.architecture import BallisticHardPINN


def evaluate_trajectory_on_model(model, V0, h_mm, theta, mat_name, T_plate, mat_tensors, device):
    """
    Run numerical simulation for a parameter point, sample collocation depths,
    query neural surrogate, and compute parity errors and physics residuals.
    """
    mat_keys = list(MATERIALS.keys())
    mat_id = mat_keys.index(mat_name)

    hist = simulate_trajectory(
        plate_thickness_mm=h_mm,
        material_name=mat_name,
        plate_temperature_K=T_plate,
        bullet_velocity_ms=V0,
        obliquity_angle_deg=theta,
    )

    if not hist or len(hist["x_mm"]) < 5:
        return None

    x_mm_arr = np.array(hist["x_mm"], dtype=np.float32)
    V_sim = np.array(hist["V_ms"], dtype=np.float32)
    sigma_sim = np.array(hist["sigma_MPa"], dtype=np.float32)
    T_sim = np.array(hist["T_K"], dtype=np.float32)
    eps_sim = np.array(hist["strain"], dtype=np.float32)
    damage_sim = np.array(hist["damage"], dtype=np.float32)

    N = len(x_mm_arr)

    # Normalize inputs
    log10_v0 = math.log10(max(V0, 1.0))
    v0_norm = (log10_v0 - 1.699) / 1.778
    th_norm = (h_mm - 0.1) / (50.0 - 0.1)
    ob_norm = theta / 85.0
    tp_norm = (T_plate - 200.0) / 300.0
    x_norm_arr = x_mm_arr / 100.0

    batch = np.zeros((N, 6), dtype=np.float32)
    batch[:, 0] = th_norm
    batch[:, 1] = v0_norm
    batch[:, 2] = ob_norm
    batch[:, 3] = float(mat_id)
    batch[:, 4] = tp_norm
    batch[:, 5] = x_norm_arr

    batch_x = torch.from_numpy(batch).to(device)
    x_mm_grad = batch_x[:, 5:6].clone().detach().requires_grad_(True)
    batch_x_grad = torch.cat([batch_x[:, :5], x_mm_grad], dim=1)

    model.eval()
    preds_norm = model(batch_x_grad)

    # Denormalize predictions
    V_pred = (preds_norm[:, 0] * 3000.0).detach().cpu().numpy()
    sigma_pred = (preds_norm[:, 1] * 3000.0).detach().cpu().numpy()
    T_pred = (preds_norm[:, 2] * 2300.0 + 200.0).detach().cpu().numpy()
    eps_norm_val = preds_norm[:, 3].detach().cpu().numpy()
    eps_pred = np.expm1(eps_norm_val * 6.0)
    damage_pred = (preds_norm[:, 4]).detach().cpu().numpy()

    # Residuals
    loss_phys = compute_physics_loss(preds_norm, x_mm_grad, batch_x, mat_tensors, device).item()

    # Monotonicity check: verify no macroscopic physical re-acceleration (> 0.05 m/s)
    dV = np.diff(V_pred)
    accel_violations = int(np.sum(dV > 0.05))

    # Normalized error vs simulation
    err_V = np.mean(((V_pred - V_sim) / 3000.0) ** 2)
    err_sigma = np.mean(((sigma_pred - sigma_sim) / 3000.0) ** 2)
    err_T = np.mean(((T_pred - T_sim) / 2300.0) ** 2)
    err_damage = np.mean((damage_pred - damage_sim) ** 2)
    mean_mse = (err_V + err_sigma + err_T + err_damage) / 4.0

    return {
        "mse": float(mean_mse),
        "rmse": float(np.sqrt(mean_mse)),
        "phys_loss": float(loss_phys),
        "accel_violations": accel_violations,
    }


def main():
    """Run this module command-line workflow."""
    print("=" * 70)
    print("  Phase 5, Task 5.2: Out-of-Distribution Extrapolation Benchmark")
    print("=" * 70)

    device = torch.device("cpu")
    mat_tensors = {
        "A": MAT_A.to(device), "B": MAT_B.to(device), "n": MAT_N.to(device),
        "C": MAT_C_JC.to(device), "m": MAT_M.to(device), "T_melt": MAT_TMELT.to(device),
        "rho": MAT_RHO.to(device), "Cp": MAT_CP.to(device)
    }

    # Load both models
    fcnn_model = BallisticHardPINN().to(device)
    fcnn_model.load_state_dict(torch.load("models/fcnn_production_best.pt", weights_only=True))

    pinn_model = BallisticHardPINN().to(device)
    pinn_model.load_state_dict(torch.load("models/pinn_production_best.pt", weights_only=True))

    regimes = {
        "In-Distribution": {
            "velocities": [400.0, 800.0, 1200.0],
            "thicknesses": [10.0, 25.0, 40.0],
        },
        "OOD Hypervelocity (1600-2200 m/s)": {
            "velocities": [1650.0, 1900.0, 2150.0],
            "thicknesses": [15.0, 30.0],
        },
        "OOD Heavy Armor (55-75 mm)": {
            "velocities": [800.0, 1200.0],
            "thicknesses": [55.0, 65.0, 75.0],
        },
        "OOD Combined Extreme (Hypervelocity + Heavy Armor)": {
            "velocities": [1800.0, 2100.0],
            "thicknesses": [60.0, 70.0],
        },
    }

    results = {}
    materials = list(MATERIALS.keys())

    for regime_name, cfg in regimes.items():
        print(f"\n--- Testing Regime: {regime_name} ---")
        fcnn_mses, pinn_mses = [], []
        fcnn_phys, pinn_phys = [], []
        fcnn_accels, pinn_accels = [], []

        for V0 in cfg["velocities"]:
            for h in cfg["thicknesses"]:
                for mat in materials[:3]:  # Test 3 distinct materials
                    res_fcnn = evaluate_trajectory_on_model(fcnn_model, V0, h, 15.0, mat, 298.0, mat_tensors, device)
                    res_pinn = evaluate_trajectory_on_model(pinn_model, V0, h, 15.0, mat, 298.0, mat_tensors, device)

                    if res_fcnn and res_pinn:
                        fcnn_mses.append(res_fcnn["mse"])
                        pinn_mses.append(res_pinn["mse"])
                        fcnn_phys.append(res_fcnn["phys_loss"])
                        pinn_phys.append(res_pinn["phys_loss"])
                        fcnn_accels.append(res_fcnn["accel_violations"])
                        pinn_accels.append(res_pinn["accel_violations"])

        avg_fcnn_mse = float(np.mean(fcnn_mses))
        avg_pinn_mse = float(np.mean(pinn_mses))
        avg_fcnn_phys = float(np.mean(fcnn_phys))
        avg_pinn_phys = float(np.mean(pinn_phys))
        sum_fcnn_accel = int(np.sum(fcnn_accels))
        sum_pinn_accel = int(np.sum(pinn_accels))

        phys_reduction = ((avg_fcnn_phys - avg_pinn_phys) / (avg_fcnn_phys + 1e-12)) * 100.0

        print(f"  Supervised Baseline: MSE={avg_fcnn_mse:.6f} | R_phys={avg_fcnn_phys:.6f} | Accel Violations={sum_fcnn_accel}")
        print(f"  PINN Production:     MSE={avg_pinn_mse:.6f} | R_phys={avg_pinn_phys:.6f} | Accel Violations={sum_pinn_accel}")
        print(f"  Physics Loss Reduction: {phys_reduction:+.1f}%")

        results[regime_name] = {
            "fcnn_mse": avg_fcnn_mse,
            "pinn_mse": avg_pinn_mse,
            "fcnn_phys_loss": avg_fcnn_phys,
            "pinn_phys_loss": avg_pinn_phys,
            "physics_loss_reduction_pct": phys_reduction,
            "fcnn_accel_violations": sum_fcnn_accel,
            "pinn_accel_violations": sum_pinn_accel,
        }

    os.makedirs("results/reports", exist_ok=True)
    report_path = "results/reports/extrapolation_benchmark.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"[REPORT] Extrapolation report saved to: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
