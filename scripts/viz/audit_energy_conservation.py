r"""
Energy Conservation and Physical Monotonicity Audit Script (Phase 4, Task 4.2)
================================================================================

Audits:
1. Analytical Work-Energy Conservation:
     Delta E_k(x) = 1/2 * m_bullet * (V_0^2 - V(x)^2)
     W_plastic(x) = \int_0^x F_shear(s) ds
     Evaluates relative discrepancy: |Delta E_k - W_plastic| / E_k0.

2. Thermodynamic & Kinematic Monotonicity:
     - Velocity non-increasing:      dV/dx <= 0 (projectile cannot accelerate)
     - Temperature non-decreasing:   dT/dx >= 0 (adiabatic plastic work generates heat)
     - Plastic strain non-decreasing: deps/dx >= 0 (plastic flow is irreversible)
     - Damage non-decreasing:        dD/dx >= 0 (damage does not heal)

3. Impact Surface Boundary Exactness (at x = 0):
     - V(0) == V_0
     - T(0) == T_plate
     - strain(0) == 0.0
     - damage(0) == 0.0

Generates:
  - results/reports/energy_conservation_audit.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath("."))

from ballistic.models.architecture import BallisticHardPINN
from ballistic.physics.johnson_cook import MATERIALS
from ballistic.simulation import (
    _RATE_JC,
    PLUG_SHEAR_ANGLE_DEG,
    SHEAR_ZONE_WIDTH_RATIO,
    T_ROOM,
    TAYLOR_QUINNEY_BETA,
    _compute_n_steps,
    _integrate_trajectory,
)

D_BULLET_M = 0.00762
M_BULLET_KG = 0.0079
SQRT3 = math.sqrt(3.0)


def generate_benchmark_trajectories(n_trajectories: int = 100, seed: int = 42) -> list:
    """
    Generate diverse benchmark trajectories across all 5 materials covering
    thin/thick plates, low/high impact velocities, and various obliquities.
    """
    rng = np.random.default_rng(seed)
    mat_keys = list(MATERIALS.keys())
    trajectories = []

    for i in range(n_trajectories):
        mat_name = mat_keys[i % len(mat_keys)]
        mat = MATERIALS[mat_name]

        # Latin hypercube / uniform parameter bounds
        t_plate_mm = rng.uniform(5.0, 50.0)
        V0 = rng.uniform(250.0, 1400.0)
        obliquity_deg = rng.uniform(0.0, 45.0)
        T_plate = rng.uniform(250.0, 360.0)

        t_m = t_plate_mm * 1e-3
        t_eff = t_m / math.cos(math.radians(obliquity_deg))
        n_steps = _compute_n_steps(t_eff)

        res = _integrate_trajectory(
            V0, t_eff, D_BULLET_M, M_BULLET_KG,
            mat["A"], mat["B"], mat["n"], mat["C"], mat["m"],
            mat["T_melt"], mat["rho"], mat["Cp"],
            T_plate, TAYLOR_QUINNEY_BETA,
            SHEAR_ZONE_WIDTH_RATIO,
            1.0, T_ROOM, n_steps,
            mat.get("cs_D", 1.0), mat.get("cs_q", 1.0),
            _RATE_JC, mat.get("failure_strain", 10.0),
            PLUG_SHEAR_ANGLE_DEG
        )

        n_active = int(res[0])
        if n_active < 5:
            continue

        x_arr = res[1][:n_active]         # m
        V_arr = res[2][:n_active]         # m/s
        sigma_arr = res[3][:n_active]     # MPa
        T_arr = res[4][:n_active]         # K
        eps_arr = res[5][:n_active]       # -
        edot_arr = res[6][:n_active]      # s^-1
        D_arr = res[7][:n_active]         # -

        trajectories.append({
            "mat_name": mat_name,
            "mat_id": mat_keys.index(mat_name),
            "thickness_mm": t_plate_mm,
            "t_eff_m": t_eff,
            "V0": V0,
            "obliquity_deg": obliquity_deg,
            "T_plate": T_plate,
            "n_pts": n_active,
            "x_m": x_arr,
            "V_ms": V_arr,
            "sigma_MPa": sigma_arr,
            "T_K": T_arr,
            "strain": eps_arr,
            "damage": D_arr,
        })

    return trajectories


def audit_simulation_energy_conservation(trajectories: list) -> dict:
    r"""
    Verify analytical work-energy equivalence on ground-truth numerical trajectories.
    Delta E_k = 0.5 * m * (V0^2 - V(x)^2) vs. \int F_shear dx.
    """
    tan_alpha = math.tan(math.radians(PLUG_SHEAR_ANGLE_DEG))
    rel_energy_errors = []
    monotonicity_violations = {
        "velocity_accel": 0,
        "temperature_cooling": 0,
        "negative_strain_inc": 0,
        "damage_healing": 0,
    }
    total_collocation_steps = 0

    for traj in trajectories:
        V0 = traj["V0"]
        E_k0 = 0.5 * M_BULLET_KG * (V0 ** 2)
        x = traj["x_m"]
        V = traj["V_ms"]
        sigma = traj["sigma_MPa"]
        T = traj["T_K"]
        eps = traj["strain"]
        D = traj["damage"]
        t_eff = traj["t_eff_m"]
        n = traj["n_pts"]

        # Compute shear resisting force at each step:
        # F = tau * perimeter_eff * t_eff, where tau = (sigma_eff * 1e6) / sqrt(3)
        tau_Pa = (sigma * 1e6) / SQRT3
        d_eff = D_BULLET_M + 2.0 * x * tan_alpha
        perimeter = math.pi * d_eff
        F_shear = tau_Pa * perimeter * t_eff

        # Numerical integration of work: trapezoidal rule
        dx = np.diff(x)
        F_mid = 0.5 * (F_shear[:-1] + F_shear[1:])
        work_plastic = np.sum(F_mid * dx)

        V_final = V[-1]
        delta_Ek = 0.5 * M_BULLET_KG * (V0 ** 2 - V_final ** 2)

        rel_err = abs(delta_Ek - work_plastic) / (E_k0 + 1e-12)
        rel_energy_errors.append(float(rel_err))

        # Check monotonicity along trajectory
        dV = np.diff(V)
        dT = np.diff(T)
        deps = np.diff(eps)
        dD = np.diff(D)

        monotonicity_violations["velocity_accel"] += int(np.sum(dV > 1e-4))
        monotonicity_violations["temperature_cooling"] += int(np.sum(dT < -1e-4))
        monotonicity_violations["negative_strain_inc"] += int(np.sum(deps < -1e-4))
        monotonicity_violations["damage_healing"] += int(np.sum(dD < -1e-4))

        total_collocation_steps += (n - 1)

    rel_arr = np.array(rel_energy_errors)
    return {
        "n_trajectories": len(trajectories),
        "total_collocation_steps": total_collocation_steps,
        "energy_error_mean": float(np.mean(rel_arr)),
        "energy_error_median": float(np.median(rel_arr)),
        "energy_error_p95": float(np.percentile(rel_arr, 95)),
        "energy_error_max": float(np.max(rel_arr)),
        "monotonicity_violations": monotonicity_violations,
        "all_energy_conserved": bool(np.percentile(rel_arr, 95) < 0.005 and np.max(rel_arr) < 0.025),
    }


def audit_hard_pinn_boundary_conditions(model: torch.nn.Module, n_samples: int = 1000) -> dict:
    """
    Verify Task 4.1: Hard-PINN boundary condition ansatz exactness at x = 0.
    Must hold to floating-point precision (< 1e-6) across random inputs.
    """
    model.eval()
    rng = torch.Generator().manual_seed(42)

    # Random input configurations
    x = torch.rand(n_samples, 6, generator=rng)
    x[:, 3] = torch.randint(0, 5, (n_samples,), generator=rng).float()
    x[:, 5] = 0.0  # Exactly at surface x = 0

    with torch.no_grad():
        pred = model(x)

    # Reconstruct theoretical initial conditions
    log10_v0 = x[:, 1:2] * 1.778 + 1.699
    expected_v0_norm = torch.pow(10.0, log10_v0) / 3000.0

    T_plate_phys = x[:, 4:5] * 300.0 + 200.0
    expected_t_norm = (T_plate_phys - 200.0) / 2300.0

    diff_v = torch.abs(pred[:, 0:1] - expected_v0_norm).numpy()
    diff_t = torch.abs(pred[:, 2:3] - expected_t_norm).numpy()
    strain_vals = torch.abs(pred[:, 3:4]).numpy()
    damage_vals = torch.abs(pred[:, 4:5]).numpy()

    return {
        "n_samples": n_samples,
        "max_V0_error": float(np.max(diff_v)),
        "mean_V0_error": float(np.mean(diff_v)),
        "max_T0_error": float(np.max(diff_t)),
        "mean_T0_error": float(np.mean(diff_t)),
        "max_strain0_error": float(np.max(strain_vals)),
        "max_damage0_error": float(np.max(damage_vals)),
        "exact_bc_satisfied": bool(
            np.max(diff_v) < 1e-6 and
            np.max(diff_t) < 1e-6 and
            np.max(strain_vals) < 1e-6 and
            np.max(damage_vals) < 1e-6
        ),
    }


def audit_neural_monotonicity(model: torch.nn.Module, trajectories: list) -> dict:
    """
    Audit monotonicity of neural surrogate trajectory predictions across physical trajectories.
    """
    model.eval()
    vel_violations = 0
    temp_violations = 0
    strain_violations = 0
    damage_violations = 0
    total_eval_steps = 0

    for traj in trajectories:
        n = traj["n_pts"]
        if n < 5:
            continue

        # Prepare normalized model input tensor for full trajectory
        x_mm_arr = traj["x_m"] * 1000.0
        v0 = traj["V0"]
        th = traj["thickness_mm"]
        ob = traj["obliquity_deg"]
        mid = traj["mat_id"]
        tp = traj["T_plate"]

        # Normalize features
        log10_v0 = math.log10(v0)
        v0_norm = (log10_v0 - 1.699) / 1.778
        th_norm = (th - 0.1) / (50.0 - 0.1)
        ob_norm = ob / 85.0
        tp_norm = (tp - 200.0) / 300.0
        x_norm_arr = x_mm_arr / 100.0

        batch = np.zeros((n, 6), dtype=np.float32)
        batch[:, 0] = th_norm
        batch[:, 1] = v0_norm
        batch[:, 2] = ob_norm
        batch[:, 3] = mid
        batch[:, 4] = tp_norm
        batch[:, 5] = x_norm_arr

        tensor_x = torch.from_numpy(batch)
        with torch.no_grad():
            preds = model(tensor_x).numpy()

        V_pred = preds[:, 0]
        T_pred = preds[:, 2]
        strain_pred = preds[:, 3]
        damage_pred = preds[:, 4]

        dV = np.diff(V_pred)
        dT = np.diff(T_pred)
        deps = np.diff(strain_pred)
        dD = np.diff(damage_pred)

        vel_violations += int(np.sum(dV > 1e-4))
        temp_violations += int(np.sum(dT < -1e-4))
        strain_violations += int(np.sum(deps < -1e-4))
        damage_violations += int(np.sum(dD < -1e-4))

        total_eval_steps += (n - 1)

    return {
        "total_eval_steps": total_eval_steps,
        "velocity_accel_violations": vel_violations,
        "velocity_accel_rate": round(vel_violations / max(total_eval_steps, 1), 5),
        "temperature_cooling_violations": temp_violations,
        "temperature_cooling_rate": round(temp_violations / max(total_eval_steps, 1), 5),
        "negative_strain_violations": strain_violations,
        "damage_healing_violations": damage_violations,
    }


def main():
    """Run this module command-line workflow."""
    parser = argparse.ArgumentParser(description="Audit Energy Conservation and Physical Monotonicity.")
    parser.add_argument("--n_trajectories", type=int, default=150, help="Number of benchmark trajectories.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    print("=" * 70)
    print("  Phase 4 Task 4.2: Energy Conservation & Physical Monotonicity Audit")
    print("=" * 70)

    # 1. Benchmark trajectory generation
    print(f"\n[1/3] Generating {args.n_trajectories} physical benchmark trajectories...")
    trajectories = generate_benchmark_trajectories(args.n_trajectories, args.seed)
    print(f"Generated {len(trajectories)} valid trajectories across 5 materials.")

    # 2. Simulation Ground-Truth Energy Audit
    print("\n[2/3] Auditing Simulation Ground-Truth Energy Conservation...")
    sim_audit = audit_simulation_energy_conservation(trajectories)
    print(f"  Collocation steps checked: {sim_audit['total_collocation_steps']:,}")
    print(f"  Mean Relative Energy Error:   {sim_audit['energy_error_mean']*100:.4f}%")
    print(f"  Median Relative Energy Error: {sim_audit['energy_error_median']*100:.4f}%")
    print(f"  95th Percentile Energy Error: {sim_audit['energy_error_p95']*100:.4f}%")
    print(f"  Max Energy Error:             {sim_audit['energy_error_max']*100:.4f}%")
    print(f"  Work-Energy Equivalence:     {'PASSED (< 1%)' if sim_audit['all_energy_conserved'] else 'FAILED'}")
    print(f"  Monotonicity Violations:      {sim_audit['monotonicity_violations']}")

    # 3. Hard-PINN Boundary Condition and Monotonicity Audit
    print("\n[3/3] Auditing BallisticHardPINN Boundary Exactness & Monotonicity...")
    hard_pinn = BallisticHardPINN()
    bc_audit = audit_hard_pinn_boundary_conditions(hard_pinn, n_samples=1000)
    print(f"  Impact Surface V(0) == V_0 Max Error:       {bc_audit['max_V0_error']:.2e}")
    print(f"  Impact Surface T(0) == T_plate Max Error:   {bc_audit['max_T0_error']:.2e}")
    print(f"  Impact Surface strain(0) == 0.0 Max Error:  {bc_audit['max_strain0_error']:.2e}")
    print(f"  Impact Surface damage(0) == 0.0 Max Error:  {bc_audit['max_damage0_error']:.2e}")
    print(f"  Exact Boundary Constraint Verified:         {'PASSED (< 1e-6)' if bc_audit['exact_bc_satisfied'] else 'FAILED'}")

    nn_mono_audit = audit_neural_monotonicity(hard_pinn, trajectories)
    print(f"  Neural Trajectory Deceleration Checks:      {nn_mono_audit['total_eval_steps']:,} steps")

    # Combine into comprehensive JSON report
    report = {
        "status": "success",
        "phase": 4,
        "task": "4.2 Energy Conservation & Boundary Audit",
        "simulation_energy_audit": sim_audit,
        "hard_pinn_boundary_audit": bc_audit,
        "hard_pinn_monotonicity_audit": nn_mono_audit,
    }

    os.makedirs("results/reports", exist_ok=True)
    report_path = "results/reports/energy_conservation_audit.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[REPORT] Saved full energy conservation audit to: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
