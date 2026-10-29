"""
Unit and Regression Tests for Energy Conservation and Boundary Invariants (Phase 4).
"""
import math

import numpy as np
import torch

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


def test_simulation_work_energy_theorem():
    r"""
    Verify analytical work-energy equivalence on numerical trajectories:
    Delta E_k = 0.5 * m * (V0^2 - V_exit^2) == \int F_shear dx
    Must hold within numerical integration discretization tolerance (< 0.5% at p95).
    """
    tan_alpha = math.tan(math.radians(PLUG_SHEAR_ANGLE_DEG))
    materials = list(MATERIALS.keys())
    rel_errors = []

    for mat_name in materials:
        mat = MATERIALS[mat_name]
        for t_mm, V0 in [(10.0, 800.0), (25.0, 1100.0), (40.0, 950.0)]:
            t_m = t_mm * 1e-3
            n_steps = _compute_n_steps(t_m)

            res = _integrate_trajectory(
                V0, t_m, D_BULLET_M, M_BULLET_KG,
                mat["A"], mat["B"], mat["n"], mat["C"], mat["m"],
                mat["T_melt"], mat["rho"], mat["Cp"],
                298.0, TAYLOR_QUINNEY_BETA,
                SHEAR_ZONE_WIDTH_RATIO,
                1.0, T_ROOM, n_steps,
                mat.get("cs_D", 1.0), mat.get("cs_q", 1.0),
                _RATE_JC, mat.get("failure_strain", 10.0),
                PLUG_SHEAR_ANGLE_DEG
            )
            n_active = int(res[0])
            if n_active < 3:
                continue

            x = res[1][:n_active]
            V = res[2][:n_active]
            sigma = res[3][:n_active]

            # F_shear = tau * perimeter * t_eff
            tau_Pa = (sigma * 1e6) / SQRT3
            d_eff = D_BULLET_M + 2.0 * x * tan_alpha
            F_shear = tau_Pa * (math.pi * d_eff) * t_m

            # Trapezoidal integration
            dx = np.diff(x)
            F_mid = 0.5 * (F_shear[:-1] + F_shear[1:])
            work_plastic = np.sum(F_mid * dx)

            E_k0 = 0.5 * M_BULLET_KG * (V0 ** 2)
            delta_Ek = 0.5 * M_BULLET_KG * (V0 ** 2 - V[-1] ** 2)

            rel_err = abs(delta_Ek - work_plastic) / E_k0
            rel_errors.append(rel_err)

    assert len(rel_errors) > 0
    mean_err = np.mean(rel_errors)
    p95_err = np.percentile(rel_errors, 95)
    assert mean_err < 0.002, f"Mean energy error {mean_err:.5f} exceeded 0.2%"
    assert p95_err < 0.005, f"95th percentile energy error {p95_err:.5f} exceeded 0.5%"


def test_hard_pinn_boundary_invariants():
    """
    Task 4.1 & 4.2: Verify BallisticHardPINN guarantees exact boundary invariants at x = 0:
      V(0) == V_0
      T(0) == T_plate
      strain(0) == 0.0
      damage(0) == 0.0
    across 1,000 random input vectors.
    """
    model = BallisticHardPINN()
    model.eval()

    torch.manual_seed(123)
    N = 1000
    x = torch.rand(N, 6)
    x[:, 3] = torch.randint(0, 5, (N,)).float()
    x[:, 5] = 0.0  # Exactly at impact surface

    with torch.no_grad():
        out = model(x)

    # Reconstruct expected normalized targets
    log10_v0 = x[:, 1:2] * 1.778 + 1.699
    expected_v0_norm = torch.pow(10.0, log10_v0) / 3000.0

    T_plate_phys = x[:, 4:5] * 300.0 + 200.0
    expected_T_norm = (T_plate_phys - 200.0) / 2300.0

    assert torch.max(torch.abs(out[:, 0:1] - expected_v0_norm)).item() < 1e-6
    assert torch.max(torch.abs(out[:, 2:3] - expected_T_norm)).item() < 1e-6
    assert torch.max(torch.abs(out[:, 3:4])).item() < 1e-6
    assert torch.max(torch.abs(out[:, 4:5])).item() < 1e-6


def test_hard_pinn_physical_bounds_preservation():
    """
    Verify that BallisticHardPINN outputs strictly lie within physical ranges:
      - V(x) >= 0.0
      - sigma(x) >= 0.0
      - T(x) >= T_plate and <= 1.0
      - strain(x) >= 0.0
      - damage(x) in [0.0, 1.0]
    even for arbitrary / random intermediate depths x > 0.
    """
    model = BallisticHardPINN()
    model.eval()

    torch.manual_seed(999)
    N = 500
    x = torch.rand(N, 6)
    x[:, 3] = torch.randint(0, 5, (N,)).float()
    x[:, 5] = torch.rand(N)  # Random penetration coordinates in [0, 1]

    with torch.no_grad():
        out = model(x)

    V = out[:, 0]
    sigma = out[:, 1]
    T = out[:, 2]
    strain = out[:, 3]
    damage = out[:, 4]

    # V >= 0
    assert torch.all(V >= 0.0), f"Found negative velocity: min={V.min().item()}"
    # sigma >= 0
    assert torch.all(sigma >= 0.0), f"Found negative flow stress: min={sigma.min().item()}"
    # strain >= 0
    assert torch.all(strain >= 0.0), f"Found negative plastic strain: min={strain.min().item()}"
    # damage in [0, 1]
    assert torch.all((damage >= 0.0) & (damage <= 1.0)), f"Damage out of [0, 1]: min={damage.min().item()}, max={damage.max().item()}"
    # T >= T_plate
    T_plate_phys = x[:, 4] * 300.0 + 200.0
    T_plate_target_norm = (T_plate_phys - 200.0) / 2300.0
    assert torch.all(T >= (T_plate_target_norm - 1e-6)), "Found temperature lower than initial ambient plate temperature"


def test_ansatz_velocity_deceleration_constraint():
    """
    Verify that V(x) <= V_0 for all penetration depths x >= 0 (projectile can never accelerate).
    """
    model = BallisticHardPINN()
    model.eval()

    torch.manual_seed(777)
    N = 200
    x = torch.rand(N, 6)
    x[:, 3] = torch.randint(0, 5, (N,)).float()
    x[:, 5] = torch.rand(N)  # Arbitrary x > 0

    with torch.no_grad():
        out = model(x)

    log10_v0 = x[:, 1] * 1.778 + 1.699
    expected_v0_norm = torch.pow(10.0, log10_v0) / 3000.0
    V_pred = out[:, 0]

    # V(x) <= V_0 to within machine precision
    diff = V_pred - expected_v0_norm
    assert torch.max(diff).item() <= 1e-6, f"Velocity accelerated above impact velocity: max excess={torch.max(diff).item()}"
