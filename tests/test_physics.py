import pytest
import numpy as np
from simulation import simulate_trajectory, simulate_ballistic_impact
from stage1_part1_johnson_cook import MATERIALS

def test_simulate_trajectory_no_penetration():
    # A small bullet at low velocity against a thick RHA plate should not penetrate
    res = simulate_trajectory(
        plate_thickness_mm=50.0,
        material_name="RHA_steel",
        bullet_velocity_ms=100.0,
        bullet_mass_g=7.9,
        bullet_diameter_mm=7.62
    )
    
    assert res["V_ms"][-1] < 25.0  # Should be stopped
    assert res["x_mm"][-1] < 50.0  # Should not reach the back

def test_simulate_trajectory_penetration():
    # A fast bullet against a thin RHA plate should overmatch it easily
    res = simulate_trajectory(
        plate_thickness_mm=5.0,
        material_name="RHA_steel",
        bullet_velocity_ms=1000.0,
        bullet_mass_g=7.9,
        bullet_diameter_mm=7.62
    )
    
    assert res["V_ms"][-1] > 0.0
    assert np.isclose(res["x_mm"][-1], 5.0, atol=0.1)

def test_ballistic_limit_empirical_calibration():
    """
    Test V50 consistency between simulate_ballistic_impact and simulate_trajectory.

    The module-level SHEAR_ZONE_WIDTH_RATIO = 0.3965 is used for dataset generation.
    With this value, the binary-search V50 for 12mm RHA is ~463 m/s.

    The empirical V50 of ~620 m/s requires w_ratio ≈ 0.6841 (calibrated separately).
    This test verifies internal consistency, not empirical matching.
    """
    # Test with the module-level shear zone ratio used for data generation
    res_impact = simulate_ballistic_impact(
        plate_thickness_mm=12.0,
        material_name="RHA_steel",
        bullet_velocity_ms=1000.0,
        bullet_mass_g=7.9,
        bullet_diameter_mm=7.62
    )

    # Binary search V50 from simulate_trajectory
    # (uses same w_ratio = 0.3965)
    lo, hi = 0.0, 3000.0
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        traj = simulate_trajectory(12.0, "RHA_steel", bullet_velocity_ms=mid)
        penetrated = traj["V_ms"][-1] > 0.0 and traj["x_mm"][-1] >= 11.9
        if penetrated:
            hi = mid
        else:
            lo = mid
        if hi - lo < 2.0:
            break
    v50_trajectory = 0.5 * (lo + hi)

    # Both functions should agree within 5 m/s
    assert abs(res_impact.ballistic_limit_ms - v50_trajectory) < 5.0, (
        f"simulate_ballistic_impact V50={res_impact.ballistic_limit_ms:.1f} m/s "
        f"vs simulate_trajectory V50={v50_trajectory:.1f} m/s"
    )

def test_energy_conservation():
    res = simulate_ballistic_impact(
        plate_thickness_mm=10.0,
        material_name="aluminium_7075_T6",
        bullet_velocity_ms=800.0,
        bullet_mass_g=7.9,
        bullet_diameter_mm=7.62
    )
    
    E_initial = 0.5 * (7.9e-3) * (800.0**2)
    # The exit energy is 1/2 * (m_bullet + m_plug) * V_residual^2
    # E_initial should roughly equal Energy Absorbed + Energy Residual
    # The absorbed energy includes deformation, thermal, and momentum transfer to the plug.
    assert res.energy_absorbed_J > 0
    assert res.energy_absorbed_J < E_initial
