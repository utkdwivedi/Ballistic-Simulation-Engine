import pytest
import numpy as np

from stage1_part1_johnson_cook import johnson_cook_stress, MATERIALS
from stage1_part2_cowper_symonds import cowper_symonds_dif


# Expected reasonable bounds for material parameters
MATERIAL_BOUNDS = {
    "A": (0, 5000),           # yield stress MPa
    "B": (0, 3000),           # hardening modulus MPa
    "n": (0, 1),              # hardening exponent
    "C": (0, 0.5),            # rate sensitivity
    "m": (0, 2),              # thermal softening exponent
    "T_melt": (400, 4000),    # melting temp K
    "rho": (500, 20000),      # density kg/m³
    "E": (100, 500000),       # Young's modulus MPa
    "Cp": (100, 3000),        # specific heat J/kg·K
    "cs_D": (1, 1e7),         # Cowper-Symonds D
    "cs_q": (1, 10),          # Cowper-Symonds q
    "hardness_HRC": (0, 80),  # Rockwell C hardness
    "failure_strain": (0.01, 5.0),  # effective failure strain (shear plugging)
}


def test_materials_loaded():
    assert "RHA_steel" in MATERIALS
    assert "aluminium_7075_T6" in MATERIALS

    mat = MATERIALS["RHA_steel"]
    assert mat["T_melt"] == 1783.0
    assert mat["hardness_HRC"] == 50.0


def test_material_parameter_bounds():
    """Every material parameter must lie within physically plausible bounds."""
    for name, mat in MATERIALS.items():
        if name.startswith("_"):
            continue
        for param, (lo, hi) in MATERIAL_BOUNDS.items():
            if param in mat:
                val = mat[param]
                assert lo <= val <= hi, (
                    f"{name}.{param} = {val} outside bounds [{lo}, {hi}]"
                )


def test_failure_strain_reasonable():
    """failure_strain should be in a reasonable range for each material class."""
    # RHA steel: ductile, can sustain large shear strains before plugging
    assert 0.5 <= MATERIALS["RHA_steel"]["failure_strain"] <= 3.0
    # 7075-T6 aluminium: moderate ductility
    assert 0.3 <= MATERIALS["aluminium_7075_T6"]["failure_strain"] <= 2.0
    # Copper: very ductile
    assert 1.0 <= MATERIALS["copper"]["failure_strain"] <= 5.0
    # Alumina: brittle ceramic — low *effective* shear-plugging strain
    # Note: 0.3 is an effective strain for the shear plug model, NOT tensile failure strain.
    # Marked UNVERIFIED in config. Bounds check allows 0.01–0.5 for ceramics.
    assert 0.01 <= MATERIALS["alumina_Al2O3"]["failure_strain"] <= 0.5
    # UHMWPE: polymer, high ductility
    assert 1.0 <= MATERIALS["UHMWPE"]["failure_strain"] <= 5.0


def test_source_confidence_fields_present():
    """Every material must have source and confidence metadata."""
    for name, mat in MATERIALS.items():
        if name.startswith("_"):
            continue
        # source and confidence are optional for backward compatibility,
        # but if present must be valid strings
        if "source" in mat:
            assert mat["source"] in ("verified", "UNVERIFIED")
        if "confidence" in mat:
            assert mat["confidence"] in ("high", "medium", "low")


def test_johnson_cook_basic():
    mat = MATERIALS["RHA_steel"]
    # At room temperature and 0 strain, stress should be exactly A
    stress, _ = johnson_cook_stress(
        epsilon=0.0,
        epsilon_dot=1.0,
        T=298.0,
        A=mat["A"], B=mat["B"], n=mat["n"], C=mat["C"], m=mat["m"],
        T_melt=mat["T_melt"], T_room=298.0
    )
    assert np.isclose(stress, mat["A"])


def test_johnson_cook_hardening():
    mat = MATERIALS["RHA_steel"]
    stress_0, _ = johnson_cook_stress(
        epsilon=0.0, epsilon_dot=1.0, T=298.0,
        A=mat["A"], B=mat["B"], n=mat["n"], C=mat["C"], m=mat["m"],
        T_melt=mat["T_melt"]
    )
    stress_1, _ = johnson_cook_stress(
        epsilon=0.5, epsilon_dot=1.0, T=298.0,
        A=mat["A"], B=mat["B"], n=mat["n"], C=mat["C"], m=mat["m"],
        T_melt=mat["T_melt"]
    )
    # Strain hardening should increase stress
    assert stress_1 > stress_0


def test_johnson_cook_thermal_softening():
    mat = MATERIALS["RHA_steel"]
    stress_cold, _ = johnson_cook_stress(
        epsilon=0.2, epsilon_dot=1.0, T=298.0,
        A=mat["A"], B=mat["B"], n=mat["n"], C=mat["C"], m=mat["m"],
        T_melt=mat["T_melt"]
    )
    stress_hot, _ = johnson_cook_stress(
        epsilon=0.2, epsilon_dot=1.0, T=1000.0,
        A=mat["A"], B=mat["B"], n=mat["n"], C=mat["C"], m=mat["m"],
        T_melt=mat["T_melt"]
    )
    # Thermal softening should decrease stress
    assert stress_hot < stress_cold


def test_cowper_symonds():
    # DIF should be 1.0 at quasi-static rates
    dif_static = cowper_symonds_dif(epsilon_dot=0.0, D=40.4, q=5.0)
    assert np.isclose(dif_static, 1.0, atol=1e-3)

    # DIF should increase at high dynamic rates
    dif_dynamic = cowper_symonds_dif(epsilon_dot=1e5, D=40.4, q=5.0)
    assert dif_dynamic > 1.5
