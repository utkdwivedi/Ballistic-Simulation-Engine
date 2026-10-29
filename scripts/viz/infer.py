"""
Ballistic Simulation — Unified Inference Script

Usage:
    python scripts/infer.py \
        --thickness_mm 12.0 \
        --velocity_ms 850.0 \
        --obliquity_deg 15.0 \
        --material RHA_steel \
        --temperature_K 298.0 \
        --x_mm 6.0

Outputs predicted physical quantities:
    - V_ms: Residual bullet velocity (m/s)
    - sigma_MPa: Flow stress (MPa)
    - T_K: Temperature in deformation zone (K)
    - strain: Plastic shear strain
    - damage: Johnson-Cook damage state [0, 1]
"""
import argparse
import json
import math

import numpy as np
import torch

from ballistic.dataloader import INPUT_RANGES, LOG1P_TARGET_COLS, LOG_INPUT_COLS, TARGET_RANGES
from ballistic.models.architecture import BallisticHardPINN, BallisticResMLP
from ballistic.physics.johnson_cook import MATERIALS


def load_model(checkpoint: str = "models/pinn_production_best.pt",
               meta: str = "models/pinn_production_meta.json"):
    with open(meta) as f:
        m = json.load(f)

    arch = m.get("arch", "hard_pinn")
    num_materials = m.get("num_materials", len(MATERIALS))
    embedding_dim = m.get("embedding_dim", 8)
    hidden_dim = m.get("hidden_dim", 256)

    if arch == "hard_pinn":
        model = BallisticHardPINN(
            num_materials=num_materials,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_blocks=m.get("n_blocks", 3)
        )
    elif arch == "resmlp":
        model = BallisticResMLP(
            num_materials=num_materials,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_blocks=m.get("n_blocks", 3)
        )
    else:
        from scripts.train_fcnn import BallisticFCNN
        model = BallisticFCNN(
            num_materials=num_materials,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_layers=m.get("n_layers", 6),
        )

    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    return model, m


def prepare_input(thickness_mm: float, velocity_ms: float, obliquity_deg: float,
                  material: str, temperature_K: float, x_mm: float) -> torch.Tensor:
    mat_keys = list(MATERIALS.keys())
    if material not in mat_keys:
        raise ValueError(f"Unknown material '{material}'. Available: {mat_keys}")
    mat_id = mat_keys.index(material)

    raw = np.array([thickness_mm, velocity_ms, obliquity_deg,
                    float(mat_id), temperature_K, x_mm], dtype=np.float32)

    # 1. Log10 on velocity
    if "velocity_ms" in LOG_INPUT_COLS:
        raw[1] = math.log10(max(raw[1], 1.0))

    # 2. Min-max normalize (skip categorical material_id at index 3)
    input_cols = ["thickness_mm", "velocity_ms", "obliquity_deg",
                  "material_id", "plate_temperature_K", "x_mm"]
    for j, col in enumerate(input_cols):
        if col == "material_id":
            continue
        lo, hi = INPUT_RANGES[col]
        raw[j] = (raw[j] - lo) / (hi - lo + 1e-12)

    return torch.from_numpy(raw).unsqueeze(0)  # [1, 6]


def denormalize_output(pred_norm: np.ndarray) -> dict:
    target_cols = ["V_ms", "sigma_MPa", "T_K", "strain", "damage"]
    clamped_norm = np.clip(pred_norm, 0.0, 1.0)
    result = {}
    for j, col in enumerate(target_cols):
        lo, hi = TARGET_RANGES[col]
        val = clamped_norm[j] * (hi - lo) + lo
        if col in LOG1P_TARGET_COLS:
            val = math.expm1(val)
        result[col] = round(float(val), 4)
    return result


def main():
    """Run this module command-line workflow."""
    parser = argparse.ArgumentParser(description="Ballistic Model Inference")
    parser.add_argument("--thickness_mm", type=float, required=True, help="Plate thickness (mm)")
    parser.add_argument("--velocity_ms", type=float, required=True, help="Impact velocity (m/s)")
    parser.add_argument("--obliquity_deg", type=float, required=True, help="Impact obliquity angle (degrees)")
    parser.add_argument("--material", type=str, required=True, choices=list(MATERIALS.keys()), help="Target material")
    parser.add_argument("--temperature_K", type=float, default=298.0, help="Initial temperature (K)")
    parser.add_argument("--x_mm", type=float, required=True, help="Penetration depth coordinate (mm)")
    parser.add_argument("--checkpoint", type=str, default="models/pinn_production_best.pt", help="Path to model checkpoint")
    parser.add_argument("--meta", type=str, default="models/pinn_production_meta.json", help="Path to metadata JSON")
    args = parser.parse_args()

    model, meta = load_model(args.checkpoint, args.meta)
    x = prepare_input(args.thickness_mm, args.velocity_ms, args.obliquity_deg,
                      args.material, args.temperature_K, args.x_mm)

    with torch.no_grad():
        pred_norm = model(x).squeeze(0).numpy()

    result = denormalize_output(pred_norm)

    print("\n" + "=" * 50)
    print("  Ballistic Prediction Results")
    print("=" * 50)
    print(f"  Material    : {args.material}")
    print(f"  Thickness   : {args.thickness_mm} mm")
    print(f"  Impact vel. : {args.velocity_ms} m/s @ {args.obliquity_deg} deg")
    print(f"  Position    : {args.x_mm} mm into plate")
    print(f"  Temperature : {args.temperature_K} K")
    print(f"  Checkpoint  : {args.checkpoint}")
    print("-" * 50)
    for k, v in result.items():
        print(f"  {k:12s}: {v}")
    print("=" * 50)


if __name__ == "__main__":
    main()
