"""
Ballistic Simulation — Stage 2: Data Generation

Generates 100K trajectories using Latin Hypercube Sampling (LHS).
Zero noise is injected to preserve the physical ODE manifold for PINN autograd.
Outputs stratified chunked Parquet files.

Run: python data_gen.py
"""
import os
import glob
import argparse
import numpy as np
from scipy.stats import qmc
from simulation import generate_pinn_data, MATERIALS


# Per-material velocity ranges to achieve ~30% perforation rate
# P(V > V50) = ln(v_max/V50) / ln(v_max/v_min) = 0.3 => solve for v_max
# Actual rates are lower (~19-23%) because plate thickness variation raises effective V50
MATERIAL_VELOCITY_RANGES = {
    "RHA_steel":           (200, 1500),  # V50=463, actual ~24% perforated
    "aluminium_7075_T6":   (100, 900),   # V50=259, actual ~25% perforated
    "copper":              (200, 1100),   # V50=388, slightly wider for ~25%
    "alumina_Al2O3":       (100, 1000),  # V50=267, actual ~25% perforated
    "UHMWPE":              (100, 750),   # V50=216, actual ~25% perforated
}


def sample_space(n_samples: int, seed: int = 42):
    """
    Sample the 5D input space using Latin Hypercube Sampling for maximum coverage.
    Uses per-material velocity ranges to balance perforation rates.
    """
    sampler = qmc.LatinHypercube(d=5, seed=seed)
    lhs_samples = sampler.random(n=n_samples)

    # Scale each dimension to its physical bounds

    # 1. Plate Thickness: 5 - 50 mm
    thicknesses = 5.0 + lhs_samples[:, 0] * (50.0 - 5.0)

    # 2. Obliquity: 0 - 60 degrees
    obliquities = lhs_samples[:, 2] * 60.0

    # 3. Plate Temperature: 233 - 373 K (-40 to 100 C)
    temperatures = 233.0 + lhs_samples[:, 3] * (373.0 - 233.0)

    # 4. Material Type: categorical mapping (uniform across materials)
    mat_keys = list(MATERIALS.keys())
    mat_indices = np.floor(lhs_samples[:, 4] * len(mat_keys)).astype(int)
    mat_indices = np.clip(mat_indices, 0, len(mat_keys) - 1)
    materials = [mat_keys[i] for i in mat_indices]

    # 5. Velocity: per-material log-scale sampling for balanced perforation
    velocities = np.zeros(n_samples)
    for mat_id, mat_name in enumerate(mat_keys):
        v_min, v_max = MATERIAL_VELOCITY_RANGES.get(mat_name, (200, 1500))
        log_v_min = np.log10(v_min)
        log_v_max = np.log10(v_max)
        mask = (mat_indices == mat_id)
        n_mat = mask.sum()
        if n_mat > 0:
            mat_lhs = qmc.LatinHypercube(d=1, seed=seed + 100 + mat_id).random(n=n_mat)
            velocities[mask] = 10 ** (log_v_min + mat_lhs[:, 0] * (log_v_max - log_v_min))

    return thicknesses, velocities, obliquities, temperatures, materials


def main():
    parser = argparse.ArgumentParser(description="Generate PINN trajectory data.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed for reproducibility.")
    parser.add_argument("--n_samples", type=int, default=100_000, help="Number of trajectories to generate (default: 100000).")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Splits to generate (default: train val test).")
    args = parser.parse_args()
    seed = args.seed
    n_total = args.n_samples
    target_splits = set(args.splits)

    print("=" * 60)
    print(f"  Stage 2: Intelligent Data Generation (LHS)")
    print("=" * 60)
    print(f"\nGenerating {n_total} trajectories across 5 dimensions with seed {seed}...")

    thicknesses, velocities, obliquities, temperatures, materials = sample_space(n_total, seed=seed)

    # Splitting strategy:
    # LHS guarantees uniform space-filling across all 5 dimensions for the
    # FULL 100K sample. A random shuffle then split into 70/15/15 preserves
    # approximate uniformity in each subset because any large random subset
    # of an LHS design retains near-uniform marginal coverage. This is
    # standard practice for LHS-based experiment design.
    indices = np.arange(n_total)
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_end = int(0.70 * n_total)
    val_end = int(0.85 * n_total)

    splits = {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:]
    }

    active_splits = {k: v for k, v in splits.items() if k in target_splits}

    for split_name, idxs in active_splits.items():
        n_split = len(idxs)
        print(f"\n--- Processing {split_name.upper()} split ({n_split} trajectories) ---")

        output_dir = os.path.join("pinn_data", split_name)
        os.makedirs(output_dir, exist_ok=True)
        # Clean existing shards to avoid mixing datasets
        for old_shard in glob.glob(os.path.join(output_dir, "*.parquet")):
            try:
                os.remove(old_shard)
            except OSError:
                pass

        try:
            # Pull 15 collocation points per trajectory
            generate_pinn_data(
                thicknesses_mm=thicknesses[idxs],
                velocities_ms=velocities[idxs],
                material_names=[materials[i] for i in idxs],
                obliquities_deg=obliquities[idxs],
                plate_temperatures_K=temperatures[idxs],
                output_dir=output_dir,
                n_collocation=15,
                seed=seed
            )

            # Verify generated sizes
            shards = glob.glob(os.path.join(output_dir, "*.parquet"))
            total_kb = sum(os.path.getsize(s) for s in shards) / 1024
            print(f"  Finished {split_name}. Wrote {len(shards)} shards ({total_kb:.0f} KB total)")

        except Exception as e:
            print(f"  ERROR in {split_name} split: {e}")
            print(f"  Continuing with remaining splits...")

    print("\n" + "=" * 60)
    print("  Data generation complete. No noise injected.")
    print("=" * 60)


if __name__ == "__main__":
    main()
