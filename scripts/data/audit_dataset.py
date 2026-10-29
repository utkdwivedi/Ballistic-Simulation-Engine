"""
Dataset Physical and Statistical Quality Audit Script
Validates 100K trajectory dataset for Phase 1 of Master Plan.
"""
import glob
import json
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

import pandas as pd

from ballistic.physics.johnson_cook import MATERIALS

DATA_DIR = "pinn_data"
SPLITS = ["train", "val", "test"]
OUTPUT_REPORT = "results/reports/data_audit_summary.json"


def audit_split(split_name: str) -> dict:
    split_dir = os.path.join(DATA_DIR, split_name)
    shards = sorted(glob.glob(os.path.join(split_dir, "*.parquet")))
    if not shards:
        return {"error": f"No shards found in {split_dir}"}

    dfs = [pd.read_parquet(s) for s in shards]
    df = pd.concat(dfs, ignore_index=True)

    # 1. Null / NaN / Inf checks
    nan_counts = df.isna().sum().to_dict()
    has_nan = any(v > 0 for v in nan_counts.values())

    # 2. Row count and shards
    n_rows = len(df)
    n_shards = len(shards)

    # 3. Value bounds
    v_min, v_max = float(df["V_ms"].min()), float(df["V_ms"].max())
    s_min, s_max = float(df["sigma_MPa"].min()), float(df["sigma_MPa"].max())
    t_min, t_max = float(df["T_K"].min()), float(df["T_K"].max())
    e_min, e_max = float(df["strain"].min()), float(df["strain"].max())
    d_min, d_max = float(df["damage"].min()), float(df["damage"].max())

    # 4. Material class counts
    mat_counts = df["material_id"].value_counts().to_dict()
    mat_keys = list(MATERIALS.keys())
    mat_distribution = {mat_keys[k]: int(count) for k, count in mat_counts.items() if k < len(mat_keys)}

    # 5. Penetration / Perforation statistics
    perforated_rows = int((df["damage"] >= 0.999).sum())
    arrested_rows = int((df["V_ms"] <= 1e-3).sum())
    perforation_rate = round(perforated_rows / n_rows, 4)

    return {
        "split": split_name,
        "n_shards": n_shards,
        "n_rows": n_rows,
        "has_nan": has_nan,
        "nan_counts": nan_counts,
        "ranges": {
            "V_ms": [v_min, v_max],
            "sigma_MPa": [s_min, s_max],
            "T_K": [t_min, t_max],
            "strain": [e_min, e_max],
            "damage": [d_min, d_max],
        },
        "material_distribution": mat_distribution,
        "perforated_rows": perforated_rows,
        "arrested_rows": arrested_rows,
        "perforation_rate": perforation_rate,
    }


def main():
    """Run this module command-line workflow."""
    print("=" * 60)
    print("  Phase 1: Dataset Physical & Statistical Audit")
    print("=" * 60)

    os.makedirs("results/reports", exist_ok=True)
    summary = {}
    all_valid = True
    total_rows = 0

    for split in SPLITS:
        print(f"Auditing {split} split...")
        res = audit_split(split)
        summary[split] = res

        if "error" in res:
            print(f"  [ERROR] {res['error']}")
            all_valid = False
            continue

        total_rows += res["n_rows"]
        print(f"  Rows: {res['n_rows']:,} across {res['n_shards']} shards")
        print(f"  Zero NaNs: {not res['has_nan']}")
        print(f"  Velocity: [{res['ranges']['V_ms'][0]:.1f}, {res['ranges']['V_ms'][1]:.1f}] m/s")
        print(f"  Stress:   [{res['ranges']['sigma_MPa'][0]:.1f}, {res['ranges']['sigma_MPa'][1]:.1f}] MPa")
        print(f"  Temp:     [{res['ranges']['T_K'][0]:.1f}, {res['ranges']['T_K'][1]:.1f}] K")
        print(f"  Damage:   [{res['ranges']['damage'][0]:.3f}, {res['ranges']['damage'][1]:.3f}]")
        print(f"  Perforation Rate: {res['perforation_rate']*100:.1f}%")

        # Acceptance checks
        assert not res["has_nan"], f"NaN detected in {split}!"
        assert res["ranges"]["V_ms"][0] >= 0.0, f"Negative velocity in {split}!"
        assert res["ranges"]["sigma_MPa"][0] >= 0.0, f"Negative flow stress in {split}!"
        assert res["ranges"]["damage"][0] >= 0.0 and res["ranges"]["damage"][1] <= 1.0001, f"Invalid damage range in {split}!"

    summary["total_rows"] = total_rows
    summary["audit_passed"] = all_valid and total_rows >= 1_400_000

    with open(OUTPUT_REPORT, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Total Rows Audited: {total_rows:,}")
    print(f"Audit Status: {'PASSED' if summary['audit_passed'] else 'FAILED'}")
    print(f"Report saved to: {OUTPUT_REPORT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
