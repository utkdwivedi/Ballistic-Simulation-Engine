import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# 1. Training Curves
log_files = glob.glob(os.path.join("reports", "training_logs", "*.csv"))
if log_files:
    latest_log = max(log_files, key=os.path.getmtime)
    df = pd.read_csv(latest_log)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["epoch"], df["total_loss"], label="Total Loss", color="#0072B2", lw=2)
    ax.plot(df["epoch"], df["data_loss"], label="Data Loss", color="#D55E00", lw=1.5, ls="--")
    ax.plot(df["epoch"], df["phys_loss"], label="Physics Loss", color="#009E73", lw=1.5, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.legend()
    ax.set_title("Training Loss Convergence")
    fig.savefig("figures_report/training_curves.png", dpi=300)

# 2. GPU Benchmark comparison
data = [109.5] # Throughput from benchmark
modes = ["RTX 4060 Laptop (PINN)"]
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(modes, data, color="#0072B2")
ax.set_ylabel("Samples per second")
ax.set_title("Inference Throughput (PINN)")
fig.savefig("figures_report/gpu_throughput.png", dpi=300)

print("Generated training_curves.png and gpu_throughput.png")
