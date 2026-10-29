# scripts/validate_gpu_benchmark.py

"""GPU benchmark harness.

This script evaluates the forward inference speed of the Hard-PINN model
using a PyTorch DataLoader on a CUDA device.  It measures:

1. **Standalone single‑step latency** (average time for a 200‑size batch).
2. **Batched throughput** (time per sample when packing a large batch).

Results are printed to stdout and saved to `results/reports/benchmarks/gpu_benchmark.csv`.

The script is **optional** – it will skip if no CUDA device is available.  It should be added to the CI matrix later.
"""

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ballistic.dataloader import BallisticPINNDataset
from ballistic.models.architecture import BallisticHardPINN

# -------- Configuration --------
NUM_TRAIN_SAMPLES = 1024  # batch size for forward pass measurement
gpu_devices = list(range(torch.cuda.device_count()))

# -------- Core helpers --------
def measure_inference_speed(model, loader, device):
    torch.cuda.synchronize(device)
    start = time.time()
    with torch.no_grad():
        for x, _ in loader:
            _ = model(x.to(device))
    torch.cuda.synchronize(device)
    elapsed = time.time() - start
    return elapsed

# -------- Main --------
def main():
    """Run this module command-line workflow."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Running GPU benchmark on {torch.cuda.get_device_name(device)}")
    else:
        print("No CUDA device found – skipping GPU benchmark.")
        return

    # Load model
    checkpoint = torch.load("models/hardpinn.pt", map_location="cpu")
    model = BallisticHardPINN()
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    # Create a synthetic DataLoader
    dataset = BallisticPINNDataset("pinn_data/train", normalize=True)
    loader = DataLoader(dataset, batch_size=NUM_TRAIN_SAMPLES, shuffle=False)

    # Warm‑up
    for _ in range(5):
        with torch.no_grad():
            batch = next(iter(loader))
            _ = model(batch[0].to(device))

    warm = measure_inference_speed(model, loader, device)
    print(f"Warm‑up latency : {warm:.6f} s for {NUM_TRAIN_SAMPLES} samples")

    # Measure steady state over 20 trials
    times = []
    for _ in range(20):
        times.append(measure_inference_speed(model, loader, device))
    avg = sum(times) / len(times)
    print(f"Avg  latency   : {avg:.6f} s for {NUM_TRAIN_SAMPLES} samples")

    # Throughput (samples per second)
    throughput = NUM_TRAIN_SAMPLES / avg
    print(f"Throughput     : {throughput:.2f} samples/s (on 1 GPU) ")

    # Save results
    bench_dir = Path("results/reports/benchmarks")
    bench_dir.mkdir(parents=True, exist_ok=True)
    csv_path = bench_dir / "gpu_benchmark.csv"
    csv_path.write_text(
        f"timestamp,batch_size,latency,throughput\n"
        f"{time.time()},{NUM_TRAIN_SAMPLES},{avg:.6f},{throughput:.2f}\n",
        encoding="utf-8",
    )
    print(f"Results saved to {csv_path})")


if __name__ == "__main__":
    main()
