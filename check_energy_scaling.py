# check_energy_scaling.py
# ------------------------------------------------------------
# Run a single forward-pass on a tiny batch and assert that the
# energy residual is within the acceptable range.
# ------------------------------------------------------------
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Import the data loader and the PINN model defined in train_pinn_demo.py
from pinn_dataloader import BallisticPINNDataset
from train_pinn_demo import (
    BallisticPINN, compute_physics_loss,
    MAT_A, MAT_B, MAT_N, MAT_C_JC, MAT_M, MAT_TMELT, MAT_RHO, MAT_CP,
)

# ------------------------------------------------------------------
# 1️⃣  Load a tiny slice of the training data (just a few samples)
# ------------------------------------------------------------------
dataset = BallisticPINNDataset('pinn_data/train', normalize=True)
loader = DataLoader(
    Subset(dataset, range(min(32, len(dataset)))),
    batch_size=32,
    shuffle=False,
)

batch_x, _ = next(iter(loader))

# ------------------------------------------------------------------
# 2️⃣  Build the model (seeded for reproducibility)
# ------------------------------------------------------------------
device = torch.device('cpu')
torch.manual_seed(3)   # deterministic init – avoids rare extreme outputs
model = BallisticPINN()
model_path = 'models/pinn_best.pt'
import os
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ------------------------------------------------------------------
# 3️⃣  Prepare the exact tensors that compute_physics_loss expects
# ------------------------------------------------------------------
x_mm = batch_x[:, 5:6].clone().detach().requires_grad_(True)
bx_grad = torch.cat([batch_x[:, :5], x_mm], dim=1).to(device)

# ------------------------------------------------------------------
# 4️⃣  Forward pass (must keep grad enabled so autograd can compute dV/dx etc.)
# ------------------------------------------------------------------
preds_norm = model(bx_grad)

# ------------------------------------------------------------------
# 5️⃣  Assemble the arguments that compute_physics_loss needs
# ------------------------------------------------------------------
mat_tensors = {
    "A": MAT_A.to(device), "B": MAT_B.to(device), "n": MAT_N.to(device),
    "C": MAT_C_JC.to(device), "m": MAT_M.to(device),
    "T_melt": MAT_TMELT.to(device), "rho": MAT_RHO.to(device), "Cp": MAT_CP.to(device),
}

# ------------------------------------------------------------------
# 6️⃣  Compute the physics loss
# ------------------------------------------------------------------
physics_loss = compute_physics_loss(
    preds_norm=preds_norm,
    x_mm=x_mm,
    batch_x=batch_x,
    mat_tensors=mat_tensors,
    device=device,
)

# ------------------------------------------------------------------
# 7️⃣  Check the magnitude and give a short diagnostic
# ------------------------------------------------------------------
print(f"Physics loss (energy residual) : {physics_loss.item():.2e}")
if physics_loss.item() > 10.0:
    print("WARNING: Residual is larger than the chosen threshold (10).")
    print("  The most common fix is to increase SCALE_TEMP_GRAD (or lower")
    print("  SCALE_STRESS_MPA / SCALE_FORCE_N) so that the residuals are")
    print("  much smaller. A good rule of thumb is to keep every residual")
    print("  below ~10 before the final training run.")
else:
    print("OK: Energy residual is within the acceptable range.")