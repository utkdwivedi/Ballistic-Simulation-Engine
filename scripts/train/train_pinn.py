"""
Ballistic Simulation — PINN Training (Fixed, Non-Dimensionalized & Complete)

Fixes applied:
  [FIX-1] Denormalization before physics residuals (dimensional correctness)
  [FIX-2] x_mm → x_m unit conversion before momentum residual
  [FIX-3] Per-sample material property indexing (dynamic across all materials)
  [FIX-4] log1p inverse (expm1) on strain before JC residual
  [FIX-5] Gradient clipping added (max_norm=0.5 for PINN autograd stability)
  [FIX-6] CosineAnnealingLR schedule added
  [FIX-7] Model depth/width improved (6 layers, 256 hidden)
  [FIX-8] Non-dimensionalized physics residuals to balance with data loss (prevents loss explosion)
  [FIX-9] Single forward pass autograd graph optimization (preserves BatchNorm stats and speeds up training)
  [FIX-10] Warmstart from supervised checkpoint support (prevents early random gradient shock)
"""

import argparse
import csv
import datetime
import json
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from ballistic.dataloader import (
    INPUT_COLS,
    INPUT_RANGES,
    LOG1P_TARGET_COLS,
    LOG_INPUT_COLS,
    TARGET_COLS,
    TARGET_RANGES,
    BallisticPINNDataset,
)
from ballistic.physics.johnson_cook import MATERIALS
from ballistic.physics.residuals import (
    SHEAR_ZONE_WIDTH_RATIO,
    SQRT3,
    energy_residual_torch,
    jc_constitutive_residual_torch,
    momentum_residual_torch,
    strain_residual_torch,
)

# Fixed bullet geometry (7.62mm NATO ball) – same as simulation.py
D_BULLET_M = 0.00762
M_BULLET_KG = 0.0079
from ballistic.models.architecture import BallisticHardPINN, BallisticResMLP

# -------------------------------------------------------------------
# Hyperparameters
# -------------------------------------------------------------------
DEFAULT_EPOCHS = 10
BATCH_SIZE = 512
LR = 2e-4
NUM_MATERIALS = len(MATERIALS)
EMBEDDING_DIM = 8
HIDDEN_DIM = 256
N_LAYERS = 6
GRAD_CLIP = 0.5

# -------------------------------------------------------------------
# Non‑dimensional loss weights (equal weighting after residual fix)
# -------------------------------------------------------------------
LAMBDA_DATA = 1.0
LAMBDA_JC = 1.0
LAMBDA_MOMENTUM = 1.0
LAMBDA_STRAIN = 1.0
LAMBDA_ENERGY = 1.0

# -------------------------------------------------------------------
# Physical normalisation scales (SI units)
# -------------------------------------------------------------------
SCALE_STRESS_MPA = 1000.0    # ~1 GPa characteristic flow stress
SCALE_FORCE_N = 500000.0    # ~500 kN characteristic shear force
SCALE_STRAIN_GRAD = 2500.0  # max strain gradient (1/(w·√3))
SCALE_TEMP_GRAD = 1_000_000.0  # temperature gradient ceiling

# -------------------------------------------------------------------
# Pre‑computed material property tensors (for fast batch indexing)
# -------------------------------------------------------------------
MAT_KEYS = list(MATERIALS.keys())
MAT_A = torch.tensor([MATERIALS[k]["A"] for k in MAT_KEYS], dtype=torch.float32)
MAT_B = torch.tensor([MATERIALS[k]["B"] for k in MAT_KEYS], dtype=torch.float32)
MAT_N = torch.tensor([MATERIALS[k]["n"] for k in MAT_KEYS], dtype=torch.float32)
MAT_C_JC = torch.tensor([MATERIALS[k]["C"] for k in MAT_KEYS], dtype=torch.float32)
MAT_M = torch.tensor([MATERIALS[k]["m"] for k in MAT_KEYS], dtype=torch.float32)
MAT_TMELT = torch.tensor([MATERIALS[k]["T_melt"] for k in MAT_KEYS], dtype=torch.float32)
MAT_RHO = torch.tensor([MATERIALS[k]["rho"] for k in MAT_KEYS], dtype=torch.float32)
MAT_CP = torch.tensor([MATERIALS[k]["Cp"] for k in MAT_KEYS], dtype=torch.float32)

# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def denorm_target(y_norm: torch.Tensor, col: str) -> torch.Tensor:
    """Denormalise a target column from [0, 1] to physical units."""
    lo, hi = TARGET_RANGES[col]
    phys = y_norm * (hi - lo) + lo
    if col in LOG1P_TARGET_COLS:
        phys = torch.expm1(phys)
    return phys

class BallisticPINN(nn.Module):
    """Simple fully‑connected PINN (used for baseline comparison)."""
    def __init__(self, num_materials=NUM_MATERIALS, embedding_dim=EMBEDDING_DIM,
                 hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS):
        super().__init__()
        self.num_continuous = 5
        self.mat_embed = nn.Embedding(num_embeddings=num_materials, embedding_dim=embedding_dim)
        input_dim = self.num_continuous + embedding_dim
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 5))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        cont = x[:, [0, 1, 2, 4, 5]]
        mat_ids = x[:, 3].long()
        embed = self.mat_embed(mat_ids)
        return self.net(torch.cat([cont, embed], dim=1))

# -------------------------------------------------------------------
# Physics loss computation (returns loss and component magnitudes)
# -------------------------------------------------------------------

def compute_physics_loss(preds_norm, x_mm, batch_x, mat_tensors, device):
    V_norm = preds_norm[:, 0:1]
    sigma_norm = preds_norm[:, 1:2]
    T_norm = preds_norm[:, 2:3]
    eps_norm = preds_norm[:, 3:4]
    damage_norm = preds_norm[:, 4:5]

    # Denormalise to physical units
    V = denorm_target(V_norm, "V_ms")
    sigma = denorm_target(sigma_norm, "sigma_MPa")
    T = denorm_target(T_norm, "T_K")
    eps = denorm_target(eps_norm, "strain")
    damage = denorm_target(damage_norm, "damage")

    # x_mm → x_m for momentum residual (input column 5 is normalised mm range)
    x_m = batch_x[:, 5:6] * 0.1  # [0, 1] → [0, 0.1] m

    # Spatial derivatives via autograd
    ones = torch.ones_like(V)
    dV_dx = torch.autograd.grad(V, x_mm, grad_outputs=ones, create_graph=True)[0]
    dT_dx = torch.autograd.grad(T, x_mm, grad_outputs=ones, create_graph=True)[0]
    deps_dx = torch.autograd.grad(eps, x_mm, grad_outputs=ones, create_graph=True)[0]
    # Convert derivative units: per‑norm‑x → per‑metre (scale = 10)
    scale = 10.0
    dV_dx = dV_dx * scale
    dT_dx = dT_dx * scale
    deps_dx = deps_dx * scale

    # Material property lookup (per sample)
    mat_ids = batch_x[:, 3].long()
    A = mat_tensors["A"][mat_ids].unsqueeze(1)
    B = mat_tensors["B"][mat_ids].unsqueeze(1)
    n = mat_tensors["n"][mat_ids].unsqueeze(1)
    C = mat_tensors["C"][mat_ids].unsqueeze(1)
    m = mat_tensors["m"][mat_ids].unsqueeze(1)
    Tmelt = mat_tensors["T_melt"][mat_ids].unsqueeze(1)
    rho = mat_tensors["rho"][mat_ids].unsqueeze(1)
    Cp = mat_tensors["Cp"][mat_ids].unsqueeze(1)

    # Strain rate from velocity (effective stress target → V)
    eps_dot = V / (SHEAR_ZONE_WIDTH_RATIO * D_BULLET_M * SQRT3)

    # JC constitutive residual (compare to sigma_JC·(1‑D))
    res_jc = jc_constitutive_residual_torch(
        sigma, eps, eps_dot, T, damage, A, B, n, C, m, Tmelt
    )

    # Plate thickness per sample (input column 0)
    lo_t, hi_t = INPUT_RANGES["thickness_mm"]
    t_plate_m = (batch_x[:, 0:1] * (hi_t - lo_t) + lo_t) / 1000.0

    # Momentum residual
    res_mom = momentum_residual_torch(
        x_m, V, dV_dx, sigma, damage,
        D_BULLET_M, t_plate_m, M_BULLET_KG
    )

    # Strain residual
    res_strain = strain_residual_torch(deps_dx, D_BULLET_M)

    # Energy residual
    res_energy = energy_residual_torch(
        dT_dx, sigma, damage, deps_dx, rho, Cp
    )

    # Non‑dimensionalise each term
    r_jc = res_jc / SCALE_STRESS_MPA
    r_mom = res_mom / SCALE_FORCE_N
    r_strain = res_strain / SCALE_STRAIN_GRAD
    r_energy = res_energy / SCALE_TEMP_GRAD

    loss_phys = (
        LAMBDA_JC * torch.mean(r_jc ** 2) +
        LAMBDA_MOMENTUM * torch.mean(r_mom ** 2) +
        LAMBDA_STRAIN * torch.mean(r_strain ** 2) +
        LAMBDA_ENERGY * torch.mean(r_energy ** 2)
    )

    components = {
        "jc": torch.mean(r_jc ** 2).item(),
        "momentum": torch.mean(r_mom ** 2).item(),
        "strain": torch.mean(r_strain ** 2).item(),
        "energy": torch.mean(r_energy ** 2).item(),
    }
    return loss_phys, components

# -------------------------------------------------------------------
# Main training routine
# -------------------------------------------------------------------

def main():
    """Run this module command-line workflow."""
    parser = argparse.ArgumentParser(description="Train a PINN for ballistic impact.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate.")
    parser.add_argument("--arch", type=str, default="hard_pinn", choices=["hard_pinn", "resmlp", "fcnn"], help="Network architecture.")
    parser.add_argument("--save_prefix", type=str, default="pinn", help="Checkpoint filename prefix.")
    parser.add_argument("--warmstart", type=str, default="", help="Path to pretrained checkpoint.")
    parser.add_argument("--adaptive_weighting", action="store_true", help="Enable simple adaptive loss weighting (balance data vs physics).")
    parser.add_argument("--lbfgs_finetune", action="store_true", help="Run a few L‑BFGS steps after SGD.")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    train_dir = "pinn_data/train"
    val_dir = "pinn_data/val"
    if not os.path.isdir(train_dir):
        raise RuntimeError("Training data not found – generate PINN data first.")

    train_ds = BallisticPINNDataset(train_dir, normalize=True)
    val_ds = BallisticPINNDataset(val_dir, normalize=True)
    print(f"Train samples: {len(train_ds):,}, Val samples: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda")

    # Model selection
    if args.arch == "hard_pinn":
        model = BallisticHardPINN(num_materials=NUM_MATERIALS, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, n_blocks=3).to(device)
    elif args.arch == "resmlp":
        model = BallisticResMLP(num_materials=NUM_MATERIALS, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, n_blocks=3).to(device)
    else:
        model = BallisticPINN().to(device)

    # Warm‑start if requested
    if args.warmstart and os.path.exists(args.warmstart):
        try:
            model.load_state_dict(torch.load(args.warmstart, map_location=device))
            print(f"Loaded pretrained weights from {args.warmstart}")
        except Exception as e:
            print(f"Warm‑start failed ({e}) – initializing from scratch.")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Material tensors on device for physics loss
    mat_tensors = {
        "A": MAT_A.to(device),
        "B": MAT_B.to(device),
        "n": MAT_N.to(device),
        "C": MAT_C_JC.to(device),
        "m": MAT_M.to(device),
        "T_melt": MAT_TMELT.to(device),
        "rho": MAT_RHO.to(device),
        "Cp": MAT_CP.to(device),
    }

    # Logging setup
    log_dir = os.path.join("results/reports", "training_logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(log_dir, f"training_{args.arch}_{timestamp}.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "data_loss", "phys_loss", "total_loss", "lr", "grad_norm", "val_mse"])

    best_val = float("inf")
    os.makedirs("models", exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_data = epoch_phys = epoch_total = epoch_grad_norm = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            # Autograd on x_mm (column 5)
            x_mm = batch_x[:, 5:6].clone().detach().requires_grad_(True)
            batch_x_adj = torch.cat([batch_x[:, :5], x_mm], dim=1)
            preds_norm = model(batch_x_adj)

            loss_data = criterion(preds_norm, batch_y)
            loss_phys, comps = compute_physics_loss(preds_norm, x_mm, batch_x, mat_tensors, device)

            # Simple adaptive weighting (balance scales)
            if args.adaptive_weighting:
                scale = loss_phys.item() / (loss_data.item() + 1e-12)
                scale = max(0.1, min(scale, 10.0))
                lambda_data = scale
            else:
                lambda_data = 1.0

            loss_total = lambda_data * loss_data + loss_phys
            loss_total.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            # Record gradient norm (L2 across all parameters)
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm(2).item() ** 2
            epoch_grad_norm += (total_norm ** 0.5) * batch_x.size(0)

            optimizer.step()

            n = batch_x.size(0)
            epoch_data += loss_data.item() * n
            epoch_phys += loss_phys.item() * n
            epoch_total += loss_total.item() * n

        # End‑of‑epoch statistics
        N = len(train_ds)
        epoch_data /= N
        epoch_phys /= N
        epoch_total /= N
        epoch_grad_norm /= N
        scheduler.step()

        # Validation (data MSE only)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                val_pred = model(batch_x)
                val_loss += criterion(val_pred, batch_y).item() * batch_x.size(0)
        val_loss /= len(val_ds)

        # Log epoch metrics
        csv_writer.writerow([
            epoch + 1,
            f"{epoch_data:.6f}",
            f"{epoch_phys:.6f}",
            f"{epoch_total:.6f}",
            f"{optimizer.param_groups[0]['lr']:.6e}",
            f"{epoch_grad_norm:.6f}",
            f"{val_loss:.6f}"
        ])
        csv_file.flush()

        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Data {epoch_data:.6f} | Phys {epoch_phys:.6f} | Total {epoch_total:.6f} | Val {val_loss:.6f} | GradNorm {epoch_grad_norm:.3f}")

        # Checkpointing
        prefix = args.save_prefix if args.save_prefix else args.arch
        ckpt_best = f"models/{prefix}_best.pt"
        ckpt_final = f"models/{prefix}_final.pt"
        meta_path = f"models/{prefix}_meta.json"
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_best)
            if prefix == "pinn":
                torch.save(model.state_dict(), "models/pinn_best.pt")

    # Save final checkpoint
    torch.save(model.state_dict(), ckpt_final)
    if prefix == "pinn":
        torch.save(model.state_dict(), "models/pinn_final.pt")

    # Optional L‑BFGS fine‑tuning
    if args.lbfgs_finetune:
        print("\nStarting L‑BFGS fine‑tuning (5 iterations)…")
        if os.path.exists(ckpt_best):
            model.load_state_dict(torch.load(ckpt_best, map_location=device))
        else:
            print("Best checkpoint not found – skipping L‑BFGS.")
        lbfgs_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        optimizer_lbfgs = optim.LBFGS(model.parameters(), max_iter=5, line_search_fn="strong_wolfe")
        for i, (batch_x, batch_y) in enumerate(lbfgs_loader):
            if i >= 3:
                break
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            def closure():
                optimizer_lbfgs.zero_grad()
                x_mm = batch_x[:, 5:6].clone().detach().requires_grad_(True)
                batch_x_adj = torch.cat([batch_x[:, :5], x_mm], dim=1)
                preds_norm = model(batch_x_adj)
                loss_data = criterion(preds_norm, batch_y)
                loss_phys, _ = compute_physics_loss(preds_norm, x_mm, batch_x, mat_tensors, device)
                loss = loss_data + loss_phys
                loss.backward()
                return loss
            optimizer_lbfgs.step(closure)
        ckpt_lbfgs = f"models/{prefix}_lbfgs.pt"
        torch.save(model.state_dict(), ckpt_lbfgs)
        print(f"L‑BFGS fine‑tuned checkpoint saved to {ckpt_lbfgs}")

    # Save training meta‑data JSON
    meta = {
        "arch": args.arch,
        "input_cols": INPUT_COLS,
        "target_cols": TARGET_COLS,
        "input_ranges": {k: list(v) for k, v in INPUT_RANGES.items()},
        "target_ranges": {k: list(v) for k, v in TARGET_RANGES.items()},
        "log_input_cols": list(LOG_INPUT_COLS),
        "log1p_target_cols": list(LOG1P_TARGET_COLS),
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "n_layers": N_LAYERS if args.arch == "fcnn" else None,
        "n_blocks": 3 if args.arch != "fcnn" else None,
        "num_materials": NUM_MATERIALS,
        "best_val_mse": best_val,
        "adaptive_weighting": args.adaptive_weighting,
        "lbfgs_finetune": args.lbfgs_finetune,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    csv_file.close()
    print(f"\nTraining finished – log written to {csv_path}\nBest Val MSE: {best_val:.6f}")
    print(f"Saved: {ckpt_best}, {ckpt_final}, {meta_path}")

if __name__ == "__main__":
    main()
