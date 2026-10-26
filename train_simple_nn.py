import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pinn_dataloader import BallisticPINNDataset, INPUT_COLS, TARGET_COLS, INPUT_RANGES, TARGET_RANGES, LOG_INPUT_COLS, LOG1P_TARGET_COLS
from stage1_part1_johnson_cook import MATERIALS

from models.architecture import BallisticResMLP, BallisticHardPINN

# Hyperparameters
DEFAULT_EPOCHS = 30
BATCH_SIZE = 1024
LR = 3e-4
NUM_MATERIALS = len(MATERIALS)
EMBEDDING_DIM = 8
HIDDEN_DIM = 256
N_LAYERS = 6
GRAD_CLIP = 1.0


class BallisticFCNN(nn.Module):
    def __init__(self, num_materials=NUM_MATERIALS, embedding_dim=EMBEDDING_DIM,
                 hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS):
        super().__init__()
        self.num_continuous = 5  # thickness, velocity, obliquity, temp, x_mm
        self.mat_embed = nn.Embedding(
            num_embeddings=num_materials, embedding_dim=embedding_dim
        )

        input_dim = self.num_continuous + embedding_dim
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
            ]
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 5))  # 5 targets: V_ms, sigma_MPa, T_K, strain, damage
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: [Batch, 6]
        # features: thickness, velocity, obliquity, material_id, temp, x_mm
        cont_features = x[:, [0, 1, 2, 4, 5]]
        mat_ids = x[:, 3].long()

        embeds = self.mat_embed(mat_ids)
        combined = torch.cat([cont_features, embeds], dim=1)

        return self.net(combined)


def main():
    parser = argparse.ArgumentParser(description="Train Improved FCNN / ResMLP / HardPINN Supervised Baseline.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate.")
    parser.add_argument("--arch", type=str, default="hard_pinn", choices=["hard_pinn", "resmlp", "fcnn"], help="Architecture variant.")
    parser.add_argument("--save_prefix", type=str, default="", help="Prefix for checkpoint filenames.")
    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr

    print("=" * 60)
    print("  Phase 2: Training Improved FCNN (Supervised Baseline)")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load Data
    train_dir = "pinn_data/train"
    val_dir = "pinn_data/val"

    if not os.path.exists(train_dir):
        print("Could not find 'pinn_data/train'. Run 'python data_gen.py' first!")
        return

    train_ds = BallisticPINNDataset(train_dir, normalize=True)
    val_ds = BallisticPINNDataset(val_dir, normalize=True)

    print(f"Train samples: {len(train_ds):,}")
    print(f"Val samples:   {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda")
    )

    if args.arch == "hard_pinn":
        model = BallisticHardPINN(
            num_materials=NUM_MATERIALS, embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM, n_blocks=3
        ).to(device)
    elif args.arch == "resmlp":
        model = BallisticResMLP(
            num_materials=NUM_MATERIALS, embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM, n_blocks=3
        ).to(device)
    else:
        model = BallisticFCNN().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Architecture: {args.arch} | Model parameters: {total_params:,}")

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    os.makedirs("models", exist_ok=True)

    prefix = args.save_prefix if args.save_prefix else args.arch
    best_ckpt = f"models/{prefix}_best.pt"
    final_ckpt = f"models/{prefix}_final.pt"
    meta_path = f"models/{prefix}_meta.json"

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_ds)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * batch_x.size(0)

        val_loss /= len(val_ds)
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | LR: {current_lr:.2e}")

        if torch.isnan(torch.tensor(val_loss)):
            print("NaN detected in val_loss! Halting training.")
            break

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_ckpt)
            if prefix == "fcnn":
                torch.save(model.state_dict(), "models/fcnn_best.pt")

    # Save final model
    torch.save(model.state_dict(), final_ckpt)
    if prefix == "fcnn":
        torch.save(model.state_dict(), "models/fcnn_baseline.pt")
        torch.save(model.state_dict(), "models/fcnn_final.pt")

    # Save normalization metadata
    norm_meta = {
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
    }
    with open(meta_path, "w") as f:
        json.dump(norm_meta, f, indent=2)

    print(f"\nBest Val MSE: {best_val:.6f}")
    print(f"Saved: {best_ckpt}, {final_ckpt}, {meta_path}")


if __name__ == "__main__":
    main()
