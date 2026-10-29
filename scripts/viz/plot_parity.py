import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from infer import load_model
from torch.utils.data import DataLoader
from train_simple_nn import BallisticFCNN

from ballistic.dataloader import BallisticPINNDataset


def main():
    """Run this module command-line workflow."""
    parser = argparse.ArgumentParser(description="Generate Parity Plots (Pred vs True)")
    parser.add_argument("--model", type=str, default="models/pinn_production_best.pt", help="Path to model weights")
    parser.add_argument("--meta", type=str, default="models/pinn_production_meta.json", help="Path to metadata JSON")
    parser.add_argument("--save_path", type=str, default="results/plots/parity_plot.png", help="Path to save output plot")
    args = parser.parse_args()

    print("="*60)
    print("  Phase 5: Generating Parity Plots (Pred vs True)")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model
    meta_path = args.meta

    # Fallback to fcnn_baseline if production model not yet run
    if not os.path.exists(model_path):
        model_path = "models/fcnn_baseline.pt"
        meta_path = "models/fcnn_meta.json"

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Run training first.")
        return

    val_dir = "pinn_data/val"
    if not os.path.exists(val_dir):
        print("Validation data not found.")
        return

    ds = BallisticPINNDataset(val_dir, normalize=True)
    loader = DataLoader(ds, batch_size=2048, shuffle=False)

    if os.path.exists(meta_path):
        model, _ = load_model(model_path, meta_path)
    else:
        model = BallisticFCNN()
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))

    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_trues = []
    
    print("Running inference on validation set...")
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x)
            
            all_preds.append(preds.cpu().numpy())
            all_trues.append(batch_y.numpy())
            
    all_preds = np.vstack(all_preds)
    all_trues = np.vstack(all_trues)
    
    # Denormalize
    preds_phys = ds.denormalize_targets(all_preds)
    trues_phys = ds.denormalize_targets(all_trues)
    
    # Plotting
    target_names = ["Velocity (m/s)", "Stress (MPa)", "Temperature (K)", "Strain", "Damage"]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(5):
        ax = axes[i]
        true_val = trues_phys[:, i]
        pred_val = preds_phys[:, i]
        
        # Subsample for plotting speed if necessary
        idx = np.random.choice(len(true_val), min(10000, len(true_val)), replace=False)
        
        ax.scatter(true_val[idx], pred_val[idx], alpha=0.1, s=2)
        
        # Parity line
        min_val = min(true_val.min(), pred_val.min())
        max_val = max(true_val.max(), pred_val.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        ax.set_title(target_names[i])
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    axes[-1].axis('off') # Hide the empty 6th subplot
        
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(args.save_path, dpi=300, bbox_inches='tight')
    print(f"Saved parity plots to {args.save_path}")

if __name__ == "__main__":
    main()
