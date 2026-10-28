"""
Ballistic Neural Surrogate Architecture (Production ResMLP)
===========================================================

Key design choices:
1. LayerNorm: sample-independent normalization to ensure uncoupled spatial derivatives via autograd.
2. Residual skip connections: stable gradient flow across deep hidden layers.
3. GELU activation: smooth non-linear continuous representations.
4. Categorical embedding: maps 5 materials into an 8-dimensional continuous latent space.
"""
import torch
import torch.nn as nn

from pinn_dataloader import INPUT_RANGES, TARGET_RANGES


class ResBlock(nn.Module):
    """Pre-activation Residual Block with LayerNorm and GELU."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return x + self.net(x)


class BallisticResMLP(nn.Module):
    """
    Production Neural Surrogate Backbone for transient penetration mechanics.
    Input: [h, log10(V0), theta, mat_id, T_plate, x_mm] (normalized)
    Output: [V, sigma, T, strain, damage] (normalized [0, 1])
    """
    def __init__(self, num_materials: int = 5, embedding_dim: int = 8,
                 hidden_dim: int = 256, n_blocks: int = 3):
        super().__init__()
        self.num_continuous = 5
        self.mat_embed = nn.Embedding(num_embeddings=num_materials, embedding_dim=embedding_dim)
        
        in_dim = self.num_continuous + embedding_dim
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(n_blocks)])
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, 5)

    def forward(self, x):
        # x shape: [Batch, 6]
        cont = x[:, [0, 1, 2, 4, 5]]
        mat_id = x[:, 3].long()
        embed = self.mat_embed(mat_id)
        h = torch.cat([cont, embed], dim=1)
        
        h = self.in_proj(h)
        for block in self.blocks:
            h = block(h)
            
        h = self.final_norm(h)
        return self.out_proj(h)


class BallisticHardPINN(nn.Module):
    """
    Hard-Constrained Physics-Informed Neural Network (Phase 4).
    Algebraically guarantees exact impact surface boundary conditions at x = 0:
      1. V(0) = V_0 (initial projectile velocity)
      2. T(0) = T_plate (initial ambient temperature)
      3. strain(0) = 0 (zero initial plastic strain)
      4. damage(0) = 0 (undamaged target plate at contact)
    And strictly enforces physical range non-negativity and bounds:
      - V(x) in [0, V_0]
      - sigma(x) >= 0
      - T(x) in [T_plate, 1.0]
      - strain(x) >= 0
      - damage(x) in [0, 1.0]
    """
    def __init__(self, num_materials: int = 5, embedding_dim: int = 8,
                 hidden_dim: int = 256, n_blocks: int = 3):
        super().__init__()
        self.backbone = BallisticResMLP(
            num_materials=num_materials,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks
        )
        # Derive normalization buffers from INPUT_RANGES / TARGET_RANGES so the
        # hard-PINN ansatz stays consistent with the dataloader normalization.
        v0_lo, v0_hi = INPUT_RANGES["velocity_ms"]       # log10 scale
        self.register_buffer("v0_log_min", torch.tensor(float(v0_lo), dtype=torch.float32))
        self.register_buffer("v0_log_span", torch.tensor(float(v0_hi - v0_lo), dtype=torch.float32))

        v_max_lo, v_max_hi = TARGET_RANGES["V_ms"]
        self.register_buffer("v_max_target", torch.tensor(float(v_max_hi), dtype=torch.float32))

        tp_lo, tp_hi = INPUT_RANGES["plate_temperature_K"]
        self.register_buffer("t_plate_min", torch.tensor(float(tp_lo), dtype=torch.float32))
        self.register_buffer("t_plate_span", torch.tensor(float(tp_hi - tp_lo), dtype=torch.float32))

        tt_lo, tt_hi = TARGET_RANGES["T_K"]
        self.register_buffer("t_target_min", torch.tensor(float(tt_lo), dtype=torch.float32))
        self.register_buffer("t_target_span", torch.tensor(float(tt_hi - tt_lo), dtype=torch.float32))

    def forward(self, x):
        # x: [Batch, 6] in normalized coordinates:
        # [thickness_norm, log10_v0_norm, obliquity_norm, mat_id, temp_norm, x_norm]
        raw_pred = self.backbone(x)
        x_norm = x[:, 5:6]  # Penetration coordinate (0 at surface)

        # Reconstruct normalized initial velocity V_0 in target scale [0, 3000 m/s]
        log10_v0 = x[:, 1:2] * self.v0_log_span + self.v0_log_min
        v0_phys = torch.pow(10.0, log10_v0)
        v0_target_norm = v0_phys / self.v_max_target

        # Reconstruct normalized ambient plate temperature T_plate in target scale [200, 2500 K]
        T_plate_phys = x[:, 4:5] * self.t_plate_span + self.t_plate_min
        T_plate_target_norm = (T_plate_phys - self.t_target_min) / self.t_target_span

        # Exact boundary ansatz transformations with physical bounds clamping:
        # V(x) = clamp(V_0 - x * |f_V(x)|, min=0.0): velocity strictly starts at V_0 and decelerates
        V_pred = torch.clamp(v0_target_norm - x_norm * torch.abs(raw_pred[:, 0:1]), min=0.0)

        # Flow stress: strictly non-negative yield response via smooth softplus (prevents dead ReLU)
        sigma_pred = nn.functional.softplus(raw_pred[:, 1:2])

        # T(x) = clamp(T_plate + x * |f_T(x)|, min=0.0, max=1.0): temperature strictly starts at T_plate and heats
        T_pred = torch.clamp(T_plate_target_norm + x_norm * torch.abs(raw_pred[:, 2:3]), min=0.0, max=1.0)

        # strain(x) = x * |f_strain(x)|: strain strictly starts at 0 and cannot be negative
        strain_pred = x_norm * torch.abs(raw_pred[:, 3:4])

        # damage(x) = clamp(x * |f_damage(x)|, 0.0, 1.0): damage strictly starts at 0 and is bounded in [0, 1]
        damage_pred = torch.clamp(x_norm * torch.abs(raw_pred[:, 4:5]), min=0.0, max=1.0)

        return torch.cat([V_pred, sigma_pred, T_pred, strain_pred, damage_pred], dim=1)

