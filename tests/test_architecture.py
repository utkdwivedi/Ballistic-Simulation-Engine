"""
Unit tests for production ResMLP architecture and LayerNorm sample-independence.
"""
import torch
import pytest
from models.architecture import BallisticResMLP


def test_layernorm_sample_independence():
    """
    Verify that LayerNorm ensures sample independence in autograd spatial derivatives.
    With LayerNorm, d(y_0)/d(x_0) must be identical whether evaluated alone
    or alongside 63 other samples in a batch.
    """
    torch.manual_seed(42)
    model = BallisticResMLP()
    model.eval()

    # Single sample
    x_single = torch.rand(1, 6, requires_grad=True)
    out_single = model(x_single)
    # Compute derivative of V w.r.t x_mm
    grad_single = torch.autograd.grad(
        out_single[:, 0], x_single,
        grad_outputs=torch.ones(1),
        create_graph=False
    )[0]

    # Same sample embedded in a batch of 64
    x_others = torch.rand(63, 6)
    x_batch = torch.cat([x_single.detach(), x_others], dim=0).requires_grad_(True)
    out_batch = model(x_batch)
    grad_batch = torch.autograd.grad(
        out_batch[:, 0], x_batch,
        grad_outputs=torch.ones(64),
        create_graph=False
    )[0]

    # Verify sample 0 derivative is identical to within floating-point precision
    diff = torch.abs(grad_single[0, 5] - grad_batch[0, 5]).item()
    assert diff < 1e-5, f"LayerNorm violated sample independence: diff={diff}"


def test_model_forward_shape_and_no_nan():
    """Verify ResMLP forward pass shapes and numerical validity."""
    model = BallisticResMLP()
    x = torch.rand(32, 6)
    # Set material_id as valid discrete indices in [0, 4]
    x[:, 3] = torch.randint(0, 5, (32,)).float()

    out = model(x)
    assert out.shape == (32, 5), f"Expected shape (32, 5), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN found in model output"
    assert not torch.isinf(out).any(), "Inf found in model output"


def test_residual_gradient_flow():
    """Verify that gradients propagate cleanly through residual blocks without vanishing."""
    model = BallisticResMLP()
    x = torch.rand(16, 6)
    x[:, 3] = torch.randint(0, 5, (16,)).float()
    out = model(x)
    loss = out.sum()
    loss.backward()

    # Check input projection grad norm vs out projection grad norm
    in_grad_norm = model.in_proj.weight.grad.norm().item()
    out_grad_norm = model.out_proj.weight.grad.norm().item()

    assert in_grad_norm > 1e-6, "Input layer gradient vanished"
    assert out_grad_norm > 1e-6, "Output layer gradient vanished"
    ratio = in_grad_norm / (out_grad_norm + 1e-12)
    assert 0.01 <= ratio <= 100.0, f"Unbalanced gradient flow: ratio={ratio}"


def test_hard_boundary_ansatz_exactness():
    """
    Verify Task 4.1: BallisticHardPINN must guarantee exact boundary conditions at x = 0:
      V(0) = V_0
      T(0) = T_plate
      strain(0) = 0
      damage(0) = 0
    to within machine precision.
    """
    from models.architecture import BallisticHardPINN
    torch.manual_seed(42)
    model = BallisticHardPINN()
    model.eval()

    # Generate 100 random configurations at surface x = 0
    N = 100
    x = torch.rand(N, 6)
    x[:, 3] = torch.randint(0, 5, (N,)).float()
    x[:, 5] = 0.0  # Exactly at impact surface x = 0

    with torch.no_grad():
        out = model(x)

    # Compute expected initial conditions in normalized coordinates:
    log10_v0 = x[:, 1:2] * (3.477 - 1.699) + 1.699
    expected_v0_norm = torch.pow(10.0, log10_v0) / 3000.0

    T_plate_phys = x[:, 4:5] * (500.0 - 200.0) + 200.0
    expected_T_norm = (T_plate_phys - 200.0) / 2300.0

    # 1. V(0) == V_0
    v_diff = torch.max(torch.abs(out[:, 0:1] - expected_v0_norm)).item()
    assert v_diff < 1e-6, f"V(0) != V_0 at impact surface: max_diff={v_diff}"

    # 2. T(0) == T_plate
    t_diff = torch.max(torch.abs(out[:, 2:3] - expected_T_norm)).item()
    assert t_diff < 1e-6, f"T(0) != T_plate at impact surface: max_diff={t_diff}"

    # 3. strain(0) == 0.0
    strain_max = torch.max(torch.abs(out[:, 3:4])).item()
    assert strain_max < 1e-6, f"strain(0) != 0 at impact surface: max={strain_max}"

    # 4. damage(0) == 0.0
    damage_max = torch.max(torch.abs(out[:, 4:5])).item()
    assert damage_max < 1e-6, f"damage(0) != 0 at impact surface: max={damage_max}"


def test_hard_pinn_damage_reachability():
    """
    Verify that damage can reach full 1.0 (perforation/plugging) even at small penetration
    depths (e.g. x_norm = 0.05), preventing artificial capacity capping.
    """
    from models.architecture import BallisticHardPINN
    model = BallisticHardPINN()
    x = torch.zeros(1, 6)
    x[0, 5] = 0.05  # 5 mm depth

    # Override backbone to output positive damage signal
    with torch.no_grad():
        model.backbone.out_proj.bias[4] = 30.0  # large positive signal
        out = model(x)
        assert out[0, 4].item() >= 0.99, f"Damage failed to reach 1.0 at small x: {out[0, 4].item()}"


def test_hard_pinn_no_dead_relu_in_sigma():
    """
    Verify that flow stress maintains non-zero gradient even when raw pre-activations
    are negative (preventing dead ReLU trap).
    """
    from models.architecture import BallisticHardPINN
    model = BallisticHardPINN()
    x = torch.rand(1, 6)
    with torch.no_grad():
        model.backbone.out_proj.bias[1] = -10.0  # strongly negative

    out = model(x)
    sigma = out[0, 1]
    assert sigma > 0.0, "Flow stress must be strictly positive"
    sigma.backward()
    grad = model.backbone.out_proj.bias.grad
    assert grad[1].abs() > 0.0, "Flow stress gradient must not vanish (no dead ReLU)"


