# TorchScript trace test
# Add this as a new test file so CI verifies the traced model matches eager output.

import random

import numpy as np
import torch

from ballistic.dataloader import INPUT_COLS
from ballistic.models.architecture import BallisticHardPINN


def test_traced_model_matches_eager():
    # Seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Load the existing trained model
    model = BallisticHardPINN()
    checkpoint = torch.load("models/hardpinn.pt", map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()

    # Create a random input tensor matching one training sample
    inp_np = np.random.uniform(0, 1, size=(1, len(INPUT_COLS)))
    inp_t = torch.from_numpy(inp_np.astype("float32"))

    with torch.no_grad():
        eager_out = model(inp_t)

    # Trace the model
    traced = torch.jit.trace(model, inp_t)
    with torch.no_grad():
        traced_out = traced(inp_t)

    # Compare outputs
    assert torch.allclose(eager_out, traced_out, atol=1e-4), "Traced model output differs from eager"
