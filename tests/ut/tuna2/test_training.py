"""CPU checks for image-understanding masking and true training restart."""
import os
os.environ['NON_MEGATRON'] = 'true'

import torch
from mindspeed_mm.models.tuna2.upstream.models.jit_utils import jit_x0_prediction_loss
from mindspeed_mm.models.tuna2.upstream.models._common import build_rope


def test_empty_generation_mask_has_finite_zero_gradient():
    prediction = torch.randn(1, 4, 3, requires_grad=True)
    loss = jit_x0_prediction_loss(prediction, torch.zeros_like(prediction),
                                  torch.ones(1), torch.zeros_like(prediction))
    assert torch.isfinite(loss) and loss.item() == 0
    loss.backward()
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))


def test_generation_mask_preserves_weighted_mse():
    prediction = torch.ones(1, 4, 3, requires_grad=True)
    mask = torch.zeros_like(prediction)
    mask[:, :2] = 1
    loss = jit_x0_prediction_loss(prediction, torch.zeros_like(prediction), torch.tensor([.5]), mask)
    assert loss.item() == 4
    loss.backward()
    assert torch.count_nonzero(prediction.grad[:, 2:]) == 0


def test_rope_supports_tiny_and_full_diffusion_heads():
    for head_dim in (16, 64):
        rope = build_rope([1, 32, 32], patch_size=16, attention_head_dim=head_dim)
        assert rope.shape == (4, head_dim // 2, 2, 2)
