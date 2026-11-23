from dataclasses import replace

import pytest
import torch

from reread.config import ReReadConfig
from reread.loss import action_mse


def config():
    return replace(ReReadConfig(), horizon=4, action_tokens=4, decision_stride=2)


def test_masked_mse_matches_manual_loss_and_gradients():
    torch.manual_seed(7)
    prediction = torch.randn(2, 4, 14, requires_grad=True)
    target = torch.randn_like(prediction, requires_grad=True)
    valid = torch.tensor([[True, False, True, False], [True, True, True, False]])
    with torch.no_grad():
        target[~valid] = float("nan")
    loss, metrics = action_mse(prediction, target, valid, config())
    reference = (prediction[valid] - target.detach()[valid]).square().mean()
    torch.testing.assert_close(loss, reference)
    loss.backward()
    expected = torch.zeros_like(prediction)
    expected[valid] = (
        2 * (prediction.detach()[valid] - target.detach()[valid]) / (5 * 14)
    )
    torch.testing.assert_close(prediction.grad, expected)
    assert target.grad is None
    assert metrics["label_elements"] == 70
    assert metrics["mse"] == metrics["total_loss"] == pytest.approx(reference.item())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_loss_accumulates_in_float32(dtype):
    prediction = torch.ones(1, 4, 14, dtype=dtype, requires_grad=True)
    loss, metrics = action_mse(
        prediction,
        torch.zeros_like(prediction),
        torch.ones(1, 4, dtype=torch.bool),
        config(),
    )
    assert loss.dtype == torch.float32
    assert loss.item() == metrics["mse"] == 1
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["prediction", "target"])
def test_nonfinite_active_values_are_rejected(field, bad):
    prediction, target = torch.zeros(1, 4, 14), torch.zeros(1, 4, 14)
    value = prediction if field == "prediction" else target
    value[0, 0, 0] = bad
    with pytest.raises(FloatingPointError):
        action_mse(prediction, target, torch.ones(1, 4, dtype=torch.bool), config())


def test_empty_supervision_is_rejected():
    with pytest.raises(ValueError, match="No valid"):
        action_mse(
            torch.zeros(1, 4, 14),
            torch.zeros(1, 4, 14),
            torch.zeros(1, 4, dtype=torch.bool),
            config(),
        )
