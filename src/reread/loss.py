import math

import torch
import torch.distributed as dist


def action_mse(prediction, target, valid, config, *, distributed=False, group=None):
    if (
        prediction.ndim != 3
        or prediction.shape != target.shape
        or prediction.shape[1:] != (config.horizon, config.action_dim)
        or valid.shape != prediction.shape[:2]
        or valid.dtype != torch.bool
    ):
        raise ValueError("Prediction/target must be [B,H,D], valid boolean [B,H]")
    mask = valid.to(device=prediction.device)
    predicted = prediction.float().masked_fill(~mask[..., None], 0)
    expected = target.detach().to(device=prediction.device, dtype=torch.float32)
    expected = expected.masked_fill(~mask[..., None], 0)
    squared_error = (predicted - expected).square().sum()
    count = mask.sum(dtype=torch.int64) * config.action_dim
    invalid = ~(torch.isfinite(predicted).all() & torch.isfinite(expected).all())
    stats = torch.stack(
        (squared_error.detach().double(), count.double(), invalid.double())
    )
    world_size = 1
    if distributed:
        if not dist.is_initialized():
            raise RuntimeError("Distributed loss requires an initialized process group")
        dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=group)
        world_size = dist.get_world_size(group)
    total_error, total_count, invalid_count = stats.tolist()
    if invalid_count or not math.isfinite(total_error):
        raise FloatingPointError("Nonfinite active supervision on at least one rank")
    if total_count < 1:
        raise ValueError("No valid action labels across all ranks")
    loss = predicted.sum() * 0 + squared_error * (world_size / total_count)
    mse = total_error / total_count
    return loss, {"mse": mse, "total_loss": mse, "label_elements": int(total_count)}
