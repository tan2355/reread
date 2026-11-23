import math

import torch
from torch.nn import functional as F

from reread.config import ReReadConfig


def _camera_regions(config):
    h, w = config.video_grid
    return (
        (slice(0, config.head_rows), slice(0, w)),
        (slice(config.head_rows, h), slice(0, config.wrist_split)),
        (slice(config.head_rows, h), slice(config.wrist_split, w)),
    )


def pool_video_tokens(tensor, config=ReReadConfig()):
    h, w = config.video_grid
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.ndim != 3
        or tensor.shape[0] < 1
        or tensor.shape[1] != h * w
        or tensor.shape[2] != config.kv_dim
        or not tensor.is_floating_point()
    ):
        raise ValueError(f"Video content must be floating [B,{h * w},{config.kv_dim}]")
    grid = tensor.detach().reshape(tensor.shape[0], h, w, tensor.shape[2])
    pieces = []
    for (rows, cols), output_size in zip(_camera_regions(config), config.camera_pools):
        camera = grid[:, rows, cols, :].permute(0, 3, 1, 2).float()
        pooled = F.adaptive_avg_pool2d(camera, output_size)
        pieces.append(pooled.flatten(2).transpose(1, 2))
    return torch.cat(pieces, dim=1).to(tensor.dtype).clone()


def _bin_centers(source_size, output_size):
    centers = []
    for i in range(output_size):
        start = math.floor(i * source_size / output_size)
        end = math.ceil((i + 1) * source_size / output_size)
        centers.append((start + end) / source_size - 1.0)
    return centers


def video_token_metadata(config=ReReadConfig(), *, device=None, dtype=torch.float32):
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError("Spatial coordinates require floating dtype")
    h, w = config.video_grid
    sizes = (
        (config.head_rows, w),
        (h - config.head_rows, config.wrist_split),
        (h - config.head_rows, w - config.wrist_split),
    )
    cameras, positions = [], []
    for camera, (source_h, source_w), (out_h, out_w) in zip(
        range(3), sizes, config.camera_pools
    ):
        xs, ys = _bin_centers(source_w, out_w), _bin_centers(source_h, out_h)
        for y in ys:
            for x in xs:
                cameras.append(camera)
                positions.append((x, y))
    return (
        torch.tensor(cameras, device=device, dtype=torch.long),
        torch.tensor(positions, device=device, dtype=dtype),
    )
