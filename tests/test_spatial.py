from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from reread.config import ReReadConfig
from reread.memory.spatial import pool_video_tokens, video_token_metadata


def test_pool_camera_isolation_row_major_and_detached_storage():
    c = replace(ReReadConfig(), kv_dim=2)
    grid = torch.zeros(2, 12, 10, 2)
    grid[:, :8] = 1
    grid[:, 8:, :5] = 10
    grid[:, 8:, 5:] = 100
    x = grid.flatten(1, 2).requires_grad_()
    result = pool_video_tokens(x, c)
    assert result.shape == (2, 64, 2)
    assert torch.equal(result[:, :32], torch.ones(2, 32, 2))
    assert torch.equal(result[:, 32:48], torch.full((2, 16, 2), 10.0))
    assert torch.equal(result[:, 48:], torch.full((2, 16, 2), 100.0))
    assert not result.requires_grad and result.grad_fn is None
    assert result.device == x.device and result.dtype == x.dtype
    with torch.no_grad():
        x.fill_(500)
    assert result[0, 0, 0] == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_pool_matches_three_independent_adaptive_2d_calls(dtype):
    c = replace(ReReadConfig(), kv_dim=3)
    x = torch.arange(2 * 120 * 3, dtype=dtype).reshape(2, 120, 3)
    grid = x.reshape(2, 12, 10, 3)
    regions = (grid[:, :8], grid[:, 8:, :5], grid[:, 8:, 5:])
    expected = torch.cat(
        [
            F.adaptive_avg_pool2d(r.permute(0, 3, 1, 2).float(), p)
            .flatten(2)
            .transpose(1, 2)
            for r, p in zip(regions, c.camera_pools)
        ],
        1,
    ).to(dtype)
    result = pool_video_tokens(x, c)
    assert torch.equal(result, expected)
    assert result.dtype == dtype


def test_metadata_exact_adaptive_bin_centers_not_output_linspace():
    cameras, xy = video_token_metadata()
    assert cameras.shape == (64,) and cameras.dtype == torch.long
    assert cameras.tolist() == [0] * 32 + [1] * 16 + [2] * 16

    torch.testing.assert_close(
        xy[:8, 0], torch.tensor([-0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8])
    )
    torch.testing.assert_close(xy[:8, 1], torch.full((8,), -0.75))

    torch.testing.assert_close(xy[32:36, 0], torch.tensor([-0.6, -0.2, 0.2, 0.6]))
    torch.testing.assert_close(xy[32:36, 1], torch.full((4,), -0.75))
    assert torch.equal(xy[32:48], xy[48:64])
    assert not xy.requires_grad and (xy.abs() <= 1).all()


def test_metadata_dtype_and_invalid_input_contracts():
    assert video_token_metadata(dtype=torch.float64)[1].dtype == torch.float64
    with pytest.raises(ValueError, match="floating"):
        video_token_metadata(dtype=torch.long)
    for x in (
        torch.zeros(1, 121, 3072),
        torch.zeros(1, 120, 4),
        torch.zeros(120, 3072),
        torch.zeros(1, 120, 3072, dtype=torch.long),
    ):
        with pytest.raises(ValueError):
            pool_video_tokens(x)
