import argparse
import json
from dataclasses import replace

import torch

from reread.config import ReReadConfig
from reread.model import ReRead


def main():
    parser = argparse.ArgumentParser(
        description="Run a small synthetic CPU correction and backward pass."
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    config = replace(
        ReReadConfig(),
        proprio_dim=3,
        horizon=4,
        action_tokens=4,
        video_tokens=6,
        kv_dim=12,
        layers=(1, 2),
        width=24,
        heads=4,
        depth=2,
        ff_dim=48,
        inference_steps=3,
        decision_stride=2,
        history_dropout=0,
        video_grid=(3, 4),
        head_rows=1,
        wrist_split=2,
        camera_pools=((1, 2), (1, 2), (1, 2)),
    )
    model = ReRead(config)
    candidate = torch.randn(2, 4, 14)
    memory = {
        "k": torch.randn(2, 3, 2, 10, 12),
        "v": torch.randn(2, 3, 2, 10, 12),
        "states": torch.randn(2, 3, 3),
        "valid": torch.ones(2, 3, dtype=torch.bool),
        "timestamps": torch.tensor([[0, 2, 4], [0, 2, 4]]),
        "executed_steps": torch.tensor([[2, 2, 0], [2, 2, 0]]),
    }
    corrected = model(candidate, memory)
    torch.testing.assert_close(corrected, candidate, atol=0, rtol=0)
    loss = model.loss(candidate, memory, candidate + 0.1)
    loss.backward()
    assert torch.isfinite(loss) and model.output.weight.grad.abs().sum() > 0
    print(
        json.dumps(
            {
                "shape": list(corrected.shape),
                "loss": float(loss.detach()),
                "finite_gradients": True,
            }
        )
    )


if __name__ == "__main__":
    main()
