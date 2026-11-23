import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from reread.data.demonstrations import (
    RawTrajectoryDataset,
    action_window,
    discover_episodes,
    iter_trajectory_batches,
    normalize_rgb_image,
    trajectory_requests,
)


class FakeNormalizer:
    def forward(self, batch):
        return {
            kind: {key: value / 10 for key, value in values.items()}
            for kind, values in batch.items()
        }


class FakeProcessor:
    shape_meta = {
        "state": [{"key": "default", "shape": 14}],
        "action": [{"key": "default", "shape": 14}],
    }
    normalizer = FakeNormalizer()

    def action_state_transform(self, batch):
        return batch


def write_raw(root, task="task_one", episode=0, length=43, broken_future=False):
    h5py = pytest.importorskip("h5py")
    directory = Path(root) / task / "aloha-agilex_clean_50"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"episode{episode}.hdf5"
    images = []
    for frame in range(length):
        stream = io.BytesIO()
        Image.fromarray(np.full((8, 10, 3), frame, np.uint8)).save(stream, format="PNG")
        images.append(stream.getvalue())
    if broken_future:
        images[1] = b"bad image not touched by a current-only sample"
    with h5py.File(path, "w") as file:
        file.create_dataset(
            "joint_action/vector",
            data=np.arange(length, dtype=np.float32)[:, None].repeat(14, 1),
        )
        for camera in ("head_camera", "left_camera", "right_camera"):
            file.create_dataset(
                f"observation/{camera}/rgb",
                data=np.array(images, dtype=f"S{max(map(len, images))}"),
            )
    path.with_suffix(".json").write_text(
        json.dumps(
            {"seen": ["instruction one", "instruction two"], "unseen": ["never used"]}
        )
    )
    return path


def test_next_joint_target_and_final_transition():
    indices, valid = action_window(40, 43)
    assert indices[:3].tolist() == [41, 42, 42]
    assert valid.sum() == 2 and not valid[2:].any()
    with pytest.raises(ValueError):
        action_window(42, 43)


def test_raw_current_only_image_and_processor(tmp_path):
    write_raw(tmp_path, broken_future=True)
    rows = discover_episodes(tmp_path, split="all", val_episodes=0)
    dataset = RawTrajectoryDataset(rows, FakeProcessor())
    sample = dataset[(0, 0, 0, 1)]
    assert sample["input_image"].shape == (3, 384, 320)
    assert (
        sample["input_image"].dtype == torch.uint8 and sample["input_image"].eq(0).all()
    )
    assert sample["proprio"].eq(0).all()
    assert sample["action"][0].eq(0.1).all()
    assert "instruction two" in sample["prompt"]
    assert sample["valid"].all()
    with pytest.raises(Exception):
        dataset[(0, 0, 1, 0)]
    dataset.close()


def test_holdout_is_episode_disjoint_and_randomized_opt_in(tmp_path):
    for episode in range(4):
        write_raw(tmp_path, episode=episode)
    train = discover_episodes(tmp_path, split="train", val_episodes=1)
    val = discover_episodes(tmp_path, split="val", val_episodes=1)
    assert len(train) == 3 and len(val) == 1
    assert not {r.id for r in train} & {r.id for r in val}
    assert all(r.phase == "clean" for r in train + val)
    assert discover_episodes(tmp_path, split="val", val_episodes=1) == val


def test_trajectory_lanes_keep_instruction_and_chronology(tmp_path):
    for episode in range(4):
        write_raw(tmp_path, episode=episode)
    rows = discover_episodes(tmp_path, split="all", val_episodes=0)
    batches = list(trajectory_requests(rows, 7, 2, 20))
    assert len(batches) == 6
    for batch in batches:
        assert len({request[1] for request in batch}) == len(batch)
    for lane in range(2):
        requests = [r for b in batches for r in b if r[0] == lane]
        for episode in {r[1] for r in requests}:
            subset = [r for r in requests if r[1] == episode]
            assert [r[2] for r in subset] == [0, 20, 40]
            assert len({r[3] for r in subset}) == 1
    dataset = RawTrajectoryDataset(rows, FakeProcessor())
    actual = list(iter_trajectory_batches(dataset, 7, 2, 20, max_batches=2))
    assert [s["decision_index"] for s in actual[1]] == [1, 1]
    dataset.close()


def test_missing_image_length_is_rejected(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = write_raw(tmp_path)
    with h5py.File(path, "a") as file:
        del file["observation/head_camera/rgb"]
        file.create_dataset(
            "observation/head_camera/rgb", data=np.zeros((2,), dtype="S4")
        )
    with pytest.raises(ValueError, match="length mismatch"):
        discover_episodes(tmp_path, split="all", val_episodes=0)


def test_parallel_loader_can_close_early(tmp_path):
    for episode in range(4):
        write_raw(tmp_path, episode=episode)
    rows = discover_episodes(tmp_path, split="all", val_episodes=0)
    dataset = RawTrajectoryDataset(rows, FakeProcessor())
    batches = iter_trajectory_batches(dataset, 4, 2, 20, workers=2)
    first = next(batches)
    assert len(first) == 2 and all(sample["frame"] == 0 for sample in first)
    batches.close()
    dataset.close()


def test_rgb_normalization_matches_official_order_at_every_uint8_level():
    image = (
        torch.arange(256, dtype=torch.uint8)
        .repeat(3 * 384 * 320 // 256)
        .reshape(3, 384, 320)
    )
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        official = image.to(device="cpu", dtype=dtype) * (2.0 / 255.0) - 1.0
        actual = normalize_rgb_image(image, device="cpu", dtype=dtype)
        assert torch.equal(actual, official)
    with pytest.raises(ValueError, match="uint8"):
        normalize_rgb_image(image.float(), device="cpu", dtype=torch.bfloat16)


def test_duplicate_task_selection_is_rejected_before_reading(tmp_path):
    with pytest.raises(ValueError, match="Task names"):
        discover_episodes(tmp_path, tasks=["same", "same"], split="all", val_episodes=0)
