import hashlib
import json

import numpy as np
import pytest
import torch

from reread.data.demonstrations import DEFAULT_PROMPT, PHASES, action_window
from reread.data.preprocessing import (
    CacheEpisode,
    augment_montage,
    augmentation_params,
    build_prompt_table,
    discover_cache_episodes,
    history_coverage,
    make_action_window,
    read_endposes,
    selected_frames,
    source_binding,
)


def write_episode(
    root, task="task_one", phase="clean", episode=0, *, nested=False, length=83
):
    h5py = pytest.importorskip("h5py")
    directory = root / task / PHASES[phase]
    location = directory / "data" if nested else directory
    location.mkdir(parents=True, exist_ok=True)
    path = location / f"episode{episode}.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "joint_action/vector", data=np.zeros((length, 14), np.float32)
        )
        for camera in ("head_camera", "left_camera", "right_camera"):
            handle.create_dataset(
                f"observation/{camera}/rgb", shape=(length,), dtype="S4"
            )
    instruction = (
        directory / "instructions" / f"episode{episode}.json"
        if nested
        else path.with_suffix(".json")
    )
    instruction.parent.mkdir(exist_ok=True)
    instruction.write_text(
        json.dumps(
            {
                "seen": ["Move bottle", "Move bottle"],
                "unseen": ["Please move bottle", "Move bottle"],
            }
        )
    )
    return path


def row(
    identifier="task_one/clean/episode0",
    seen=("Move bottle",),
    unseen=("Please move bottle",),
):
    return CacheEpisode(
        identifier,
        "/raw/episode0.hdf5",
        "/raw/episode0.json",
        "task_one",
        "clean",
        80,
        seen,
        unseen,
        "train",
    )


def identity_poses(length, *, quaternion=False):
    if quaternion:
        pose = np.zeros((length, 2, 7))
        pose[:, :, 3] = 1
        return pose
    return np.broadcast_to(np.eye(4), (length, 2, 4, 4)).copy()


def test_discover_both_layouts_and_whole_trajectory_split(tmp_path):
    for phase in PHASES:
        for episode in range(3):
            write_episode(
                tmp_path, phase=phase, episode=episode, nested=phase == "randomized"
            )
    rows = discover_cache_episodes(tmp_path)
    assert len(rows) == 6
    assert {r.phase for r in rows} == {"clean", "randomized"}
    assert sum(r.split == "val" for r in rows) == 2
    assert discover_cache_episodes(tmp_path) == rows
    assert all(r.observation_count == 82 for r in rows)
    assert all(
        r.instructions == r.seen and "Please move bottle" not in r.instructions
        for r in rows
    )
    assert all(CacheEpisode.from_dict(r.to_dict()) == r for r in rows)


def test_missing_randomized_is_not_silently_ignored(tmp_path):
    write_episode(tmp_path)
    with pytest.raises(ValueError, match="randomized"):
        discover_cache_episodes(tmp_path, val_episodes=0)
    assert (
        len(discover_cache_episodes(tmp_path, val_episodes=0, phases=("clean",))) == 1
    )


def test_discovery_rejects_ambiguous_layout_and_bad_instruction(tmp_path):
    write_episode(tmp_path)
    write_episode(tmp_path, nested=True)
    with pytest.raises(ValueError, match="Ambiguous"):
        discover_cache_episodes(tmp_path, val_episodes=0, phases=("clean",))
    other = tmp_path / "other"
    path = write_episode(other)
    path.with_suffix(".json").write_text('{"seen": [], "unseen": ["something"]}')
    with pytest.raises(ValueError, match="seen"):
        discover_cache_episodes(other, val_episodes=0, phases=("clean",))


def test_discovery_rejects_conflicting_instruction_sources(tmp_path):
    path = write_episode(tmp_path)
    alternative = path.parent / "instructions" / path.with_suffix(".json").name
    alternative.parent.mkdir()
    alternative.write_text('{"seen":["different"],"unseen":[]}')
    with pytest.raises(ValueError, match="Conflicting"):
        discover_cache_episodes(tmp_path, val_episodes=0, phases=("clean",))


def test_discovery_parallel_matches_serial(tmp_path):
    for task in ("task_one", "task_two"):
        for phase in PHASES:
            for episode in range(2):
                write_episode(tmp_path, task=task, phase=phase, episode=episode)
    assert discover_cache_episodes(tmp_path, workers=2) == discover_cache_episodes(
        tmp_path
    )


def test_static_cleaning_keeps_original_grid_and_context():
    qpos = np.zeros((301, 14), np.float64)

    qpos[220:, 0] = np.arange(81) * 0.01
    frames = selected_frames(qpos)
    assert np.array_equal(qpos[:, 0], np.r_[np.zeros(220), np.arange(81) * 0.01])
    assert frames[0] == 0 and frames[-1] == 280
    assert np.all(frames % 20 == 0)
    assert not np.isin(20, frames) and all(
        frame in frames for frame in (140, 160, 180, 200, 220)
    )
    assert np.all(np.diff(frames[frames >= 140]) == 20)
    report = history_coverage(frames)
    assert report["segments"] == 2 and report["full_memory_frames"] > 0
    assert sum(report["available_counts"].values()) == len(frames)


def test_short_static_trajectory_never_disappears():
    for length in (2, 19, 21, 41, 60):
        qpos = np.zeros((length, 14))
        frames = selected_frames(qpos)
        grid = np.arange(0, length - 1, 20)
        np.testing.assert_array_equal(frames, grid)
        report = history_coverage(frames)
        assert report["segments"] == 1 and report["frames"] == len(grid)


def test_gripper_transitions_and_horizon_boundary_are_not_dropped():
    qpos = np.zeros((401, 14))
    qpos[161:, 6] = 0.8
    frames = selected_frames(qpos)

    assert all(index in frames for index in (80, 100, 120, 140, 160, 180))
    assert 20 not in frames
    for frame in frames:
        expected = action_window(int(frame), len(qpos))
        actual = make_action_window(int(frame), len(qpos))
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
    indices, valid = make_action_window(380, 401)
    assert indices[:3].tolist() == [381, 382, 383] and valid.sum() == 20


def test_cumulative_joint_drift_prevents_false_static_drop():
    qpos = np.zeros((301, 14))
    qpos[:, 0] = np.arange(301) * 0.5e-6
    np.testing.assert_array_equal(selected_frames(qpos), np.arange(0, 300, 20))


@pytest.mark.parametrize("quaternion", [False, True])
def test_measured_pose_guard_retains_actuator_settling(quaternion):
    qpos = np.zeros((301, 14))
    poses = identity_poses(301, quaternion=quaternion)
    if quaternion:
        poses[:, 0, 0] = np.arange(301) * 0.0001
    else:
        poses[:, 0, 0, 3] = np.arange(301) * 0.0001
    np.testing.assert_array_equal(selected_frames(qpos, poses), np.arange(0, 300, 20))
    assert len(selected_frames(qpos)) < 15


@pytest.mark.parametrize("quaternion", [False, True])
def test_measured_rotation_guard_and_quaternion_sign(quaternion):
    qpos = np.zeros((301, 14))
    poses = identity_poses(301, quaternion=quaternion)
    angle = np.arange(301) * 0.001
    if quaternion:
        poses[:, 0, 3] = np.cos(angle / 2)
        poses[:, 0, 6] = np.sin(angle / 2)
        poses[1::2] *= 1
        poses[1::2, :, 3:] *= -1
    else:
        poses[:, 0, 0, 0] = poses[:, 0, 1, 1] = np.cos(angle)
        poses[:, 0, 0, 1] = -np.sin(angle)
        poses[:, 0, 1, 0] = np.sin(angle)
    np.testing.assert_array_equal(selected_frames(qpos, poses), np.arange(0, 300, 20))


def test_optional_pose_guard_never_fabricates_missing_arm(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = write_episode(tmp_path)
    with h5py.File(path, "a") as handle:
        assert read_endposes(handle) is None
        handle.create_dataset(
            "endpose/left_endpose", data=identity_poses(83, quaternion=True)[:, 0]
        )
        with pytest.raises(ValueError, match="both arms"):
            read_endposes(handle)
        handle.create_dataset(
            "endpose/right_endpose", data=identity_poses(83, quaternion=True)[:, 1]
        )
        assert read_endposes(handle).shape == (83, 2, 7)


def test_cleaning_rejects_nonfinite_and_invalid_configuration():
    with pytest.raises(ValueError, match="finite"):
        selected_frames(np.full((10, 14), np.nan))
    with pytest.raises(ValueError, match="thresholds"):
        selected_frames(np.zeros((10, 14)), stride=0)
    with pytest.raises(ValueError, match="unit length"):
        selected_frames(np.zeros((10, 14)), np.zeros((10, 2, 7)))


def test_history_coverage_resets_at_every_gap_not_just_large_gaps():
    report = history_coverage(np.array([0, 20, 40, 60, 100, 120, 140, 160, 180]))
    assert report == {
        "frames": 9,
        "segments": 2,
        "available_counts": {"1": 2, "2": 2, "3": 2, "4": 3},
        "full_memory_frames": 3,
    }
    assert history_coverage(np.array([], dtype=np.int64))["segments"] == 0
    with pytest.raises(ValueError, match="increasing"):
        history_coverage(np.array([0, 20, 20]))


def test_augmentation_is_deterministic_bounded_and_spatially_aligned():
    image = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    raw = augmentation_params(42, "task/clean/episode0", 0)
    np.testing.assert_array_equal(augment_montage(image, raw), image)
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1)
    assert torch.equal(augment_montage(tensor, raw), tensor)
    first = augmentation_params(42, "task/clean/episode0", 1)
    assert first == augmentation_params(42, "task/clean/episode0", 1)
    assert first != augmentation_params(42, "task/clean/episode0", 2)
    assert first != augmentation_params(43, "task/clean/episode0", 1)
    assert all(
        0.8 <= first[name] <= 1.2 for name in ("brightness", "contrast", "saturation")
    )
    result = augment_montage(image, first)
    np.testing.assert_array_equal(
        result, augment_montage(tensor, first).permute(1, 2, 0).numpy()
    )
    assert (
        result.shape == image.shape
        and result.dtype == image.dtype
        and not np.array_equal(result, image)
    )
    np.testing.assert_array_equal(result, augment_montage(image, first))


def test_unseen_encoded_but_never_merged_into_training_role():
    rows = [
        row(
            seen=("Move bottle", "Move bottle"),
            unseen=("Please move bottle", "Move bottle"),
        ),
        row("task_one/randomized/episode1"),
    ]
    table = build_prompt_table(rows)
    assert len(table["prompts"]) == 2
    prompt = DEFAULT_PROMPT.format(task="Move bottle")
    identity = hashlib.sha256(prompt.encode()).hexdigest()
    assert table["prompts"][identity] == prompt
    first = table["episodes"][rows[0].id]
    assert first["seen"] == [identity] and len(first["unseen"]) == 2
    assert (
        len(rows[0].instructions) == 2
        and "Please move bottle" not in rows[0].instructions
    )
    with pytest.raises(ValueError, match="Duplicate"):
        build_prompt_table([rows[0], rows[0]])


def test_source_binding_content_identity_and_metadata_only(tmp_path):
    path = tmp_path / "source.json"
    path.write_bytes(b"test")
    result = source_binding(path)
    assert result["sha256"] == hashlib.sha256(b"test").hexdigest()
    assert result["size"] == 4 and result["mtime_ns"] == path.stat().st_mtime_ns
    assert result["path"] == str(path.resolve())
    stat_only = source_binding(path, hash_content=False)
    assert stat_only == {k: value for k, value in result.items() if k != "sha256"}
