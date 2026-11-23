from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance

from reread.data.cache_contract import PROMPT_TABLE_SCHEMA
from reread.data.demonstrations import CAMERAS, DEFAULT_PROMPT, PHASES, action_window

EPISODE_PATTERN = re.compile(r"episode(\d+)\.hdf5")
STATIC_POLICY = "original_grid_full_horizon_joint_gripper_measured_pose_v1"


@dataclass(frozen=True)
class CacheEpisode:
    id: str
    path: str
    instruction_path: str
    task: str
    phase: str
    raw_length: int
    seen: tuple[str, ...]
    unseen: tuple[str, ...]
    split: str

    def __post_init__(self):
        if (
            not re.fullmatch(r"[a-z0-9_]+", self.task)
            or self.phase not in PHASES
            or self.split not in ("train", "val")
            or isinstance(self.raw_length, bool)
            or not isinstance(self.raw_length, Integral)
            or self.raw_length < 2
        ):
            raise ValueError("Invalid cache episode identity, length or split")
        if not self.id or not self.path or not self.instruction_path:
            raise ValueError("Cache episode paths and identity must not be empty")
        for role in ("seen", "unseen"):
            values = getattr(self, role)
            if (
                not isinstance(values, tuple)
                or (role == "seen" and not values)
                or any(not isinstance(s, str) or not s.strip() for s in values)
            ):
                raise ValueError(f"Invalid {role} instructions")

    @property
    def observation_count(self):
        return self.raw_length - 1

    @property
    def instructions(self):
        return self.seen

    def to_dict(self):
        result = asdict(self)
        result["seen"], result["unseen"] = list(self.seen), list(self.unseen)
        return result

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Cache episode must be a mapping")
        values = dict(value)
        for role in ("seen", "unseen"):
            if not isinstance(values.get(role), (list, tuple)):
                raise ValueError(f"Invalid serialized {role} instructions")
            values[role] = tuple(values[role])
        return cls(**values)


def _inside(root, path):
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Source resolves outside raw root: {path}")
    return resolved


def _instruction_path(root, directory, path):
    candidates = (
        path.with_suffix(".json"),
        directory / "instructions" / (path.stem + ".json"),
    )
    found = [_inside(root, p) for p in candidates if p.is_file()]
    if not found:
        raise ValueError(f"Missing instruction JSON: {path}")

    objects = [json.loads(p.read_text(encoding="utf-8")) for p in found]
    if any(obj != objects[0] for obj in objects[1:]):
        raise ValueError(f"Conflicting instruction JSONs: {path}")
    return found[0], objects[0]


def discover_cache_episodes(
    root,
    *,
    tasks=None,
    val_episodes=1,
    split_seed=0,
    phases=("clean", "randomized"),
    workers=1,
):
    root = Path(root).resolve()
    if (
        isinstance(val_episodes, bool)
        or not isinstance(val_episodes, Integral)
        or val_episodes < 0
    ):
        raise ValueError("val_episodes must be a nonnegative integer")
    if isinstance(workers, bool) or not isinstance(workers, Integral) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if (
        not phases
        or len(set(phases)) != len(phases)
        or any(p not in PHASES for p in phases)
    ):
        raise ValueError("Explicit clean/randomized phases must be unique")
    names = sorted(
        tasks
        if tasks is not None
        else [
            p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
        ]
    )
    if (
        not names
        or len(set(names)) != len(names)
        or any(
            not isinstance(n, str) or not re.fullmatch(r"[a-z0-9_]+", n) for n in names
        )
    ):
        raise ValueError("Task names must be nonempty simple identifiers")
    if workers > 1 and len(names) > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        arguments = [
            (str(root), task, val_episodes, split_seed, tuple(phases)) for task in names
        ]

        with ProcessPoolExecutor(
            max_workers=min(int(workers), len(names)),
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            return [
                row
                for group in pool.map(_discover_cache_task, arguments)
                for row in group
            ]
    import h5py

    rows = []
    for task in names:
        for phase in phases:
            directory = root / task / PHASES[phase]
            flattened = list(directory.glob("episode*.hdf5"))
            nested = list((directory / "data").glob("episode*.hdf5"))
            if flattened and nested:
                raise ValueError(f"Ambiguous raw data layouts: {task}/{phase}")
            files = flattened or nested
            if not files or any(not EPISODE_PATTERN.fullmatch(p.name) for p in files):
                raise ValueError(f"No valid raw episodes for {task}/{phase}")
            files.sort(key=lambda p: int(EPISODE_PATTERN.fullmatch(p.name).group(1)))
            if val_episodes >= len(files):
                raise ValueError(f"Holdout leaves no training episodes: {task}/{phase}")
            ordered = sorted(
                files,
                key=lambda p: hashlib.sha256(
                    f"{split_seed}:{task}:{phase}:{p.stem}".encode()
                ).digest(),
            )
            held_out = {p.name for p in ordered[:val_episodes]}
            for source in files:
                path = _inside(root, source)
                instruction_path, meta = _instruction_path(root, directory, source)
                if not isinstance(meta, dict):
                    raise ValueError(
                        f"Instruction metadata must be a mapping: {instruction_path}"
                    )
                seen, unseen = meta.get("seen"), meta.get("unseen", [])
                for role, values in (("seen", seen), ("unseen", unseen)):
                    if (
                        not isinstance(values, list)
                        or (role == "seen" and not values)
                        or any(not isinstance(s, str) or not s.strip() for s in values)
                    ):
                        raise ValueError(
                            f"Missing/invalid {role} instructions: {instruction_path}"
                        )
                with h5py.File(path, "r") as handle:
                    qpos = handle["joint_action/vector"]
                    if qpos.ndim != 2 or qpos.shape[1] != 14 or len(qpos) < 2:
                        raise ValueError(f"Expected raw [N,14] joint table: {path}")
                    length = len(qpos)
                    for camera in CAMERAS:
                        if len(handle[f"observation/{camera}/rgb"]) != length:
                            raise ValueError(
                                f"Image/joint length mismatch: {path}:{camera}"
                            )
                rows.append(
                    CacheEpisode(
                        f"{task}/{phase}/{path.stem}",
                        str(path),
                        str(instruction_path),
                        task,
                        phase,
                        length,
                        tuple(seen),
                        tuple(unseen),
                        "val" if path.name in held_out else "train",
                    )
                )
    if not rows:
        raise ValueError("No raw episodes discovered")
    return rows


def _discover_cache_task(arguments):
    root, task, val_episodes, split_seed, phases = arguments
    return discover_cache_episodes(
        root,
        tasks=[task],
        val_episodes=val_episodes,
        split_seed=split_seed,
        phases=phases,
        workers=1,
    )


def _pose_components(endposes, length):
    pose = np.asarray(endposes, dtype=np.float64)
    if not np.isfinite(pose).all():
        raise ValueError("Measured endposes must be finite")
    if pose.shape == (length, 2, 7):
        quat = pose[:, :, 3:]
        norm = np.linalg.norm(quat, axis=-1, keepdims=True)
        if np.any(np.abs(norm - 1) > 1e-3):
            raise ValueError("Measured endpose quaternions must be unit length")
        return pose[:, :, :3], quat / norm, None
    if pose.shape == (length, 2, 4, 4):
        rotation = pose[:, :, :3, :3]
        if (
            not np.allclose(rotation.swapaxes(-1, -2) @ rotation, np.eye(3), atol=1e-3)
            or not np.allclose(np.linalg.det(rotation), 1, atol=1e-3)
            or not np.allclose(pose[:, :, 3], [0, 0, 0, 1], atol=1e-6)
        ):
            raise ValueError("Measured endpose matrices must be rigid transforms")
        return pose[:, :, :3, 3], None, rotation
    raise ValueError("Expected measured endposes [N,2,7] or [N,2,4,4]")


def read_endposes(handle):
    keys = ("endpose/left_endpose", "endpose/right_endpose")
    present = [key in handle for key in keys]
    if not any(present):
        return None
    if not all(present):
        raise ValueError("Measured endpose guard requires both arms")
    return np.stack([np.asarray(handle[key][:]) for key in keys], axis=1)


def _rotation_distance(quaternion, rotation, first, last):
    if quaternion is not None:
        dot = np.abs(np.sum(quaternion[first] * quaternion[last], axis=-1))
        return 2 * np.arccos(np.clip(dot, 0, 1))
    relative = rotation[first].swapaxes(-1, -2) @ rotation[last]
    cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1) / 2
    return np.arccos(np.clip(cosine, -1, 1))


def selected_frames(
    qpos,
    endposes=None,
    *,
    stride=20,
    horizon=32,
    min_static_run=6,
    joint_threshold=1e-6,
    gripper_threshold=1e-6,
    position_threshold_m=5e-4,
    rotation_threshold_rad=2e-3,
    memory_frames=4,
):
    qpos = np.asarray(qpos, dtype=np.float64)
    if (
        qpos.ndim != 2
        or qpos.shape[1] != 14
        or len(qpos) < 2
        or not np.isfinite(qpos).all()
    ):
        raise ValueError("Expected finite qpos [N,14], N>=2")
    thresholds = (
        joint_threshold,
        gripper_threshold,
        position_threshold_m,
        rotation_threshold_rad,
    )
    if any(not np.isfinite(x) or x < 0 for x in thresholds) or any(
        isinstance(x, bool) or not isinstance(x, Integral) or x < 1
        for x in (stride, horizon, min_static_run, memory_frames)
    ):
        raise ValueError("Invalid static cleaning thresholds")
    grid = np.arange(0, len(qpos) - 1, stride, dtype=np.int64)
    joints = np.r_[0:6, 7:13]
    grippers = np.array([6, 13])
    delta = np.abs(np.diff(qpos, axis=0))
    static = (delta[:, joints].max(axis=1) <= joint_threshold) & (
        delta[:, grippers].max(axis=1) <= gripper_threshold
    )
    components = None
    if endposes is not None:
        components = _pose_components(endposes, len(qpos))
        xyz, quaternion, rotation = components
        indexes = np.arange(len(qpos) - 1)
        static &= (
            np.linalg.norm(np.diff(xyz, axis=0), axis=-1).max(axis=1)
            <= position_threshold_m
        )
        static &= (
            _rotation_distance(quaternion, rotation, indexes, indexes + 1).max(axis=1)
            <= rotation_threshold_rad
        )
    edges = np.flatnonzero(np.diff(np.r_[False, static, False]))
    runs = [
        (int(start), int(end))
        for start, end in zip(edges[::2], edges[1::2])
        if end - start >= min_static_run
    ]
    protected = {int(grid[0]), int(grid[-1])}

    for boundary in np.flatnonzero(static[1:] != static[:-1]) + 1:
        at = int(np.searchsorted(grid, boundary))
        for index in (at - 1, at):
            if 0 <= index < len(grid):
                protected.add(int(grid[index]))
    kept = []
    for frame in grid:
        frame = int(frame)
        stop = frame + horizon
        drop = (
            frame not in protected
            and stop < len(qpos)
            and any(start <= frame and stop <= end for start, end in runs)
        )
        if drop:
            window = qpos[frame : stop + 1]
            drop = (
                np.ptp(window[:, joints], axis=0).max() <= joint_threshold
                and np.ptp(window[:, grippers], axis=0).max() <= gripper_threshold
            )
            if drop and components is not None:
                xyz, quaternion, rotation = components
                indexes = np.arange(frame, stop + 1)
                drop = (
                    np.linalg.norm(xyz[indexes] - xyz[frame], axis=-1).max()
                    <= position_threshold_m
                    and _rotation_distance(quaternion, rotation, frame, indexes).max()
                    <= rotation_threshold_rad
                )
        if not drop:
            kept.append(frame)
    keep_indexes = set()
    for frame in kept:
        index = int(np.searchsorted(grid, frame))
        keep_indexes.update(range(max(0, index - memory_frames + 1), index + 1))
    return grid[sorted(keep_indexes)].copy()


def history_coverage(frames, *, stride=20, memory_frames=4):
    frames = np.asarray(frames)
    if (
        frames.ndim != 1
        or not np.issubdtype(frames.dtype, np.integer)
        or (len(frames) and (frames[0] < 0 or np.any(np.diff(frames) <= 0)))
    ):
        raise ValueError(
            "Frames must be strictly increasing nonnegative integer indices"
        )
    if any(
        isinstance(x, bool) or not isinstance(x, Integral) or x < 1
        for x in (stride, memory_frames)
    ):
        raise ValueError("Positive stride and memory_frames required")
    counts = {str(i): 0 for i in range(1, memory_frames + 1)}
    available, segments = 0, 0
    for index, frame in enumerate(frames):
        if index == 0 or frame - frames[index - 1] != stride:
            available, segments = 1, segments + 1
        else:
            available = min(memory_frames, available + 1)
        counts[str(available)] += 1
    return {
        "frames": len(frames),
        "segments": segments,
        "available_counts": counts,
        "full_memory_frames": counts[str(memory_frames)],
    }


def augmentation_params(seed, episode_id, variant):
    if isinstance(variant, bool) or not isinstance(variant, Integral) or variant < 0:
        raise ValueError("Augmentation variant must be a nonnegative integer")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("Augmentation needs a stable episode identity")
    if variant == 0:
        return {
            "brightness": 1.0,
            "contrast": 1.0,
            "saturation": 1.0,
            "order": ["brightness", "contrast", "saturation"],
        }
    digest = hashlib.sha256(f"{seed}:{episode_id}:{variant}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:16], "big"))
    names = ["brightness", "contrast", "saturation"]
    result = {name: float(rng.uniform(0.8, 1.2)) for name in names}
    result["order"] = [names[i] for i in rng.permutation(3)]
    return result


def augment_montage(image, params):
    is_tensor = isinstance(image, torch.Tensor)
    if is_tensor:
        if (
            image.device.type != "cpu"
            or image.dtype != torch.uint8
            or image.ndim != 3
            or image.shape[0] != 3
        ):
            raise ValueError("Expected CPU uint8 [3,H,W] montage")
        array = image.permute(1, 2, 0).numpy()
    else:
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError("Expected uint8 [H,W,3] montage")
    names = ("brightness", "contrast", "saturation")
    if (
        set(params) != {*names, "order"}
        or sorted(params["order"]) != sorted(names)
        or any(
            not np.isfinite(params[name]) or not 0.8 <= params[name] <= 1.2
            for name in names
        )
    ):
        raise ValueError("Expected bounded brightness/contrast/saturation recipe")
    if all(params[name] == 1 for name in names):
        result = array.copy()
    else:
        output = Image.fromarray(array)
        enhancers = {
            "brightness": ImageEnhance.Brightness,
            "contrast": ImageEnhance.Contrast,
            "saturation": ImageEnhance.Color,
        }
        for name in params["order"]:
            output = enhancers[name](output).enhance(params[name])
        result = np.asarray(output, dtype=np.uint8).copy()
    return (
        torch.from_numpy(result).permute(2, 0, 1).contiguous() if is_tensor else result
    )


def prompt_id(prompt):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a nonempty string")
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def build_prompt_table(rows, *, prompt_template=DEFAULT_PROMPT):
    prompts, episodes = {}, {}
    for row in rows:
        if row.id in episodes:
            raise ValueError(f"Duplicate cache episode identity: {row.id}")
        roles = {}
        for role in ("seen", "unseen"):
            ids = []
            for instruction in getattr(row, role):
                prompt = prompt_template.format(task=instruction)
                identity = prompt_id(prompt)
                if identity in prompts and prompts[identity] != prompt:
                    raise ValueError("Prompt SHA256 collision")
                prompts[identity] = prompt
                if identity not in ids:
                    ids.append(identity)
            roles[role] = ids
        episodes[row.id] = roles
    return {
        "schema": PROMPT_TABLE_SCHEMA,
        "prompts": dict(sorted(prompts.items())),
        "episodes": episodes,
    }


def source_binding(path, *, hash_content=True):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    if not path.is_file():
        raise ValueError(f"Source must be a regular file: {path}")
    if not hash_content:
        return {
            "path": str(path),
            "size": before.st_size,
            "mtime_ns": before.st_mtime_ns,
        }
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Source changed while hashing: {path}")
    return {
        "path": str(path),
        "size": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def make_action_window(frame, raw_length, horizon=32):
    return action_window(frame, raw_length, horizon)
