import hashlib
import io
import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DEFAULT_RAW_ROOT = "artifacts/robotwin2_aloha_raw"
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
PHASES = {"clean": "aloha-agilex_clean_50", "randomized": "aloha-agilex_randomized_500"}
CAMERAS = ("head_camera", "left_camera", "right_camera")


@dataclass(frozen=True)
class EpisodeSpec:
    id: str
    path: str
    instruction_path: str
    task: str
    phase: str
    raw_length: int
    instructions: tuple[str, ...]

    @property
    def observation_count(self):
        return self.raw_length - 1


def discover_episodes(
    root, *, tasks=None, phases=("clean",), split="train", val_episodes=1, split_seed=0
):
    root = Path(root).resolve()
    if split not in ("train", "val", "all") or val_episodes < 0:
        raise ValueError("split must be train/val/all and val_episodes nonnegative")
    if (
        not phases
        or any(p not in PHASES for p in phases)
        or len(set(phases)) != len(phases)
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
        or any(not re.fullmatch(r"[a-z0-9_]+", n) for n in names)
    ):
        raise ValueError("Task names must be nonempty simple identifiers")
    import h5py

    rows = []
    for task in names:
        for phase in phases:
            directory = root / task / PHASES[phase]
            files = sorted(
                directory.glob("episode*.hdf5"),
                key=lambda p: (
                    int(re.fullmatch(r"episode(\d+)\.hdf5", p.name).group(1))
                    if re.fullmatch(r"episode(\d+)\.hdf5", p.name)
                    else -1
                ),
            )
            if not files or any(
                not re.fullmatch(r"episode\d+\.hdf5", p.name) for p in files
            ):
                raise ValueError(f"No valid raw episodes for {task}/{phase}")
            if val_episodes >= len(files):
                raise ValueError(f"Holdout leaves no training episodes: {task}/{phase}")
            ordered = sorted(
                files,
                key=lambda p: hashlib.sha256(
                    f"{split_seed}:{task}:{phase}:{p.stem}".encode()
                ).digest(),
            )
            held_out = {p.name for p in ordered[:val_episodes]}
            for path in files:
                if split != "all" and (path.name in held_out) != (split == "val"):
                    continue
                if not path.resolve().is_relative_to(root):
                    raise ValueError("Raw episode resolves outside the source root")
                instruction_path = path.with_suffix(".json")
                meta = json.loads(instruction_path.read_text(encoding="utf-8"))
                instructions = meta.get("seen")
                if (
                    not isinstance(instructions, list)
                    or not instructions
                    or any(
                        not isinstance(s, str) or not s.strip() for s in instructions
                    )
                ):
                    raise ValueError(f"Missing seen instructions: {instruction_path}")
                with h5py.File(path, "r") as f:
                    q = f["joint_action/vector"]
                    if q.ndim != 2 or q.shape[1] != 14 or len(q) < 2:
                        raise ValueError(f"Expected raw [N,14] joint table: {path}")
                    length = len(q)
                    for camera in CAMERAS:
                        if len(f[f"observation/{camera}/rgb"]) != length:
                            raise ValueError(
                                f"Image/joint length mismatch: {path}:{camera}"
                            )
                rows.append(
                    EpisodeSpec(
                        f"{task}/{phase}/{path.stem}",
                        str(path),
                        str(instruction_path),
                        task,
                        phase,
                        length,
                        tuple(instructions),
                    )
                )
    if not rows:
        raise ValueError("Selected split contains no episodes")
    return rows


def action_window(frame, raw_length, horizon=32):
    if horizon < 1 or not 0 <= frame < raw_length - 1:
        raise ValueError("Window must start at an actual observation")
    logical = frame + np.arange(horizon)
    return np.minimum(logical + 1, raw_length - 1), logical < raw_length - 1


def decode_rgb(encoded):
    raw = encoded.tobytes() if hasattr(encoded, "tobytes") else bytes(encoded)
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def rgb_montage(images):
    def resize(array, wh):
        return np.asarray(
            Image.fromarray(array).resize(wh, Image.Resampling.BILINEAR), dtype=np.uint8
        )

    head = resize(images["head_camera"], (320, 256))
    left = resize(images["left_camera"], (160, 128))
    right = resize(images["right_camera"], (160, 128))
    image = np.concatenate([head, np.concatenate([left, right], axis=1)], axis=0)
    return torch.from_numpy(image.copy()).permute(2, 0, 1)


def normalize_rgb_image(image, *, device, dtype):
    if not isinstance(image, torch.Tensor) or image.dtype != torch.uint8:
        raise ValueError("RGB transport must be an unnormalized uint8 tensor")
    if image.ndim not in (3, 4) or image.shape[-3:] != (3, 384, 320):
        raise ValueError("RGB must have the official [3,384,320] montage shape")
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("RGB model dtype must be floating point")
    return image.to(device=device, dtype=dtype, non_blocking=True) * (2.0 / 255.0) - 1.0


class RawTrajectoryDataset(torch.utils.data.Dataset):

    def __init__(self, rows, processor, *, horizon=32, max_open_files=8):
        self.rows, self.processor, self.horizon = list(rows), processor, horizon
        self.max_open_files = max_open_files
        self._files = OrderedDict()
        if max_open_files < 1 or horizon < 1:
            raise ValueError("Positive file limit and horizon required")
        for kind in ("state", "action"):
            meta = processor.shape_meta[kind]
            if len(meta) != 1 or int(meta[0]["shape"]) != 14:
                raise ValueError(
                    "Raw joint deployment requires exactly one merged 14D field"
                )

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_files"] = OrderedDict()
        return state

    def close(self):
        for file in self._files.values():
            file.close()
        self._files.clear()

    def _file(self, index):
        import h5py

        if index not in self._files:
            if len(self._files) >= self.max_open_files:
                _, old = self._files.popitem(last=False)
                old.close()
            self._files[index] = h5py.File(self.rows[index].path, "r")
        self._files.move_to_end(index)
        return self._files[index]

    def __getitem__(self, request):
        if len(request) not in (4, 5):
            raise ValueError(
                "Raw requests require four fields and an optional trajectory pass"
            )
        lane, index, frame, instruction_index = map(int, request[:4])
        trajectory_pass_id = request[4] if len(request) == 5 else None
        if trajectory_pass_id is not None:
            if (
                isinstance(trajectory_pass_id, bool)
                or not isinstance(trajectory_pass_id, Integral)
                or trajectory_pass_id < 0
            ):
                raise ValueError("Trajectory pass must be a nonnegative integer")
            trajectory_pass_id = int(trajectory_pass_id)
        row, file = self.rows[index], self._file(index)
        indices, valid = action_window(frame, row.raw_length, self.horizon)
        qpos = np.asarray(file["joint_action/vector"][:], dtype=np.float32)
        if not np.isfinite(qpos).all():
            raise ValueError(f"Nonfinite joint data: {row.id}")
        state_key = self.processor.shape_meta["state"][0]["key"]
        action_key = self.processor.shape_meta["action"][0]["key"]
        batch = {
            "state": {state_key: torch.from_numpy(qpos[frame : frame + 1].copy())},
            "action": {action_key: torch.from_numpy(qpos[indices].copy())},
        }
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        state, action = batch["state"][state_key][0], batch["action"][action_key]
        if state.shape != (14,) or action.shape != (self.horizon, 14):
            raise ValueError("Upstream action/state processor changed dimensions")
        if not torch.isfinite(state).all() or not torch.isfinite(action).all():
            raise ValueError("Processor produced nonfinite targets/state")
        images = {
            camera: decode_rgb(file[f"observation/{camera}/rgb"][frame])
            for camera in CAMERAS
        }
        instruction = row.instructions[instruction_index % len(row.instructions)]
        sample = {
            "lane": lane,
            "episode_id": row.id,
            "task": row.task,
            "phase": row.phase,
            "frame": frame,
            "decision_index": None,
            "input_image": rgb_montage(images),
            "proprio": state.float(),
            "action": action.float(),
            "valid": torch.from_numpy(valid),
            "prompt": DEFAULT_PROMPT.format(task=instruction),
        }
        if trajectory_pass_id is not None:
            sample["trajectory_pass_id"] = trajectory_pass_id
        return sample


def trajectory_requests(
    rows,
    seed,
    batch_size,
    decision_stride,
    *,
    max_batches=None,
    task_probabilities=None,
    phase_probabilities=None,
):
    if task_probabilities is not None or phase_probabilities is not None:
        from reread.data.sampling import weighted_trajectory_requests

        yield from weighted_trajectory_requests(
            rows,
            seed,
            batch_size,
            decision_stride,
            max_batches=max_batches,
            task_probabilities=task_probabilities,
            phase_probabilities=phase_probabilities,
        )
        return
    if batch_size < 1 or decision_stride < 1:
        raise ValueError("Batch and decision stride must be positive")
    rng = np.random.default_rng(seed)
    order = iter(rng.permutation(len(rows)).tolist())
    lanes, batches = {}, 0
    while True:
        requests = []
        for lane in range(batch_size):
            entry = lanes.get(lane)
            if entry is None or entry[1] >= rows[entry[0]].observation_count:
                index = next(order, None)
                if index is None:
                    lanes.pop(lane, None)
                    continue
                entry = (index, 0, int(rng.integers(len(rows[index].instructions))))
            index, frame, instruction = entry
            requests.append((lane, index, frame, instruction))
            lanes[lane] = (index, frame + decision_stride, instruction)
        if not requests:
            return
        yield requests
        batches += 1
        if max_batches is not None and batches >= max_batches:
            return


def _identity(value):
    return value


def iter_trajectory_batches(
    dataset,
    seed,
    batch_size,
    decision_stride,
    *,
    workers=0,
    max_batches=None,
    task_probabilities=None,
    phase_probabilities=None,
):
    if workers < 0:
        raise ValueError("workers cannot be negative")
    requests = trajectory_requests(
        dataset.rows,
        seed,
        batch_size,
        decision_stride,
        max_batches=max_batches,
        task_probabilities=task_probabilities,
        phase_probabilities=phase_probabilities,
    )
    stopping = False

    def sampler():
        for batch in requests:
            if stopping:
                return
            yield batch

    kwargs = dict(
        dataset=dataset,
        batch_sampler=sampler(),
        collate_fn=_identity,
        num_workers=workers,
        pin_memory=bool(workers),
        generator=torch.Generator().manual_seed(seed),
    )
    if workers:
        kwargs.update(multiprocessing_context="spawn", prefetch_factor=2)
    loader = torch.utils.data.DataLoader(**kwargs)
    iterator = iter(loader)
    try:
        for samples in iterator:
            for sample in samples:
                if sample.get("decision_index") is None:
                    sample["decision_index"] = sample["frame"] // decision_stride
            yield samples
    finally:
        stopping = True
        for _ in iterator:
            pass
