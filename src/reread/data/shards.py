import hashlib
import json
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np

from reread.data.cache_contract import CACHE_SCHEMA as SCHEMA_V2

LATENT_SHAPE = (48, 1, 24, 20)
TEXT_SHAPE = (128, 4096)
LABEL_NAMES = ("frames", "segments", "proprio", "action", "valid")


def array_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def labels_sha256(payload):
    checksum = hashlib.sha256()
    for name in LABEL_NAMES:
        value = np.ascontiguousarray(payload[name])
        checksum.update(name.encode() + b"\0")
        checksum.update(value.dtype.str.encode() + b"\0")
        checksum.update(
            json.dumps(list(value.shape), separators=(",", ":")).encode() + b"\0"
        )
        checksum.update(value.tobytes())
    return checksum.hexdigest()


def _file_sha256(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            checksum.update(block)
    return checksum.hexdigest()


class ShardWriter:

    def __init__(
        self, root, *, rank, plan_sha256, text_shard_mib=8192, rgb_shard_mib=2048
    ):
        if text_shard_mib < 1 or rgb_shard_mib < 1:
            raise ValueError("Shard target sizes must be positive MiB")
        self.root = Path(root)
        self.rank, self.plan_sha = rank, plan_sha256
        self.text_capacity = max(
            1, int(text_shard_mib * 1024**2) // (int(np.prod(TEXT_SHAPE)) * 2)
        )
        self.rgb_limit = int(rgb_shard_mib * 1024**2)
        self.text, self.episodes, self.shards = [], [], {}
        self.published = {"text": 0, "episodes": 0}
        self.timings = {"flush_publish_seconds": 0.0, "shard_hash_seconds": 0.0}
        self._active = {}
        self._serial = {"text": 0, "rgb": 0}
        self._closed = False

    def _open(self, kind):
        import h5py

        relative = (
            f"shards/{kind}/rank_{self.rank:05d}_shard_{self._serial[kind]:05d}.h5"
        )
        self._serial[kind] += 1
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        temporary = path.with_name(path.name + f".pending.{os.getpid()}")
        handle = h5py.File(temporary, "x")
        handle.attrs.update(schema=SCHEMA_V2, plan_sha256=self.plan_sha, kind=kind)
        if kind == "text":
            handle.attrs["text_padding"] = (
                "zero_context_after_tokenizer_length_then_all_true_mask"
            )
            handle.create_dataset(
                "contexts",
                shape=(0, *TEXT_SHAPE),
                maxshape=(None, *TEXT_SHAPE),
                dtype=np.uint16,
                chunks=(1, *TEXT_SHAPE),
            )
            for name in ("prompt_id", "sha256"):
                handle.create_dataset(
                    name, shape=(0,), maxshape=(None,), dtype="S64", chunks=(256,)
                )
        else:
            handle.create_group("episodes")
        state = dict(
            handle=handle,
            path=path,
            temporary=temporary,
            relative=relative,
            records=0,
            bytes=0,
        )
        self._active[kind] = state
        return state

    def _finalize(self, kind):
        state = self._active.pop(kind, None)
        if state is None:
            return
        handle = state["handle"]
        stamp = time.monotonic()
        handle.attrs["records"] = state["records"]
        handle.flush()
        handle.close()

        os.link(state["temporary"], state["path"])
        state["temporary"].unlink()
        self.timings["flush_publish_seconds"] += time.monotonic() - stamp
        stamp = time.monotonic()
        digest = _file_sha256(state["path"])
        self.timings["shard_hash_seconds"] += time.monotonic() - stamp
        self.shards[state["relative"]] = dict(
            kind=kind,
            records=state["records"],
            size=state["path"].stat().st_size,
            sha256=digest,
        )
        self.published["text" if kind == "text" else "episodes"] += state["records"]

    def append_text(self, ids, contexts, masks):
        import torch

        if (
            contexts.device.type != "cpu"
            or masks.device.type != "cpu"
            or contexts.dtype != torch.bfloat16
            or contexts.shape != (len(ids), *TEXT_SHAPE)
            or masks.dtype != torch.bool
            or masks.shape != (len(ids), TEXT_SHAPE[0])
        ):
            raise ValueError(
                "Text shard requires finite CPU BF16 contexts and all-true masks"
            )
        bits = contexts.contiguous().view(torch.uint16).numpy()

        if not masks.numpy().all() or np.any((bits & 0x7F80) == 0x7F80):
            raise ValueError(
                "Text shard requires finite CPU BF16 contexts and all-true masks"
            )
        offset = 0
        while offset < len(ids):
            state = self._active.get("text") or self._open("text")
            start = state["records"]
            count = min(len(ids) - offset, self.text_capacity - start)
            selected = ids[offset : offset + count]
            if any(
                len(key) != 64 or any(c not in "0123456789abcdef" for c in key)
                for key in selected
            ):
                raise ValueError("Text prompt identity must be a SHA256 hex string")
            digests = [array_sha256(bits[offset + i]) for i in range(count)]
            handle = state["handle"]
            for name in ("contexts", "prompt_id", "sha256"):
                handle[name].resize(start + count, axis=0)
            handle["contexts"][start : start + count] = bits[offset : offset + count]
            handle["prompt_id"][start : start + count] = np.asarray(
                selected, dtype="S64"
            )
            handle["sha256"][start : start + count] = np.asarray(digests, dtype="S64")
            self.text.extend(
                dict(id=key, path=state["relative"], row=start + i, sha256=digests[i])
                for i, key in enumerate(selected)
            )
            state["records"] += count
            offset += count
            if state["records"] == self.text_capacity:
                self._finalize("text")

    def append_episode(self, payload, latents):
        import torch

        row = payload["row"]
        expected = (
            len(payload["augmentation_parameters"]),
            len(payload["frames"]),
            *LATENT_SHAPE,
        )
        if (
            latents.device.type != "cpu"
            or latents.dtype != torch.bfloat16
            or tuple(latents.shape) != expected
        ):
            raise ValueError("Invalid encoded episode for RGB shard")
        bits = latents.contiguous().view(torch.uint16).numpy()
        if np.any((bits & 0x7F80) == 0x7F80):
            raise ValueError("Nonfinite encoded episode for RGB shard")
        data_bytes = bits.nbytes + sum(
            np.asarray(payload[name]).nbytes for name in LABEL_NAMES
        )
        state = self._active.get("rgb")
        if (
            state is not None
            and state["records"]
            and state["bytes"] + data_bytes > self.rgb_limit
        ):
            self._finalize("rgb")
            state = None
        state = state or self._open("rgb")
        group_name = "episodes/" + hashlib.sha256(row["id"].encode()).hexdigest()
        group = state["handle"].create_group(group_name)
        label_digest = labels_sha256(payload)
        group.attrs.update(
            schema=SCHEMA_V2,
            episode_id=row["id"],
            plan_sha256=self.plan_sha,
            rgb_dtype="bfloat16_bits_uint16",
            labels_sha256=label_digest,
            source=json.dumps(payload["source"], sort_keys=True),
            augmentation_parameters=json.dumps(
                payload["augmentation_parameters"], sort_keys=True
            ),
        )
        group.create_dataset("rgb_latents", data=bits, chunks=(1, 1, *LATENT_SHAPE))
        digests = np.asarray(
            [[array_sha256(frame) for frame in variant] for variant in bits],
            dtype="S64",
        )
        group.create_dataset("rgb_sha256", data=digests)
        group.attrs["rgb_checksums_sha256"] = array_sha256(digests)
        for name in LABEL_NAMES:
            group.create_dataset(name, data=payload[name])
        entry = dict(
            id=row["id"],
            path=state["relative"],
            group=group_name,
            labels_sha256=label_digest,
            rgb_checksums_sha256=array_sha256(digests),
            frames=len(payload["frames"]),
            variants=latents.shape[0],
            history=payload["history"],
            raw_history=payload["raw_history"],
            raw_grid_frames=(row["raw_length"] - 2) // 20 + 1,
            split=row["split"],
            task=row["task"],
            phase=row["phase"],
            source=payload["source"],
            augmentation_parameters=payload["augmentation_parameters"],
        )
        self.episodes.append(entry)
        state["records"] += 1
        state["bytes"] += data_bytes

    def finish(self):
        if not self._closed:
            self._finalize("text")
            self._finalize("rgb")
            self._closed = True
        return dict(text=self.text, episodes=self.episodes, shards=self.shards)

    def abort(self):
        for state in self._active.values():
            state["handle"].close()
        self._active.clear()
        self._closed = True


class AsyncShardWriter:

    def __init__(self, root, *, depth=2, **kwargs):
        if depth < 1:
            raise ValueError("Writer queue depth must be positive")
        self.queue = queue.Queue(maxsize=depth)
        self.writer = ShardWriter(root, **kwargs)
        self.error = None
        self.result = None
        self.counts = {"text": 0, "episodes": 0}
        self.timings = {"queue_wait_seconds": 0.0, "write_seconds": 0.0}
        self._lock = threading.Lock()
        self._done = False
        self.thread = threading.Thread(
            target=self._run, name="reread-shard-writer", daemon=True
        )
        self.thread.start()

    def _check(self):
        if self.error is not None:
            raise RuntimeError("Observation shard writer failed") from self.error

    def _run(self):
        try:
            while True:
                work = self.queue.get()
                try:
                    if work is None:
                        self.result = self.writer.finish()
                        return
                    kind, values = work
                    started = time.monotonic()
                    if kind == "text":
                        self.writer.append_text(*values)
                        count = len(values[0])
                    else:
                        self.writer.append_episode(*values)
                        count = 1
                    with self._lock:
                        self.counts[kind] += count
                        self.timings["write_seconds"] += time.monotonic() - started
                finally:
                    self.queue.task_done()
        except BaseException as exc:
            self.error = exc
            self.writer.abort()

    def _put(self, work):
        started = time.monotonic()
        while True:
            self._check()
            try:
                self.queue.put(work, timeout=0.1)
                break
            except queue.Full:
                continue
        with self._lock:
            self.timings["queue_wait_seconds"] += time.monotonic() - started
        self._check()

    def submit_text(self, ids, contexts, masks):
        if self._done:
            raise RuntimeError("Writer is closed")
        if contexts.device.type != "cpu" or masks.device.type != "cpu":
            raise ValueError("Only CPU tensors may enter the writer queue")
        self._put(("text", (tuple(ids), contexts, masks)))

    def submit_episode(self, payload, latents):
        if self._done:
            raise RuntimeError("Writer is closed")
        if latents.device.type != "cpu":
            raise ValueError("Only CPU tensors may enter the writer queue")
        self._put(("episodes", (payload, latents)))

    def progress(self):
        self._check()
        with self._lock:
            return dict(
                text_written=self.counts["text"],
                episodes_written=self.counts["episodes"],
                text_committed=self.writer.published["text"],
                episodes_committed=self.writer.published["episodes"],
                committed_shards=len(self.writer.shards),
                **self.writer.timings,
                **self.timings,
                queue_depth=self.queue.qsize(),
            )

    def finish(self):
        if not self._done:
            self._put(None)
            self._done = True
        while self.thread.is_alive():
            self.thread.join(timeout=0.1)
            self._check()
        self._check()
        return self.result

    def abort(self):
        if not self._done:
            try:
                self._put(None)
            except RuntimeError:
                pass
            self._done = True
        self.thread.join()
