import hashlib
import json
import os
import re
import sqlite3
from collections import OrderedDict
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from reread.data.cache_contract import (
    CACHED_PROMPT_ROLES,
    TEXT_REUSE_SCHEMA,
    TEXT_REUSE_VERIFICATION,
    validate_prompt_roles,
)
from reread.data.dataset import (
    LATENT_SHAPE,
    TEXT_SHAPE,
    CachedEpisode,
    CachedTrajectoryDataset,
    _digest,
    _integer,
    _json,
    _relative_file,
    _sha256,
    _validate_encoder,
)
from reread.data.demonstrations import DEFAULT_PROMPT, PHASES
from reread.data.preprocessing import CacheEpisode, make_action_window
from reread.data.shards import LABEL_NAMES, SCHEMA_V2, array_sha256, labels_sha256

RGB_PREFETCH_FRAMES = 16


def _text_reuse_plan(manifest, plan, current_plan_sha):
    reuse = manifest.get("text_reuse")
    bound = plan.get("source_cache_identity")
    if reuse is None:
        if bound is not None:
            raise ValueError("Missing explicit sealed text reuse provenance")
        return current_plan_sha
    required = {
        "schema",
        "mode",
        "source_cache",
        "source_manifest_sha256",
        "source_plan_sha256",
        "source_index_sha256",
        "source_index_size",
        "source_prompts_sha256",
        "verification",
    }
    if (
        not isinstance(reuse, dict)
        or set(reuse) != required
        or reuse != bound
        or reuse.get("schema") != TEXT_REUSE_SCHEMA
        or reuse.get("mode") != "hardlink_unchanged_shards"
        or reuse.get("verification") != TEXT_REUSE_VERIFICATION
    ):
        raise ValueError("Unsupported or unbound sealed text reuse provenance")
    source = reuse.get("source_cache")
    if not isinstance(source, str) or not Path(source).is_absolute():
        raise ValueError("Text reuse source cache identity must be an absolute path")
    for key in (
        "source_manifest_sha256",
        "source_plan_sha256",
        "source_index_sha256",
        "source_prompts_sha256",
    ):
        _digest(reuse.get(key), "text reuse " + key)
    if _integer(reuse.get("source_index_size"), "text reuse source index size") < 1:
        raise ValueError("Text reuse source index size must be positive")
    return reuse["source_plan_sha256"]


def _role_digest(identities):
    return hashlib.sha256("\n".join(identities).encode("ascii")).hexdigest()


def _ascii(value):
    return bytes(value).decode("ascii")


class _RoleInstructions(Sequence):

    def __init__(self, dataset, episode_id, specifications):
        self.dataset, self.episode_id = dataset, episode_id
        self.specifications = tuple(specifications)
        self.count = sum(count for _, count, _ in self.specifications)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(self.count)))
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError(index)
        for role, count, digest in self.specifications:
            if index < count:
                return self.dataset._role_ids(self.episode_id, role, count, digest)[
                    index
                ]
            index -= count
        raise IndexError(index)


class ShardedCachedTrajectoryDataset(CachedTrajectoryDataset):

    def __init__(
        self,
        root,
        processor,
        *,
        split="train",
        tasks=None,
        phases=("clean", "randomized"),
        max_open_files=64,
        text_cache_entries=128,
        verify_files=True,
        seed=0,
        rgb_cache_mib=64,
        instruction_roles_by_phase=None,
        augmentation=True,
    ):
        self.root = Path(root).resolve(strict=True)
        self.processor, self.split = processor, split
        self._configure_sampling(instruction_roles_by_phase, augmentation)
        self.max_open_files = _integer(max_open_files, "max_open_files")
        self.text_cache_entries = _integer(text_cache_entries, "text_cache_entries")
        self.seed = _integer(seed, "seed")
        self.rgb_cache_bytes = _integer(rgb_cache_mib, "rgb_cache_mib") * 1024**2
        if (
            split not in ("train", "val")
            or min(self.max_open_files, self.text_cache_entries) < 1
        ):
            raise ValueError("Expected train/val split and positive cache limits")
        if not isinstance(verify_files, bool):
            raise ValueError("verify_files must be boolean")

        self.verify_files = verify_files
        if (
            not phases
            or len(set(phases)) != len(phases)
            or any(p not in PHASES for p in phases)
        ):
            raise ValueError("Explicit clean/randomized phases must be unique")
        names = None if tasks is None else list(tasks)
        if names is not None and (
            not names
            or len(set(names)) != len(names)
            or any(
                not isinstance(n, str) or not re.fullmatch(r"[a-z0-9_]+", n)
                for n in names
            )
        ):
            raise ValueError("Invalid selected task names")
        for kind in ("state", "action"):
            meta = processor.shape_meta[kind]
            if len(meta) != 1 or int(meta[0]["shape"]) != 14:
                raise ValueError(
                    "Cached joint deployment requires one merged 14D field"
                )
        complete = _json(_relative_file(self.root, "complete.json"))
        manifest_path = _relative_file(self.root, "manifest.json")
        manifest = _json(manifest_path)
        manifest_sha = _sha256(manifest_path)
        plan_path = _relative_file(self.root, "plan.json")
        plan_sha = _sha256(plan_path)
        if (
            complete.get("schema") != SCHEMA_V2
            or complete.get("complete") is not True
            or manifest.get("schema") != SCHEMA_V2
            or complete.get("plan_sha256") != plan_sha
            or manifest.get("plan_sha256") != plan_sha
            or complete.get("manifest_sha256") != manifest_sha
        ):
            raise ValueError("Cache is incomplete or sealed metadata SHA256 changed")
        plan_metadata = _json(plan_path)
        validate_prompt_roles(plan_metadata, manifest, complete)
        self._text_plan_sha = _text_reuse_plan(manifest, plan_metadata, plan_sha)
        variants = manifest.get("variants")
        if (
            manifest.get("stride") != 20
            or manifest.get("horizon") != 32
            or isinstance(variants, bool)
            or not isinstance(variants, int)
            or variants < 2
            or complete.get("variants") != variants
        ):
            raise ValueError("Unsupported cache timeline or variant count")
        _validate_encoder(manifest.get("encoder"))
        index = manifest.get("index")
        if not isinstance(index, dict) or index.get("path") != "index.sqlite":
            raise ValueError("Missing sealed SQLite index")
        self._entry_metadata(index)
        self._index_path = _relative_file(self.root, index["path"])
        before = self._index_path.stat()
        if (
            before.st_size != index["size"]
            or _sha256(self._index_path) != index["sha256"]
        ):
            raise ValueError("Cache index SHA256 differs from sealed manifest")
        after = self._index_path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Cache index changed while being verified")
        shards = manifest.get("shards")
        if not isinstance(shards, dict) or not shards:
            raise ValueError("Missing sealed shard inventory")
        for path, entry in shards.items():
            if (
                not isinstance(path, str)
                or not re.fullmatch(r"shards/(text|rgb)/rank_\d+_shard_\d+\.h5", path)
                or not isinstance(entry, dict)
                or entry.get("kind") != path.split("/")[1]
            ):
                raise ValueError("Invalid cache shard path or kind")
            if {"plan_sha256", "source_plan_sha256", "text_plan_sha256"}.intersection(
                entry
            ):
                raise ValueError("Per-shard plan overrides are not permitted")
            self._entry_metadata(entry)
        self._shards = deepcopy(shards)
        self._plan_sha, self._variants = plan_sha, variants
        self._metadata = {
            key: deepcopy(manifest.get(key))
            for key in (
                "encoder",
                "stride",
                "horizon",
                "variants",
                "cleaning",
                "augmentation",
                "source_binding",
                "raw_root",
            )
        }
        self._metadata.update(
            {
                key: deepcopy(manifest[key])
                for key in (
                    "color_contract",
                    "observation_color_contract",
                    "text_reuse",
                )
                if key in manifest
            }
        )
        self._metadata.update(
            schema=SCHEMA_V2,
            prompt_roles=CACHED_PROMPT_ROLES,
            plan_sha256=plan_sha,
            manifest_sha256=manifest_sha,
            index_sha256=index["sha256"],
            split=split,
            sealed_complete=True,
            source_cache_identity=deepcopy(plan_metadata.get("source_cache_identity")),
            verification="sealed_index_and_per_record_sha256",
            immutability="no_filesystem_checks_on_memory_cache_hits",
            rgb_prefetch="selected_variant_bounded_chunk_current_frame_return_only",
            rgb_prefetch_max_frames=RGB_PREFETCH_FRAMES,
            rgb_cache_bytes=self.rgb_cache_bytes,
        )
        self._reset_runtime()
        self.rows = []
        try:
            db = self._connection()
            schemas = {
                "texts": ("id", "prompt", "shard", "row", "sha256"),
                "episodes": ("id", "row_json", "entry_json"),
                "roles": ("episode_id", "role", "ordinal", "prompt_id"),
            }
            for table, columns in schemas.items():
                if (
                    tuple(row[1] for row in db.execute(f"PRAGMA table_info({table})"))
                    != columns
                ):
                    raise ValueError("Unsupported cache index table schema: " + table)
            if db.execute("SELECT COUNT(*) FROM episodes").fetchone()[
                0
            ] != complete.get("episodes") or db.execute(
                "SELECT COUNT(*) FROM texts"
            ).fetchone()[0] != complete.get("prompts"):
                raise ValueError(
                    "Sealed index does not cover the complete source/prompt inventory"
                )
            if db.execute(
                "SELECT 1 FROM roles WHERE role NOT IN ('seen','unseen') OR ordinal < 0 LIMIT 1"
            ).fetchone():
                raise ValueError("Invalid instruction role or ordinal")
            counts = {
                (row[0], row[1]): row[2:]
                for row in db.execute(
                    "SELECT episode_id,role,COUNT(*),MIN(ordinal),MAX(ordinal) FROM roles "
                    "GROUP BY episode_id,role"
                )
            }
            available_tasks = set()
            expected_role_groups = 0
            dataclass_names = {field.name for field in fields(CacheEpisode)}
            for identity, row_json, entry_json in db.execute(
                "SELECT id,row_json,entry_json FROM episodes ORDER BY rowid"
            ):
                raw, entry = json.loads(row_json), json.loads(entry_json)
                row = CacheEpisode.from_dict(
                    {k: v for k, v in raw.items() if k in dataclass_names}
                )
                if row.id != identity:
                    raise ValueError("Indexed source episode identity differs")
                self._validate_entry(row, entry)
                role_specs = {}
                for role in ("seen", "unseen"):
                    ids = list(
                        dict.fromkeys(
                            hashlib.sha256(
                                DEFAULT_PROMPT.format(task=s).encode()
                            ).hexdigest()
                            for s in getattr(row, role)
                        )
                    )
                    expected = (len(ids), 0, len(ids) - 1) if ids else None
                    if counts.get((identity, role)) != expected:
                        raise ValueError(
                            role.capitalize()
                            + " instruction index coverage differs from source episode"
                        )
                    expected_role_groups += bool(ids)
                    role_specs[role] = (role, len(ids), _role_digest(ids))
                available_tasks.add(row.task)
                if (
                    row.split == split
                    and row.phase in phases
                    and (names is None or row.task in names)
                ):
                    selected = [
                        role_specs[role]
                        for role in self.instruction_roles_by_phase[row.phase]
                    ]
                    self._register_instruction_counts(
                        row, [(role, count) for role, count, _ in selected]
                    )
                    instructions = _RoleInstructions(self, row.id, selected)
                    self.rows.append(
                        CachedEpisode(
                            row.id,
                            str(self.root / entry["path"]),
                            row.task,
                            row.phase,
                            row.raw_length,
                            instructions,
                            entry["frames"],
                            row.split,
                        )
                    )
            if len(counts) != expected_role_groups:
                raise ValueError(
                    "Unexpected instruction roles outside source episode inventory"
                )
            if names is not None and not set(names) <= available_tasks:
                raise ValueError("Selected tasks are absent from cache")
            if not self.rows:
                raise ValueError("Selected cache split has no trajectories")
            self._metadata.update(self._sampling_metadata())
        finally:
            self.close()

    def _reset_runtime(self):
        self._pid, self._db = os.getpid(), None
        self._handles, self._files = OrderedDict(), OrderedDict()
        self._texts, self._roles = OrderedDict(), OrderedDict()
        self._rgbs, self._rgb_bytes = OrderedDict(), 0

    def _ensure_process(self):
        if self._pid != os.getpid():
            self.close()
            self._reset_runtime()

    def _connection(self):
        self._ensure_process()
        if self._db is None:
            self._db = sqlite3.connect(
                self._index_path.as_uri() + "?mode=ro&immutable=1", uri=True
            )
            self._db.execute("PRAGMA query_only=ON")
            self._db.execute("PRAGMA cache_size=-8192")
        return self._db

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(
            _pid=None,
            _db=None,
            _handles=OrderedDict(),
            _files=OrderedDict(),
            _texts=OrderedDict(),
            _roles=OrderedDict(),
            _rgbs=OrderedDict(),
            _rgb_bytes=0,
        )
        return state

    def close(self):
        for handle in self._handles.values():
            handle.close()
        if self._db is not None:
            self._db.close()
            self._db = None
        for cache in (self._handles, self._files, self._texts, self._roles, self._rgbs):
            cache.clear()
        self._rgb_bytes = 0

    def _validate_entry(self, row, entry):
        group = "episodes/" + hashlib.sha256(row.id.encode()).hexdigest()
        if (
            not isinstance(entry, dict)
            or entry.get("id") != row.id
            or entry.get("group") != group
            or entry.get("path") not in self._shards
            or self._shards[entry["path"]]["kind"] != "rgb"
            or any(
                entry.get(key) != getattr(row, key)
                for key in ("task", "phase", "split")
            )
            or entry.get("variants") != self._variants
            or _integer(entry.get("frames"), "retained frames") < 1
        ):
            raise ValueError("Episode cache identity differs from source index")
        _digest(entry.get("labels_sha256"), "episode labels")
        _digest(entry.get("rgb_checksums_sha256"), "RGB checksum table")

    def _shard(self, path, kind):
        self._ensure_process()
        if path in self._handles:
            self._handles.move_to_end(path)
            return self._handles[path]
        import h5py

        entry = self._shards.get(path)
        if entry is None or entry["kind"] != kind:
            raise ValueError("Record references an unknown shard or wrong kind")
        resolved = _relative_file(self.root, path)
        if resolved.stat().st_size != entry["size"]:
            raise ValueError("Sealed shard size changed")
        handle = h5py.File(resolved, "r", rdcc_nbytes=1024 * 1024)
        try:
            if (
                handle.attrs.get("schema") != SCHEMA_V2
                or handle.attrs.get("kind") != kind
                or handle.attrs.get("plan_sha256")
                != (self._text_plan_sha if kind == "text" else self._plan_sha)
            ):
                raise ValueError("Shard header identity differs from sealed plan")
            if (
                kind == "text"
                and handle.attrs.get("text_padding")
                != "zero_context_after_tokenizer_length_then_all_true_mask"
            ):
                raise ValueError("Unsupported text shard padding contract")
        except BaseException:
            handle.close()
            raise
        if len(self._handles) >= self.max_open_files:
            _, old = self._handles.popitem(last=False)
            old.close()
        self._handles[path] = handle
        return handle

    def _role_ids(self, episode_id, role, count, digest):
        self._ensure_process()
        key = (episode_id, role)
        if key not in self._roles:
            records = (
                self._connection()
                .execute(
                    "SELECT ordinal,prompt_id FROM roles WHERE episode_id=? AND role=? ORDER BY ordinal",
                    (episode_id, role),
                )
                .fetchall()
            )
            ids = tuple(_digest(row[1], role + " prompt") for row in records)
            if (
                len(ids) != count
                or [r[0] for r in records] != list(range(count))
                or _role_digest(ids) != digest
            ):
                raise ValueError(
                    role.capitalize() + " instruction roles differ from source episode"
                )
            if len(self._roles) >= self.max_open_files:
                self._roles.popitem(last=False)
            self._roles[key] = ids
        self._roles.move_to_end(key)
        return self._roles[key]

    def _file(self, index):
        self._ensure_process()
        if index in self._files:
            self._files.move_to_end(index)
            return self._files[index]
        row = self.rows[index]
        record = (
            self._connection()
            .execute("SELECT entry_json FROM episodes WHERE id=?", (row.id,))
            .fetchone()
        )
        if record is None:
            raise ValueError("Missing indexed episode")
        entry = json.loads(record[0])
        self._validate_entry(row, entry)
        group = self._shard(entry["path"], "rgb")[entry["group"]]
        if (
            group.attrs.get("schema") != SCHEMA_V2
            or group.attrs.get("episode_id") != row.id
            or group.attrs.get("plan_sha256") != self._plan_sha
            or group.attrs.get("rgb_dtype") != "bfloat16_bits_uint16"
        ):
            raise ValueError("Episode group header identity differs from sealed plan")
        payload = {name: np.asarray(group[name][:]) for name in LABEL_NAMES}
        if (
            group.attrs.get("labels_sha256") != entry["labels_sha256"]
            or labels_sha256(payload) != entry["labels_sha256"]
        ):
            raise ValueError("Episode labels SHA256 differs from sealed index")
        frames, segments = payload["frames"], payload["segments"]
        expected_segments = np.cumsum(np.r_[True, np.diff(frames) != 20]) - 1
        if (
            frames.shape != (row.retained_count,)
            or not np.issubdtype(frames.dtype, np.integer)
            or frames[0] != 0
            or frames[-1] != (row.raw_length - 2) // 20 * 20
            or np.any(np.diff(frames) <= 0)
            or np.any(frames % 20 != 0)
            or np.any(frames >= row.raw_length - 1)
            or not np.issubdtype(segments.dtype, np.integer)
            or not np.array_equal(segments, expected_segments)
        ):
            raise ValueError("Cached original-frame chronology/segments are invalid")
        latents, digests = group["rgb_latents"], group["rgb_sha256"]
        if (
            latents.shape != (self._variants, row.retained_count, *LATENT_SHAPE)
            or latents.dtype != np.uint16
            or digests.shape != (self._variants, row.retained_count)
            or digests.dtype != np.dtype("S64")
        ):
            raise ValueError("Invalid RGB shard record shape or dtype")
        checksums = np.asarray(digests[:])
        if (
            group.attrs.get("rgb_checksums_sha256") != entry["rgb_checksums_sha256"]
            or array_sha256(checksums) != entry["rgb_checksums_sha256"]
        ):
            raise ValueError("RGB checksum table SHA256 differs from sealed index")
        proprio, action, valid = (
            payload[name] for name in ("proprio", "action", "valid")
        )
        if (
            proprio.shape != (row.retained_count, 14)
            or action.shape != (row.retained_count, 32, 14)
            or not np.issubdtype(proprio.dtype, np.floating)
            or not np.issubdtype(action.dtype, np.floating)
            or not np.isfinite(proprio).all()
            or not np.isfinite(action).all()
            or valid.shape != (row.retained_count, 32)
            or valid.dtype != np.bool_
            or not np.array_equal(
                valid,
                np.stack(
                    [
                        make_action_window(int(frame), row.raw_length)[1]
                        for frame in frames
                    ]
                ),
            )
        ):
            raise ValueError("Cache state/action targets or valid masks are invalid")
        segment_start = np.maximum.accumulate(
            np.where(np.r_[True, np.diff(segments) != 0], np.arange(len(frames)), 0)
        )
        payload.update(
            entry=entry,
            rgb_checksums=checksums,
            decisions=np.arange(len(frames)) - segment_start,
        )
        if len(self._files) >= self.max_open_files:
            self._files.popitem(last=False)
        self._files[index] = payload
        return payload

    def _rgb(self, cached, variant, frame_index):
        entry = cached["entry"]
        bytes_per_frame = int(np.prod(LATENT_SHAPE)) * 2
        chunk_frames = min(RGB_PREFETCH_FRAMES, self.rgb_cache_bytes // bytes_per_frame)
        start = (
            frame_index // chunk_frames * chunk_frames if chunk_frames else frame_index
        )
        end = (
            min(start + chunk_frames, entry["frames"])
            if chunk_frames
            else frame_index + 1
        )
        key = (entry["id"], variant, start)
        if key not in self._rgbs:
            group = self._shard(entry["path"], "rgb")[entry["group"]]
            bytes_needed = (end - start) * bytes_per_frame
            if chunk_frames:
                while (
                    self._rgbs and self._rgb_bytes + bytes_needed > self.rgb_cache_bytes
                ):
                    _, old = self._rgbs.popitem(last=False)
                    self._rgb_bytes -= old.nbytes
                    del old
                sequence = np.asarray(group["rgb_latents"][variant, start:end])
                for ordinal, bits in enumerate(sequence):
                    expected = _digest(
                        _ascii(cached["rgb_checksums"][variant, start + ordinal]),
                        "RGB record",
                    )
                    if array_sha256(bits) != expected:
                        raise ValueError("RGB record SHA256 differs from shard record")
                self._rgbs[key] = sequence
                self._rgb_bytes += sequence.nbytes
            else:
                bits = np.asarray(group["rgb_latents"][variant, frame_index])
                expected = _digest(
                    _ascii(cached["rgb_checksums"][variant, frame_index]), "RGB record"
                )
                if array_sha256(bits) != expected:
                    raise ValueError("RGB record SHA256 differs from shard record")
                return torch.from_numpy(bits.copy()).view(torch.bfloat16).contiguous()
        self._rgbs.move_to_end(key)
        bits = self._rgbs[key][frame_index - start].copy()
        return torch.from_numpy(bits).view(torch.bfloat16).contiguous()

    def _text(self, identity):
        self._ensure_process()
        if identity not in self._texts:
            record = (
                self._connection()
                .execute(
                    'SELECT prompt,shard,"row",sha256 FROM texts WHERE id=?',
                    (identity,),
                )
                .fetchone()
            )
            if record is None:
                raise ValueError("Missing indexed text record")
            prompt, path, ordinal, expected = record
            _digest(identity, "prompt identity")
            _digest(expected, "text record")
            if (
                not isinstance(prompt, str)
                or hashlib.sha256(prompt.encode()).hexdigest() != identity
            ):
                raise ValueError("Text prompt identity differs from indexed prompt")
            ordinal = _integer(ordinal, "text row")
            handle = self._shard(path, "text")
            contexts = handle["contexts"]
            if (
                len(contexts.shape) != 3
                or contexts.shape[1:] != TEXT_SHAPE
                or contexts.dtype != np.uint16
                or ordinal >= contexts.shape[0]
                or handle["prompt_id"].shape != (contexts.shape[0],)
                or handle["sha256"].shape != (contexts.shape[0],)
                or handle["prompt_id"].dtype != np.dtype("S64")
                or handle["sha256"].dtype != np.dtype("S64")
            ):
                raise ValueError("Invalid text shard record shape, dtype or index")
            if (
                _ascii(handle["prompt_id"][ordinal]) != identity
                or _ascii(handle["sha256"][ordinal]) != expected
            ):
                raise ValueError(
                    "Text shard record identity/checksum differs from sealed index"
                )
            bits = np.asarray(contexts[ordinal]).copy()
            if array_sha256(bits) != expected:
                raise ValueError("Text record SHA256 differs from sealed index")
            context = torch.from_numpy(bits).view(torch.bfloat16).contiguous()
            if not torch.isfinite(context).all():
                raise ValueError("Cached text context must be finite")
            if len(self._texts) >= self.text_cache_entries:
                self._texts.popitem(last=False)
            self._texts[identity] = (context, prompt)
        self._texts.move_to_end(identity)
        return self._texts[identity][0].clone(), torch.ones(
            TEXT_SHAPE[0], dtype=torch.bool
        )

    def _prompt(self, identity):
        return self._texts[identity][1]
