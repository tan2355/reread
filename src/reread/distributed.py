import argparse
import json
import re
from collections import Counter

import torch

import reread.backbone.observations as train

BATCH_PER_RANK = 64
TRAIN_STEPS = 20_000


def parser():
    p = train.parser()
    p.description = "Shared distributed validation and verified frozen inference."
    p.set_defaults(
        steps=TRAIN_STEPS,
        batch_size=BATCH_PER_RANK,
        grad_accum=1,
        data_workers=4,
        cpu_threads=4,
        phases=["clean", "randomized"],
        rgb_cache_mib=0,
        text_cache_mib=0,
        frozen_inference_graph=False,
        phase_weights=[1.0, 10.0],
    )
    p.add_argument("--distributed-timeout-seconds", type=int, default=600)
    p.add_argument(
        "--launch-id",
        required=True,
        help="Fresh launcher nonce; prevents cross-run failure-file writes",
    )
    p.add_argument("--source-manifest-sha256")
    p.add_argument(
        "--corrector-autocast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="BF16 operator autocast; parameters, optimizer and loss remain FP32",
    )
    p.add_argument(
        "--fused-fp64-rope",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Opt-in fused FP64 RoPE, first-batch checked against ORIGINAL native actions/K/V",
    )
    return p


def validate_args(args, *, expected_steps=TRAIN_STEPS):
    if not re.fullmatch(r"[0-9a-f]{32}", args.launch_id):
        raise ValueError("Expected a fresh 32-hex launcher identity")
    if args.source_manifest_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", args.source_manifest_sha256
    ):
        raise ValueError("Invalid release manifest SHA256")
    if not args.cache_root or not args.output or not args.task_list:
        raise ValueError("--cache-root, --output and --task-list are required")
    if (args.steps, args.batch_size, args.grad_accum) != (
        expected_steps,
        BATCH_PER_RANK,
        1,
    ):
        raise ValueError(
            f"This entry fixes {expected_steps} updates, batch {BATCH_PER_RANK} per rank, accumulation 1"
        )
    if args.tasks or args.phases != ["clean", "randomized"] or args.split != "train":
        raise ValueError(
            "This entry requires all 50 tasks and clean/randomized train splits"
        )
    if args.preflight or args.preflight_report:
        raise ValueError("Use the CPU preflight or launcher --print-plan separately")
    if args.vae_encoding != "native" or args.rgb_cache_mib or args.text_cache_mib:
        raise ValueError("Cached training uses no encoder or encoding cache")
    if (args.frozen_inference_graph or args.fused_fp64_rope) and (
        args.timestep_cache
        or args.fused_visible_attention
        or args.patch_projection != "native_addmm"
    ):
        raise ValueError(
            "Verified Graph/RoPE requires native_addmm and rejects timestep/unmasked optimizations"
        )
    if args.save_every != 0 or args.data_workers < 1 or args.cpu_threads < 1:
        raise ValueError(
            "Use final-only saving and at least one timeout-protected loader worker per rank"
        )
    if args.device != "cuda" or not args.batched_bank:
        raise ValueError(
            "Distributed entry uses local-rank CUDA devices and batched private memory banks"
        )
    if (
        not 0 <= args.warmup_steps < args.steps
        or not 0 <= args.min_lr <= args.lr
        or args.lr <= 0
    ):
        raise ValueError("Invalid learning rate schedule")
    if args.phase_weights != [1.0, 10.0]:
        raise ValueError(
            "Phase decision targets must be exactly clean:randomized = 1:10"
        )


class VerifiedCachedInference:

    def __init__(
        self,
        frozen,
        *,
        device,
        batch_size,
        on_ready=None,
        use_graph=True,
        fused_fp64_rope=False,
    ):
        self.frozen = frozen
        self.device = torch.device(device)
        if (
            self.device.type != "cuda"
            or self.device.index is None
            or torch.device(frozen.backbone.device) != self.device
        ):
            raise ValueError(
                "Verified cached graph must bind the explicit rank-local CUDA device"
            )
        if (
            batch_size < 1
            or getattr(frozen, "unmasked_attention", False)
            or getattr(frozen, "timestep_cache", False)
        ):
            raise ValueError(
                "Verified graph requires a fixed batch and no unmasked/timestep optimization"
            )
        if any(
            getattr(frozen.backbone, name, None) is not None
            for name in ("vae", "text_encoder", "tokenizer")
        ):
            raise ValueError(
                "Verified cached graph cannot instantiate or use VAE/T5/tokenizer"
            )
        if (
            type(use_graph) is not bool
            or type(fused_fp64_rope) is not bool
            or not (use_graph or fused_fp64_rope)
        ):
            raise ValueError(
                "Select explicit Graph and/or fused FP64 RoPE optimization flags"
            )
        self.batch_size, self.on_ready = batch_size, on_ready
        self.use_graph, self.fused_fp64_rope = use_graph, fused_fp64_rope
        self._graph = self._validation = None
        self._rope_scope = self._rope_stats = None
        self._ready = self._prepared = False
        self._seed = None
        self._failed = self._closed = False

    @property
    def backbone(self):
        return self.frozen.backbone

    @property
    def config(self):
        return self.frozen.config

    @property
    def graph_ready(self):
        return (
            self._ready
            and not self._failed
            and not self._closed
            and (
                not self.use_graph
                or (self._graph is not None and self._graph.graph_ready)
            )
        )

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.frozen, name)

    def _check_inputs(self, encoded):
        if self._failed or self._closed:
            raise RuntimeError("A failed/closed verified graph cannot be reused")
        if torch.device(self.backbone.device) != self.device:
            raise RuntimeError(
                "Verified graph backbone moved away from its rank-local device"
            )
        if set(encoded) != {"rgb_latents", "proprio", "context", "context_mask"}:
            raise ValueError(
                "Verified cached graph accepts only current cached RGB/text/proprio tensors"
            )
        if not all(
            isinstance(value, torch.Tensor)
            and value.ndim > 0
            and value.shape[0] == self.batch_size
            for value in encoded.values()
        ):
            raise ValueError(
                "Verified graph input differs from the fixed per-rank batch size"
            )

        if any(
            value.device.type != "cpu" and value.device != self.device
            for value in encoded.values()
        ):
            raise ValueError("Verified graph cannot read another rank CUDA device")
        if getattr(self.frozen, "unmasked_attention", False) or getattr(
            self.frozen, "timestep_cache", False
        ):
            raise ValueError(
                "Verified graph cannot be combined with unmasked/timestep optimization"
            )

    @torch.no_grad()
    def prepare_graph(self, *, seed=0, **encoded):
        self._check_inputs(encoded)
        if self._prepared:
            raise RuntimeError(
                "Verified graph is prepared exactly once before producer startup"
            )
        from reread.backbone.graph_candidate import (
            _compare_outputs,
            prepare_verified_graph,
        )

        self._prepared = True
        try:
            if getattr(self.backbone, "_newm_rope_candidate_active", False) or getattr(
                self.backbone, "_newm_prefill_candidate_active", False
            ):
                raise RuntimeError(
                    "Original-native calibration cannot start inside another optimization scope"
                )
            reference = self.frozen.extract_batch(**encoded, seed=seed)
            if self.fused_fp64_rope:
                from reread.backbone.rope import fused_fp64_rope

                scope = fused_fp64_rope(self.backbone)
                self._rope_stats = scope.__enter__()
                self._rope_scope = scope
            graph_report = None
            if self.use_graph:
                self._graph, graph_report = prepare_verified_graph(
                    self.frozen,
                    encoded,
                    seed=seed,
                    warmup_steps=3,
                    one_pass_signature=True,
                )
                actual = self._graph.extract_batch(**encoded, seed=seed)
            else:
                actual = self.frozen.extract_batch(**encoded, seed=seed)
            numerical = _compare_outputs(reference, actual)
            self._validation = dict(
                scope="first_real_batch_original_native_not_full_dataset_proof",
                original_native_equivalence=numerical,
                fused_fp64_rope=self.fused_fp64_rope,
                frozen_inference_graph=self.use_graph,
                graph_calibration=graph_report,
            )
            self._seed, self._ready = seed, True
            if not self.graph_ready:
                raise RuntimeError(
                    "Verified graph helper did not produce a ready graph"
                )

            if self.on_ready is not None:
                self.on_ready(self._validation)
            return self
        except BaseException:
            self._failed = True
            try:
                self.clear()
            except Exception:
                pass
            raise

    @torch.no_grad()
    def extract_batch(self, *, seed=0, **encoded):
        self._check_inputs(encoded)
        if not self.graph_ready:
            raise RuntimeError(
                "Prepare and verify this rank graph before starting GPU producers"
            )
        if seed != self._seed:
            raise ValueError("Verified cached inference uses the fixed calibrated seed")

        if self.use_graph:
            return self._graph.extract_batch(**encoded, seed=seed)
        return self.frozen.extract_batch(**encoded, seed=seed)

    def stats(self):
        return dict(
            device=str(self.device),
            batch_per_rank=self.batch_size,
            frozen_inference_graph=self.use_graph,
            fused_fp64_rope=self.fused_fp64_rope,
            validation=self._validation,
            rope=dict(self._rope_stats) if self._rope_stats is not None else None,
            rope_counter_scope="python_eager_warmup_capture_calls_excludes_graph_replays"
            if self.fused_fp64_rope
            else None,
            graph=self._graph.stats() if self._graph is not None else None,
        )

    def clear(self):
        self._closed = True
        graph, self._graph = self._graph, None
        scope, self._rope_scope = self._rope_scope, None
        try:
            if graph is not None:
                graph.clear()
        finally:
            if scope is not None:
                scope.__exit__(None, None, None)


VerifiedCachedGraph = VerifiedCachedInference


def _write_json(path, payload):
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    temporary.replace(path)


def _claim_output(path, launch_id):
    path.mkdir(parents=True, exist_ok=False)
    _write_json(path / "launch_identity.json", {"launch_id": launch_id})
    return str(path)


def _global_progress(
    context,
    task_phase_counts,
    *,
    step,
    tasks,
    phases,
    task_probabilities,
    phase_probabilities,
    coverage_plan=None,
    phase_role_counts=None,
    phase_variant_counts=None,
    variants=None,
    batch_per_rank=BATCH_PER_RANK,
):
    keys = [(task, phase) for task in tasks for phase in phases]
    counts = context.sum_counts([task_phase_counts[key] for key in keys])
    task_counts, phase_counts = Counter(), Counter()
    groups = {}
    for (task, phase), count in zip(keys, counts):
        task_counts[task] += count
        phase_counts[phase] += count
        groups[f"{task}/{phase}"] = count
    progress = dict(
        step=step,
        samples=sum(counts),
        task_counts=dict(task_counts),
        phase_counts=dict(phase_counts),
        task_phase_counts=groups,
        tasks_seen=sum(value > 0 for value in task_counts.values()),
        world_size=context.world_size,
        batch_per_rank=batch_per_rank,
        global_batch=batch_per_rank * context.world_size,
        target_task_probabilities=task_probabilities,
        target_phase_probabilities=phase_probabilities,
        accounting="consumed_only_excludes_prefetch",
    )
    if coverage_plan is not None:
        if any(value < step for value in task_counts.values()):
            raise RuntimeError(
                "Consumed task counts contradict the per-step coverage guarantee"
            )
        progress.update(
            per_batch_task_coverage=coverage_plan,
            coverage_asserted_consumed_updates=step,
        )
    if phase_role_counts is not None and phase_variant_counts is not None:
        role_keys = [(phase, role) for phase in phases for role in ("seen", "unseen")]
        variant_keys = [
            (phase, variant) for phase in phases for variant in range(variants)
        ]
        values = context.sum_counts(
            [phase_role_counts[key] for key in role_keys]
            + [phase_variant_counts[key] for key in variant_keys]
        )
        progress["phase_instruction_role_counts"] = {
            f"{phase}/{role}": count
            for (phase, role), count in zip(role_keys, values[: len(role_keys)])
        }
        progress["phase_variant_counts"] = {
            f"{phase}/{variant}": count
            for (phase, variant), count in zip(variant_keys, values[len(role_keys) :])
        }
        if (
            sum(values[: len(role_keys)]) != progress["samples"]
            or sum(values[len(role_keys) :]) != progress["samples"]
        ):
            raise RuntimeError(
                "Consumed instruction/variant accounting differs from sample count"
            )
    return progress
