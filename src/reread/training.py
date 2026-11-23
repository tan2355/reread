import argparse
import hashlib
import json
import time
from collections import Counter
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch

import reread.backbone.observations as train
import reread.distributed as distributed
from reread.backbone.capture import FrozenSpatialFastWAM
from reread.checkpoint import execution_contract, save_checkpoint
from reread.config import ReReadConfig
from reread.data.cache_contract import CACHE_SCHEMA, require_rgb_cache
from reread.data.distributed import (
    assert_local_task_coverage,
    global_coverage_plan,
    iter_rank_batches,
    partition_episodes,
)
from reread.data.sampling import load_task_list, uniform_task_probabilities
from reread.engine import stream_updates
from reread.model import ReRead
from reread.utils import sha256_file

ROOT = Path(__file__).resolve().parents[2]
BATCH_PER_RANK = 64
TRAIN_STEPS = 20_000


def validate_production_config(config):
    if config != ReReadConfig():
        raise ValueError(
            "Production entry requires the fixed ~200M start0 RGB endpoint Memory3 D20/R20 contract"
        )


def parser():
    p = distributed.parser()
    p.description = "Uniform-task, pure-MSE training of ReRead with a frozen Fast-WAM backbone."
    p.set_defaults(
        config=str(ROOT / "configs/reread.json"),
        patch_projection="native_addmm",
        frozen_inference_graph=False,
        fused_fp64_rope=False,
        fused_visible_attention=False,
        timestep_cache=False,
    )
    p.add_argument(
        "--prefetch-frozen",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Opt-in one-batch frozen producer overlapping the current corrector",
    )
    return p


def validate_args(args, *, expected_steps=TRAIN_STEPS):
    distributed.validate_args(args, expected_steps=expected_steps)
    if args.prefetch_frozen and not args.preencode_next:
        raise ValueError("Frozen overlap requires --preencode-next")
    if (
        args.patch_projection not in ("native", "native_addmm")
        or args.fused_visible_attention
        or args.timestep_cache
    ):
        raise ValueError(
            "Endpoint correction allows native/native_addmm and verified Graph/RoPE only; unmasked/timestep-cache wrappers are disabled"
        )


def _clear_verified_inference(wrapper):
    try:
        wrapper.clear()
    except Exception as error:
        print(
            json.dumps(dict(event="graph_cleanup_warning", message=str(error))),
            flush=True,
        )


@contextmanager
def verified_endpoint_inference(frozen, args, context, sampling, *, on_calibrated=None):
    wrapper = None
    try:
        if args.frozen_inference_graph or args.fused_fp64_rope:

            def on_ready(report):
                context.require_all(
                    wrapper.graph_ready,
                    "Every rank must pass original-native actions and all K/V byte equality",
                )
                sampling["training_runtime"]["original_native_equivalence"] = dict(
                    all_ranks_passed=True,
                    ranks=context.world_size,
                    scope="first_real_batch_per_rank_not_full_dataset_proof",
                    rank_zero_evidence=report if context.rank == 0 else None,
                )
                print(
                    json.dumps(
                        dict(
                            event="verified_frozen_inference_ready",
                            rank=context.rank,
                            device=str(context.device),
                            validation=report,
                        )
                    ),
                    flush=True,
                )
                if on_calibrated is not None:
                    on_calibrated()

            wrapper = distributed.VerifiedCachedInference(
                frozen,
                device=context.device,
                batch_size=args.batch_size,
                on_ready=on_ready,
                use_graph=args.frozen_inference_graph,
                fused_fp64_rope=args.fused_fp64_rope,
            )
        yield wrapper if wrapper is not None else frozen
    finally:
        if wrapper is not None:
            try:
                sampling["training_runtime"]["final_rank_local_verified_inference"] = (
                    wrapper.stats()
                )
            finally:
                _clear_verified_inference(wrapper)


def _sampling_metadata(
    args, context, config, task_p, phase_p, coverage, cache_metadata, budget
):
    return dict(
        steps=args.steps,
        batch_per_rank=args.batch_size,
        world_size=context.world_size,
        global_batch=args.batch_size * context.world_size,
        grad_accum=1,
        task_sampling="uniform",
        task_probabilities=task_p,
        phase_probabilities=phase_p,
        per_batch_task_coverage=coverage,
        phase_target_source="explicit_1_to_10_decision_budget",
        instruction_roles_by_phase={
            "clean": ["seen"],
            "randomized": ["seen", "unseen"],
        },
        unseen_is_heldout_instruction=False,
        augmentation=cache_metadata["variants"] - 1,
        history_reset="episode_pass_cleaning_segment_prompt_variant",
        cpu_budget=budget,
        encoders_loaded=False,
        image_predictor=False,
        future_video_branch=False,
        feature_schema=config.feature_schema,
        model_schema=config.model_schema,
        history="visual_and_full_plan_detached_raw_pre_rope",
        target="expert_full_endpoint_actions",
        action_perturb_probability=config.action_perturb_probability,
        action_perturb_std=config.action_perturb_std,
        perturbation="candidate_actions_only_valid_rows_excluding_grippers",
        perturb_kv=False,
        action_input_compressed=False,
        reference_trajectory=f"unperturbed_native_D{config.inference_steps}_no_expert_actions",
        correction_start_decision=0,
        supervised_start_decision=0,
        prefix_warmup_decisions=0,
        observation_color_contract="native_robotwin_rgb_v1",
        corrector_forwards_per_decision=1,
        planned_consumed_decisions=args.steps * args.batch_size * context.world_size,
        base_calls_per_decision=config.inference_steps,
        training_reference_cost=f"one_complete_native_D{config.inference_steps}",
        residual_application="candidate_plus_residual_once_at_action_endpoint",
        loss="action_mse",
        loss_normalization="global_valid_action_elements",
        learning_rate_schedule=dict(
            total_steps=args.steps,
            warmup_steps=args.warmup_steps,
            peak_lr=args.lr,
            min_lr=args.min_lr,
        ),
        ddp=dict(
            backend="nccl",
            corrector_only=True,
            bucket_cap_mb=25,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            find_unused_parameters=False,
        ),
        kernel=args.patch_projection,
        patch_projection_validation="finite_and_bitwise_first_shape",
        frozen_inference_graph=args.frozen_inference_graph,
        fused_fp64_rope=args.fused_fp64_rope,
        training_runtime=dict(
            frozen_inference_graph=args.frozen_inference_graph,
            fused_fp64_rope=args.fused_fp64_rope,
            prefetch_frozen=args.prefetch_frozen,
            frozen_lookahead_batches=int(args.prefetch_frozen),
            calibration="first_real_batch_original_native_actions_and_complete_kv_byte_equality",
            fallback=False,
        ),
        metrics_scope=dict(
            loss_components_and_counts="global_all_ranks",
            grad_norm="DDP_synchronized_corrector_gradients",
            perturb_residual_history_and_local_wall="rank_zero_local_not_global",
        ),
    )


def _checkpoint_provenance(args, cache_metadata, rows):
    source_digest = sha256_file(ROOT / "manifest.json")
    if (
        args.source_manifest_sha256 is not None
        and args.source_manifest_sha256 != source_digest
    ):
        raise ValueError(
            "Requested source manifest SHA256 differs from this source tree"
        )
    return dict(
        source_manifest_sha256=source_digest,
        launch_id=args.launch_id,
        base_checkpoint=str(Path(args.checkpoint).resolve()),
        base_sha256=sha256_file(args.checkpoint),
        stats_sha256=sha256_file(args.stats),
        task_list_sha256=sha256_file(args.task_list),
        observation_cache_manifest_sha256=cache_metadata["manifest_sha256"],
        observation_cache=cache_metadata,
        selected_episode_ids_sha256=hashlib.sha256(
            "\n".join(sorted(row.id for row in rows)).encode()
        ).hexdigest(),
    )


def _record_consumed_samples(
    samples, coverage, rank, variant_count, task_phase, roles, variants
):
    assert_local_task_coverage(samples, coverage, rank)
    for sample in samples:
        task, phase = sample["episode_id"].split("/")[:2]
        role, variant = sample["instruction_role"], sample["cache_variant"]
        allowed = ("seen",) if phase == "clean" else ("seen", "unseen")
        if (
            role not in allowed
            or not 0 <= variant < variant_count
            or sample["image_augmented"] != (variant > 0)
        ):
            raise RuntimeError("Consumed data violates instruction/augmentation policy")
        task_phase[(task, phase)] += 1
        roles[(phase, role)] += 1
        variants[(phase, variant)] += 1


def run(args, context, budget):
    output = Path(args.output).resolve()
    context.rank_zero(lambda: distributed._claim_output(output, args.launch_id))
    dataset = tracker = None
    step = 0
    try:
        config = ReReadConfig.from_dict(json.loads(Path(args.config).read_text()))
        c = config
        validate_production_config(config)
        cfg = train.official_configuration(args.upstream)
        processor = train.official_processor(cfg, args.stats)
        task_p = uniform_task_probabilities(load_task_list(args.task_list))
        phase_p = {"clean": 1.0 / 11.0, "randomized": 10.0 / 11.0}
        from reread.data.dataset import CachedTrajectoryDataset
        from reread.data.provenance import validate_cached_encoder_source

        dataset = CachedTrajectoryDataset(
            args.cache_root,
            processor,
            split="train",
            tasks=sorted(task_p),
            phases=["clean", "randomized"],
            seed=args.seed,
            instruction_roles_by_phase={
                "clean": ("seen",),
                "randomized": ("seen", "unseen"),
            },
            augmentation=True,
        )
        if dataset.cache_metadata["schema"] != CACHE_SCHEMA:
            raise ValueError("Require sealed v8-compatible large-shard observations")
        require_rgb_cache(dataset.cache_metadata)
        validate_cached_encoder_source(dataset.cache_metadata["encoder"], args.upstream)
        rows = dataset.rows
        if len(task_p) != 50 or {row.task for row in rows} != set(task_p):
            raise ValueError("All 50 official tasks are required")
        coverage = global_coverage_plan(
            task_p,
            world_size=context.world_size,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        indices = partition_episodes(
            rows,
            context.rank,
            context.world_size,
            seed=args.seed,
            coverage_plan=coverage,
            decision_stride=c.decision_stride,
        )
        first = dataset[(0, indices[0], 0, 0, 0)]
        if "cached_encoded" not in first:
            raise ValueError("Raw RGB is not accepted by this training entry")
        del first
        dataset.close()
        context.agree(
            dict(
                config=config.to_dict(),
                cache=dataset.cache_metadata,
                tasks=task_p,
                phases=phase_p,
                coverage=coverage,
                steps=args.steps,
                seed=args.seed,
                inference_seed=args.inference_seed,
                batch=args.batch_size,
                lr=args.lr,
                min_lr=args.min_lr,
                warmup=args.warmup_steps,
                autocast=args.corrector_autocast,
                patch_projection=args.patch_projection,
                frozen_inference_graph=args.frozen_inference_graph,
                fused_fp64_rope=args.fused_fp64_rope,
                prefetch_frozen=args.prefetch_frozen,
            )
        )
        provenance = context.rank_zero(
            lambda: _checkpoint_provenance(args, dataset.cache_metadata, rows)
        )
        torch.manual_seed(args.seed)
        base = train.load_training_backbone(
            cfg,
            args.checkpoint,
            args.wan,
            str(context.device),
            cached_observations=True,
        )
        if any(
            getattr(base, name, None) is not None
            for name in ("vae", "text_encoder", "tokenizer")
        ):
            raise RuntimeError("Cached-only factory loaded an encoder")
        if any(p.requires_grad for p in base.parameters()):
            raise RuntimeError("WAM must remain frozen")
        frozen = FrozenSpatialFastWAM(base, config, cache_conditioning=True)
        model = ReRead(config).to(context.device)
        parameters = dict(corrector=sum(p.numel() for p in model.parameters()))
        ddp = context.wrap(model)
        torch.manual_seed(args.seed + 100003 * context.rank)
        optimizer = torch.optim.AdamW(ddp.parameters(), lr=args.lr, weight_decay=0.01)
        optimizer.param_groups[0]["lr"] = train.scheduled_lr(
            0, args.steps, args.warmup_steps, args.lr, args.min_lr
        )
        sampling = _sampling_metadata(
            args,
            context,
            config,
            task_p,
            phase_p,
            coverage,
            dataset.cache_metadata,
            budget,
        )
        run_metadata = dict(
            config=config.to_dict(),
            trainable_parameters=parameters,
            provenance=provenance,
            sampling=sampling,
            execution=execution_contract(config, args.inference_seed),
        )
        context.rank_zero(
            lambda: distributed._write_json(output / "run.json", run_metadata)
        )
        if context.rank == 0 and args.swanlab_project:
            import swanlab

            tracker = swanlab.init(
                project=args.swanlab_project,
                experiment_name=output.name,
                logdir=str(output / "swanlog"),
                mode="online",
                public=False,
                config={**sampling, **config.to_dict()},
            )
        print(
            json.dumps(
                dict(
                    event="endpoint_rank_ready",
                    rank=context.rank,
                    parameters=parameters,
                    device=str(context.device),
                )
            ),
            flush=True,
        )
        task_phase, roles, variants = Counter(), Counter(), Counter()
        started = previous = time.monotonic()
        progress = None

        def on_samples(samples):
            _record_consumed_samples(
                samples,
                coverage,
                context.rank,
                dataset.cache_metadata["variants"],
                task_phase,
                roles,
                variants,
            )

        def on_step(metrics):
            nonlocal step, previous, progress
            step += 1
            now = time.monotonic()
            metrics = dict(
                metrics,
                step=step,
                lr=optimizer.param_groups[0]["lr"],
                step_seconds=now - previous,
                elapsed_seconds=now - started,
            )
            previous = now
            if context.rank == 0:
                if tracker is not None:
                    tracker.log(
                        {f"train/{key}": value for key, value in metrics.items()},
                        step=step,
                    )
                if step <= 5 or step % 10 == 0:
                    print(json.dumps(dict(event="update", **metrics)), flush=True)
            if step <= 5 or step % 100 == 0 or step == args.steps:
                progress = distributed._global_progress(
                    context,
                    task_phase,
                    step=step,
                    tasks=sorted(task_p),
                    phases=["clean", "randomized"],
                    task_probabilities=task_p,
                    phase_probabilities=phase_p,
                    coverage_plan=coverage,
                    phase_role_counts=roles,
                    phase_variant_counts=variants,
                    variants=dataset.cache_metadata["variants"],
                    batch_per_rank=args.batch_size,
                )
                if progress["samples"] != step * args.batch_size * context.world_size:
                    raise RuntimeError("Consumed decision accounting differs")
                context.rank_zero(
                    lambda: distributed._write_json(
                        output / "sampling_progress.json", progress
                    )
                )
            if step < args.steps:
                optimizer.param_groups[0]["lr"] = train.scheduled_lr(
                    step, args.steps, args.warmup_steps, args.lr, args.min_lr
                )
            return step == args.steps

        source = iter_rank_batches(
            dataset,
            indices,
            rank=context.rank,
            seed=args.seed,
            batch_size=args.batch_size,
            decision_stride=c.decision_stride,
            batches=args.steps,
            workers=args.data_workers,
            task_probabilities=task_p,
            phase_probabilities=phase_p,
            coverage_plan=coverage,
            pin_memory=False,
        )
        if args.patch_projection == "native_addmm":
            from reread.backbone.patch_projection import current_frame_addmm_projection

            patch_scope = current_frame_addmm_projection(base, validate=True)
        else:
            patch_scope = nullcontext()
        with patch_scope:
            with verified_endpoint_inference(
                frozen,
                args,
                context,
                sampling,
                on_calibrated=lambda: context.rank_zero(
                    lambda: distributed._write_json(output / "run.json", run_metadata)
                ),
            ) as execution:
                result = stream_updates(
                    execution,
                    ddp,
                    optimizer,
                    source,
                    distributed=True,
                    corrector_autocast=args.corrector_autocast,
                    prefetch=args.preencode_next,
                    prefetch_frozen=args.prefetch_frozen,
                    inference_seed=args.inference_seed,
                    on_step=on_step,
                    on_samples=on_samples,
                )
        context.require_all(
            step == args.steps and result["updates"] == args.steps,
            "Incomplete update budget",
        )
        context.require_all(
            not any(p.grad is not None for p in base.parameters()),
            "Frozen WAM accumulated gradients",
        )
        final = output / f"endpoint_step_{args.steps}.pt"

        def finish():
            distributed._write_json(output / "run.json", run_metadata)
            save_checkpoint(
                final,
                model,
                step=step,
                provenance=provenance,
                sampling=sampling,
                seed=args.inference_seed,
                frozen_backbone_gradients=False,
            )
            distributed._write_json(
                output / "complete.json",
                dict(
                    complete=True,
                    step=step,
                    checkpoint=final.name,
                    checkpoint_sha256=sha256_file(final),
                    checkpoint_bytes=final.stat().st_size,
                    frozen_backbone_gradients=False,
                    sampling_progress=progress,
                    elapsed_seconds=time.monotonic() - started,
                ),
            )

        context.rank_zero(finish)
        return result
    except BaseException as error:
        failure = dict(
            complete=False,
            rank=context.rank,
            step=step,
            type=type(error).__name__,
            message=str(error),
        )
        try:
            distributed._write_json(
                output / f"failed_rank_{context.rank:02d}.json", failure
            )
            if context.rank == 0:
                distributed._write_json(output / "failed.json", failure)
        except OSError:
            pass
        raise
    finally:
        if dataset is not None:
            dataset.close()
        if tracker is not None:
            try:
                tracker.finish()
            except Exception as error:
                print(
                    json.dumps(
                        dict(event="swanlab_finish_warning", message=str(error))
                    ),
                    flush=True,
                )
