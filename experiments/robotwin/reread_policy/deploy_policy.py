import sys as _site_sys
from pathlib import Path as _SitePath

_site_sys.path.insert(0, str(_SitePath(__file__).resolve().parents[3] / "src"))
import inspect
import logging
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from reread.backbone.capture import FrozenSpatialFastWAM
from reread.backbone.vae import prepare_current_vae
from reread.checkpoint import load_checkpoint, validate_checkpoint
from reread.config import ReReadConfig
from reread.data.demonstrations import DEFAULT_PROMPT
from reread.model import ReRead
from reread.policy import TemporalMemoryPolicy, validate_runtime_options

from .color_contract import policy_observation, validate_checkpoint_rgb_mode
from .file_certificate import (
    file_signature,
    issue_corrector_certificate,
    validate_corrector_certificate,
    validate_file_certificate,
)
from .live_graph import LiveGraphedFrozen, graph_mode
from .prompt_cache import FrozenPromptCache, text_cache_bytes

logger = logging.getLogger(__name__)
DEFAULT_WAN = "artifacts/ckpt/Wan2.2-TI2V-5B"


def deployment_options(args, payload):
    precision = args.get("endpoint_corrector_autocast")
    if not isinstance(precision, bool):
        raise ValueError("Explicit endpoint_corrector_autocast boolean is required")
    runtime = args.get("endpoint_runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("Explicit endpoint_runtime mapping is required")

    if set(runtime) != set(validate_runtime_options()):
        raise ValueError("endpoint_runtime must specify all known execution settings")
    runtime = validate_runtime_options(runtime)
    sampling = payload["sampling"]
    expected_runtime = dict(
        vae_encoding="native",
        patch_projection="native_addmm",
        fused_visible_attention=False,
        timestep_cache=False,
        rgb_cache_mib=0,
        text_cache_mib=0,
    )
    if (
        precision is not True
        or runtime != expected_runtime
        or sampling.get("kernel") != "native_addmm"
        or sampling.get("correction_start_decision") != 0
        or sampling.get("prefix_warmup_decisions") != 0
    ):
        raise ValueError("Require BF16 corrector and native frozen runtime")
    if "corrector_autocast" in sampling:
        saved = sampling["corrector_autocast"]
        if not isinstance(saved, bool) or saved != precision:
            raise ValueError(
                "Explicit corrector precision differs from checkpoint metadata"
            )
    if "runtime" in sampling:
        saved = sampling["runtime"]
        if not isinstance(saved, Mapping) or validate_runtime_options(saved) != runtime:
            raise ValueError("Explicit runtime differs from checkpoint metadata")
    return runtime, precision


def validate_rollout_args(args, payload):
    validate_checkpoint_rgb_mode(args.get("endpoint_rgb_mode", "native_rgb"), payload)
    config = validate_checkpoint(payload)
    deployment_options(args, payload)
    contract = payload["execution"]
    for key in (
        "action_horizon",
        "replan_steps",
        "num_inference_steps",
        "sigma_shift",
        "seed",
    ):
        raw = args[key]
        if isinstance(raw, bool):
            raise ValueError(f"RoboTwin {key} must not be boolean")
        value = float(raw) if key == "sigma_shift" else int(raw)
        expected = contract[key]
        if (
            key != "sigma_shift" and isinstance(raw, float) and raw != value
        ) or value != expected:
            raise ValueError(
                f"RoboTwin {key} differs from the endpoint training execution contract"
            )
    for key in ("text_cfg_scale", "negative_prompt", "rand_device"):
        value = args.get(key, contract[key])
        if key == "text_cfg_scale":
            if isinstance(value, bool):
                raise ValueError("RoboTwin text_cfg_scale must not be boolean")
            value = float(value)
        if value != contract[key]:
            raise ValueError(
                f"RoboTwin {key} differs from the endpoint training contract"
            )
    return config


class ReReadPolicy:
    def __init__(
        self,
        base,
        checkpoint,
        config,
        *,
        endpoint_runtime,
        endpoint_corrector_autocast,
        endpoint_text_cache_mib=256,
        endpoint_rgb_mode="native_rgb",
        endpoint_frozen_graph="native",
    ):
        validated = validate_checkpoint(checkpoint)
        if not isinstance(config, ReReadConfig) or config != validated:
            raise ValueError(
                "Endpoint deployment configuration differs from checkpoint"
            )
        self.endpoint_rgb_mode = validate_checkpoint_rgb_mode(
            endpoint_rgb_mode, checkpoint
        )
        runtime, precision = deployment_options(
            dict(
                endpoint_runtime=endpoint_runtime,
                endpoint_corrector_autocast=endpoint_corrector_autocast,
            ),
            checkpoint,
        )
        self.base = base
        if base.replan_steps != config.decision_stride:
            raise ValueError(
                "Base action queue does not match the requested execution cadence"
            )

        self._infer_action_accepts_video_frames = (
            "num_video_frames" in inspect.signature(base.model.infer_action).parameters
        )
        self.corrector = ReRead(config).to(base.model.device)
        self.corrector.load_state_dict(checkpoint["model"], strict=True)
        self.corrector.eval()
        frozen = FrozenSpatialFastWAM(base.model, config, cache_conditioning=True)

        self.prompt_cache = FrozenPromptCache(
            base.model, max_bytes=text_cache_bytes(endpoint_text_cache_mib)
        )
        self.base.model.encode_prompt = self.prompt_cache.encode
        self.vae_execution = prepare_current_vae(
            base.model.vae.model, mode=runtime["vae_encoding"]
        )
        self.frozen_graph = None
        temporal_runtime = runtime
        if graph_mode(endpoint_frozen_graph) == "auto":
            self.frozen_graph = LiveGraphedFrozen(frozen, mode="auto")
            frozen = self.frozen_graph

            temporal_runtime = {**runtime, "patch_projection": "native"}
        self.temporal = TemporalMemoryPolicy(
            frozen,
            self.corrector,
            runtime=temporal_runtime,
            bf16_corrector=precision,
        )
        self.decision_index = 0

        self.base._infer_action_chunk = self._infer_action_chunk

    def _infer_action_chunk(self, observation, instruction):
        started = time.perf_counter() if self.base.timing_enabled else None

        image_observation = policy_observation(observation, self.endpoint_rgb_mode)
        image = self.base._build_robotwin_image_tensor(image_observation)
        state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        proprio = self.base._normalize_state(state)
        kwargs = {
            "episode_id": self.base.episode_count,
            "decision_index": self.decision_index,
            "observation_step": int(self.base.step_count),
            "prompt": DEFAULT_PROMPT.format(task=instruction),
            "input_image": image,
            "proprio": proprio,
            "negative_prompt": self.base.negative_prompt,
            "text_cfg_scale": self.base.text_cfg_scale,
            "seed": self.base.seed,
            "rand_device": self.base.rand_device,
            "tiled": self.base.tiled,
        }
        if self._infer_action_accepts_video_frames:
            kwargs["num_video_frames"] = self.base._num_video_frames
        result = self.temporal.act(**kwargs)
        self.decision_index += 1
        if started is not None:
            self.base._timing_rollout["infer_s"] += time.perf_counter() - started
        return self.base._denormalize_action(result["action"])[0]

    def step(self, task_env, observation):
        return self.base.step(task_env, observation)

    def reset(self):
        self.base.reset()
        self.temporal.reset()
        self.decision_index = 0
        self.prompt_cache.reset_rollout_stats()

    def should_request_observation(self):
        return self.base.should_request_observation()

    def reset_timing_rollout(self):
        self.base.reset_timing_rollout()
        self.prompt_cache.reset_rollout_stats()

    def get_timing_rollout(self):
        result = dict(self.base.get_timing_rollout())
        result.update(self.prompt_cache.rollout_stats())
        if self.frozen_graph is not None:
            result["resident_frozen_graph"] = self.frozen_graph.stats()
        return result

    def get_prompt_cache_stats(self):
        return self.prompt_cache.stats()


def get_model(args):
    text_cache_bytes(args.get("endpoint_text_cache_mib", 256))
    graph_mode(args.get("endpoint_frozen_graph", "native"))
    weights = Path(args["corrector_checkpoint"]).resolve()
    expected_corrector_sha = args.get("endpoint_corrector_sha256")
    corrector_certificate = args.get("_endpoint_preverified_corrector")
    corrector_before = None
    if "_endpoint_preverified_corrector" in args:
        validate_corrector_certificate(
            corrector_certificate, path=weights, expected_sha256=expected_corrector_sha
        )
    elif expected_corrector_sha is not None:
        from reread.utils import sha256_file

        corrector_before = file_signature(weights)
        if sha256_file(weights) != expected_corrector_sha:
            raise ValueError("Corrector file differs from requested SHA256")
    certified_files = "_endpoint_preverified_files" in args
    if certified_files:
        payload = load_checkpoint(weights)
        validate_file_certificate(
            args["_endpoint_preverified_files"],
            base_path=args["ckpt_setting"],
            stats_path=args["dataset_stats_path"],
            provenance=payload["provenance"],
        )
    else:
        payload = load_checkpoint(
            weights,
            base_path=args["ckpt_setting"],
            stats_path=args["dataset_stats_path"],
        )
    if corrector_before is not None:
        corrector_certificate = issue_corrector_certificate(
            corrector_before, file_signature(weights), expected_corrector_sha
        )
    if corrector_certificate is not None:
        validate_corrector_certificate(
            corrector_certificate, path=weights, expected_sha256=expected_corrector_sha
        )
    config = validate_rollout_args(args, payload)
    runtime, precision = deployment_options(args, payload)
    from fastwam_policy import deploy_policy as upstream
    from omegaconf import OmegaConf

    cfg = upstream._compose_sim_cfg(
        args.get("sim_cfg_path"), args.get("sim_cfg_name"), args.get("sim_task")
    )
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    if model_cfg.get("_target_") != "fastwam.runtime.create_fastwam":
        raise ValueError("Expected the released Fast-WAM backbone")
    wan = str(Path(args.get("wan", DEFAULT_WAN)).resolve())
    model_cfg.model_id = model_cfg.tokenizer_model_id = wan
    model_cfg.redirect_common_files = False
    model_cfg.skip_dit_load_from_pretrain = True
    model_cfg.action_dit_pretrained_path = None
    base = upstream.WorldActionRobotWinPolicy(
        model_cfg=model_cfg,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(Path(args["ckpt_setting"]).resolve()),
        dataset_stats_path=Path(args["dataset_stats_path"]).resolve(),
        device=str(args.get("device", "cuda")),
        model_dtype=torch.bfloat16,
        action_horizon=config.horizon,
        replan_steps=int(args["replan_steps"]),
        num_inference_steps=config.inference_steps,
        sigma_shift=config.sigma_shift,
        seed=int(args["seed"]),
        text_cfg_scale=1.0,
        negative_prompt="",
        rand_device="cpu",
        tiled=upstream._parse_bool(args.get("tiled", False)),
        timing_enabled=upstream._parse_bool(args.get("timing_enabled", True)),
        num_video_frames=(int(cfg.data.train.num_frames) - 1)
        // int(cfg.data.train.action_video_freq_ratio)
        + 1,
    )
    logger.info(
        "Loaded endpoint checkpoint %s, step %d, schema %s, corrector BF16=%s, RGB=%s",
        weights,
        payload["step"],
        payload["schema"],
        precision,
        args.get("endpoint_rgb_mode", "native_rgb"),
    )
    policy = ReReadPolicy(
        base,
        payload,
        config,
        endpoint_runtime=runtime,
        endpoint_corrector_autocast=precision,
        endpoint_text_cache_mib=args.get("endpoint_text_cache_mib", 256),
        endpoint_frozen_graph=args.get("endpoint_frozen_graph", "native"),
        endpoint_rgb_mode=args.get("endpoint_rgb_mode", "native_rgb"),
    )
    if certified_files:
        validate_file_certificate(
            args["_endpoint_preverified_files"],
            base_path=args["ckpt_setting"],
            stats_path=args["dataset_stats_path"],
            provenance=payload["provenance"],
        )
    if corrector_certificate is not None:
        validate_corrector_certificate(
            corrector_certificate, path=weights, expected_sha256=expected_corrector_sha
        )
    return policy


def encode_obs(observation):
    return observation


def eval(task_env, model, observation):
    model.step(task_env, observation)


def reset_model(model):
    model.reset()
