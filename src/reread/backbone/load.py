from collections.abc import Mapping
from pathlib import Path

import torch
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.helpers.loader import _validate_dit_config
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from omegaconf import DictConfig, OmegaConf

_TARGET = "fastwam.runtime.create_fastwam"
_MODEL_KEYS = {
    "_target_",
    "model_id",
    "tokenizer_model_id",
    "tokenizer_max_len",
    "load_text_encoder",
    "proprio_dim",
    "video_dit_config",
    "action_dit_config",
    "action_dit_pretrained_path",
    "skip_dit_load_from_pretrain",
    "video_scheduler",
    "action_scheduler",
    "loss",
    "mot_checkpoint_mixed_attn",
    "compile_training_denoise",
    "redirect_common_files",
    "model_dtype",
    "device",
}


def _mapping(value, name):
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True, throw_on_missing=True)
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a resolved mapping")
    result = dict(value)
    if any(not isinstance(key, str) for key in result):
        raise ValueError(f"{name} keys must be strings")
    return result


def _scheduler(value, name, *, required):
    value = _mapping(value if value is not None else {}, name)
    keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    if set(value) - keys or (required and keys - set(value)):
        raise ValueError(
            f"{name} must contain only train_shift/infer_shift/num_train_timesteps"
        )
    result = {
        "train_shift": 5.0,
        "infer_shift": 5.0,
        "num_train_timesteps": 1000,
    } | value
    for key in ("train_shift", "infer_shift"):
        if (
            isinstance(result[key], bool)
            or not isinstance(result[key], (int, float))
            or not 0 < result[key] < float("inf")
        ):
            raise ValueError(f"{name}.{key} must be positive and finite")
    count = result["num_train_timesteps"]
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError(f"{name}.num_train_timesteps must be a positive integer")
    return result


class CachedOnlyFastWAM(FastWAM):

    def __init__(self, *args, **kwargs):
        if args or any(
            kwargs.get(name) is not None
            for name in ("vae", "text_encoder", "tokenizer")
        ):
            raise ValueError(
                "CachedOnlyFastWAM must be constructed without encoders or tokenizer"
            )
        super().__init__(
            **(kwargs | {"vae": None, "text_encoder": None, "tokenizer": None})
        )
        self.encoding_mode = "sealed_observations_only"

    def to(self, *args, **kwargs):
        torch.nn.Module.to(self, *args, **kwargs)
        parameter = next(self.parameters(), None)
        if parameter is not None:
            self.device, self.torch_dtype = parameter.device, parameter.dtype
        return self

    @staticmethod
    def _requires_encoder(*args, **kwargs):
        raise RuntimeError(
            "Cache-only FastWAM requires sealed current RGB latents and T5 contexts; raw image and text encoding are unavailable"
        )

    encode_prompt = _requires_encoder
    _encode_input_image_latents_tensor = _requires_encoder

    @classmethod
    def from_wan22_pretrained(cls, *args, **kwargs):
        raise RuntimeError(
            "Use load_cached_training_backbone; the upstream factory loads encoders"
        )

    def load_checkpoint(self, path, optimizer=None):
        if optimizer is not None:
            raise ValueError("Cache-only frozen backbone does not restore an optimizer")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or "mot" not in payload:
            raise ValueError(
                "Complete matching RGB and Action expert checkpoint required; legacy video-only checkpoints are unsupported"
            )
        modules = {"mot": self.mot}
        if self.proprio_encoder is not None:
            modules["proprio_encoder"] = self.proprio_encoder
        elif "proprio_encoder" in payload:
            raise ValueError(
                "Checkpoint has a proprio encoder but configuration does not"
            )

        for name, module in modules.items():
            state = payload.get(name)
            expected = module.state_dict()
            if not isinstance(state, Mapping) or set(state) != set(expected):
                raise ValueError(f"Complete matching {name} checkpoint required")
            for key, reference in expected.items():
                value = state[key]
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != reference.shape
                    or value.is_floating_point() != reference.is_floating_point()
                    or value.is_complex()
                ):
                    raise ValueError(
                        f"Checkpoint tensor identity/shape mismatch: {name}.{key}"
                    )
        for name, module in modules.items():
            module.load_state_dict(payload[name], strict=True)
        return payload


def _create_cached_fastwam(model_config, *, device, torch_dtype):
    cfg = _mapping(model_config, "model configuration")
    if cfg.get("_target_") != _TARGET:
        raise ValueError(f"Cache-only training requires official target {_TARGET}")
    if set(cfg) - _MODEL_KEYS:
        raise ValueError(
            f"Unsupported cache-only model configuration keys: {sorted(set(cfg) - _MODEL_KEYS)}"
        )
    if torch_dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Unsupported floating backbone dtype")
    video = _validate_dit_config(
        _mapping(cfg.get("video_dit_config"), "video_dit_config")
    )
    action = _mapping(cfg.get("action_dit_config"), "action_dit_config")
    if video.get("video_attention_mask_mode") != "first_frame_causal" or video.get(
        "action_conditioned", False
    ):
        raise ValueError(
            "Cache-only action inference requires first_frame_causal, non-action-conditioned RGB expert"
        )
    for key in ("num_heads", "attn_head_dim", "num_layers", "text_dim"):
        if action.get(key) != video.get(key):
            raise ValueError(f"Action and video expert {key} must match")
    for name in ("mot_checkpoint_mixed_attn", "compile_training_denoise"):
        if not isinstance(cfg.get(name, False), bool):
            raise ValueError(f"{name} must be boolean")
    if cfg.get("mot_checkpoint_mixed_attn", False) or cfg.get(
        "compile_training_denoise", False
    ):
        raise ValueError(
            "Cache-only frozen action inference requires uncompiled, non-checkpointed MoT"
        )
    if video.get("use_gradient_checkpointing", False) or action.get(
        "use_gradient_checkpointing", False
    ):
        raise ValueError("Frozen cache-only experts cannot use gradient checkpointing")
    proprio_dim = cfg.get("proprio_dim")
    if proprio_dim is not None and (
        isinstance(proprio_dim, bool)
        or not isinstance(proprio_dim, int)
        or proprio_dim <= 0
    ):
        raise ValueError("proprio_dim must be a positive integer or None")
    video_schedule = _scheduler(
        cfg.get("video_scheduler"), "video_scheduler", required=False
    )
    action_schedule = _scheduler(
        cfg.get("action_scheduler"), "action_scheduler", required=True
    )
    loss = _mapping(cfg["loss"] if cfg.get("loss") is not None else {}, "loss")
    if set(loss) - {"lambda_video", "lambda_action"}:
        raise ValueError("Unsupported official loss configuration")

    video_expert = WanVideoDiT(**video).to(device=device, dtype=torch_dtype)
    action_expert = ActionDiT.from_pretrained(
        action_dit_config=action,
        action_dit_pretrained_path=None,
        skip_dit_load_from_pretrain=True,
        device=device,
        torch_dtype=torch_dtype,
    )
    mot = MoT(
        mixtures={"video": video_expert, "action": action_expert},
        mot_checkpoint_mixed_attn=False,
    )
    return CachedOnlyFastWAM(
        video_expert=video_expert,
        action_expert=action_expert,
        mot=mot,
        text_dim=video["text_dim"],
        proprio_dim=proprio_dim,
        device=device,
        torch_dtype=torch_dtype,
        video_train_shift=video_schedule["train_shift"],
        video_infer_shift=video_schedule["infer_shift"],
        video_num_train_timesteps=video_schedule["num_train_timesteps"],
        action_train_shift=action_schedule["train_shift"],
        action_infer_shift=action_schedule["infer_shift"],
        action_num_train_timesteps=action_schedule["num_train_timesteps"],
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        compile_training_denoise=False,
    )


def load_cached_training_backbone(
    model_config, checkpoint, device, *, torch_dtype=torch.bfloat16
):
    checkpoint = Path(checkpoint).resolve(strict=True)
    if not checkpoint.is_file():
        raise ValueError("Backbone checkpoint must be a regular file")
    base = _create_cached_fastwam(model_config, device=device, torch_dtype=torch_dtype)
    base.load_checkpoint(str(checkpoint))
    base.model_paths = {
        "video_dit": str(checkpoint),
        "action_dit_backbone": str(checkpoint),
        "vae": None,
        "text_encoder": None,
        "tokenizer": None,
    }
    return base.requires_grad_(False).eval()
