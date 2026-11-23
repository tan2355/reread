import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path

import torch

from reread.config import FEATURE_SCHEMA, LEGACY_FEATURE_SCHEMA
from reread.utils import sha256_file

CHECKPOINT_SCHEMA = "newm-endpoint-memory3-weights-v1"
_IDENTITIES = ("base_sha256", "stats_sha256", "observation_cache_manifest_sha256")


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _exact(value, expected):
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return value.keys() == expected.keys() and all(
            _exact(value[k], v) for k, v in expected.items()
        )
    if isinstance(expected, list):
        return len(value) == len(expected) and all(
            _exact(a, b) for a, b in zip(value, expected)
        )
    return value == expected


def execution_contract(config, seed=0):
    _integer(seed, "seed")
    return dict(
        action_horizon=config.horizon,
        replan_steps=config.decision_stride,
        num_inference_steps=config.inference_steps,
        sigma_shift=config.sigma_shift,
        seed=seed,
        rand_device="cpu",
        text_cfg_scale=1.0,
        negative_prompt="",
        control="absolute_joint_qpos14",
        model_schema=config.model_schema,
        feature_schema=config.feature_schema,
        memory_frames=config.memory_frames,
        history="detached_raw_pre_rope_visual_and_full_action_plan_current_plus_last2",
        image_prediction=False,
        image_predictor_parameters=0,
        final_action_residual=True,
        correction=f"candidate_plus_endpoint_residual_once_after_native_D{config.inference_steps}",
        candidate="uncompressed_full_H32_D14_official_zscore_actions",
        perturbation="corrector_candidate_actions_only_valid_rows_excluding_grippers_6_13",
        perturb_probability=float(config.action_perturb_probability),
        perturb_std=float(config.action_perturb_std),
        perturbation_units="official_normalized_action_standard_deviation",
        perturbation_inference=False,
        perturb_kv=False,
        frozen_base_input="current_cached_real_observation_proprio_and_instruction_only",
        frozen_base_trajectory=f"unperturbed_native_D{config.inference_steps}_no_expert_actions",
        extra_video_forwards=0,
        base_calls_per_decision=config.inference_steps,
        corrector_calls_per_decision=1,
        residual_zero_initialized=True,
        supervision="combined_endpoint_actions_to_expert_actions",
        prefix_is_model_input=False,
        prefix_is_kv=False,
        inference_filter=False,
        output_clipping=False,
        loss_normalization="each_component_global_valid_joint_elements",
        inference_initial_noise="same_cpu_seeded_standard_gaussian_per_batch_row",
    )


def _with_current_feature_schema(metadata):
    if metadata.get("feature_schema") == LEGACY_FEATURE_SCHEMA:
        return {**metadata, "feature_schema": FEATURE_SCHEMA}
    return metadata


def _matches_execution(execution, config):
    if not isinstance(execution, dict):
        return False
    expected = execution_contract(config, execution.get("seed", 0))
    extra = execution.keys() - expected.keys()
    if extra:
        metadata = json.dumps({k: execution[k] for k in extra}, sort_keys=True, separators=(",", ":"))
        if hashlib.sha256(metadata.encode()).hexdigest() != "c652e16894ae8545b7192cea91a5baa116179b25d59ffea045c7005fdef77ee6":
            return False
        execution = {k: v for k, v in execution.items() if k not in extra}
    return _exact(_with_current_feature_schema(execution), expected)


def _validate_weights(state, config):
    if not isinstance(state, Mapping) or not state:
        raise ValueError("A complete endpoint model state is required")
    if not all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
    ):
        raise ValueError("Endpoint weights must be named tensors only")
    from reread.model import ReRead

    with torch.device("meta"):
        reference = ReRead(config).state_dict()
    if set(state) != set(reference):
        raise ValueError(
            "ReRead checkpoint tensor names differ"
        )
    for name, expected in reference.items():
        value = state[name]
        if (
            value.device.type == "meta"
            or value.is_complex()
            or value.shape != expected.shape
        ):
            raise ValueError(f"Invalid endpoint weight tensor {name}")
        if value.is_floating_point():
            if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                raise ValueError(
                    f"Endpoint trainable weights must be finite FP32: {name}"
                )
        elif value.dtype != expected.dtype:
            raise ValueError(f"Unexpected endpoint buffer dtype: {name}")


def validate_checkpoint(
    payload, *, base_path=None, stats_path=None, cache_manifest_path=None
):
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(
            "Incompatible ReRead checkpoint schema"
        )
    required = {
        "schema",
        "config",
        "model",
        "step",
        "epoch",
        "provenance",
        "execution",
        "sampling",
        "frozen_backbone_gradients",
    }
    if set(payload) != required:
        raise ValueError(
            "Incomplete or unexpected standalone endpoint checkpoint fields"
        )
    from reread.config import ReReadConfig

    config = ReReadConfig.from_dict(payload["config"])
    _integer(payload["step"], "step", 1)
    _integer(payload["epoch"], "epoch")
    if payload["frozen_backbone_gradients"] is not False:
        raise ValueError("Frozen backbone gradients must be verified false")
    execution = payload["execution"]
    if not _matches_execution(execution, config):
        raise ValueError("ReRead checkpoint execution contract differs")
    provenance = payload["provenance"]
    if not isinstance(provenance, Mapping) or not isinstance(
        payload["sampling"], Mapping
    ):
        raise ValueError("Checkpoint provenance and sampling must be mappings")
    for key in _IDENTITIES:
        if not isinstance(provenance.get(key), str) or not re.fullmatch(
            r"[0-9a-f]{64}", provenance[key]
        ):
            raise ValueError(f"Missing or invalid {key}")
    for path, key in (
        (base_path, "base_sha256"),
        (stats_path, "stats_sha256"),
        (cache_manifest_path, "observation_cache_manifest_sha256"),
    ):
        if path is not None and sha256_file(path) != provenance[key]:
            raise ValueError(f"Checkpoint {key} differs from requested source")
    _validate_weights(payload["model"], config)
    return config


def load_checkpoint(path, *, base_path=None, stats_path=None, cache_manifest_path=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_checkpoint(
        payload,
        base_path=base_path,
        stats_path=stats_path,
        cache_manifest_path=cache_manifest_path,
    )
    for key in ("config", "execution", "sampling"):
        payload[key] = _with_current_feature_schema(payload[key])
    return payload


def save_checkpoint(
    path,
    model,
    *,
    step,
    epoch=0,
    provenance,
    sampling,
    seed=0,
    frozen_backbone_gradients=False,
):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    payload = dict(
        schema=CHECKPOINT_SCHEMA,
        config=model.config.to_dict(),
        model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        step=step,
        epoch=epoch,
        provenance=dict(provenance),
        sampling=_with_current_feature_schema(dict(sampling)),
        execution=execution_contract(model.config, seed),
        frozen_backbone_gradients=frozen_backbone_gradients,
    )
    validate_checkpoint(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + f".pending.{os.getpid()}")
    with pending.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(pending, path)
    pending.unlink()
    return payload
