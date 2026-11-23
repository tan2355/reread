import json

import pytest
import torch
from test_engine import config

from reread.checkpoint import (
    CHECKPOINT_SCHEMA,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint,
)
from reread.model import ReRead


def make(tmp_path):
    model = ReRead(config())
    provenance = dict(
        base_sha256="a" * 64,
        stats_sha256="b" * 64,
        observation_cache_manifest_sha256="c" * 64,
    )
    path = tmp_path / "endpoint_step_40000.pt"
    payload = save_checkpoint(
        path,
        model,
        step=40000,
        provenance=provenance,
        sampling=dict(image_predictor=False),
    )
    return path, model, payload


def test_strict_endpoint_contract_weights_and_roundtrip_no_overwrite(tmp_path):
    path, model, payload = make(tmp_path)
    loaded = load_checkpoint(path)
    assert loaded["schema"] == CHECKPOINT_SCHEMA
    contract = loaded["execution"]
    assert contract["final_action_residual"] and not contract["image_prediction"]
    assert contract["corrector_calls_per_decision"] == 1
    assert contract["base_calls_per_decision"] == model.config.inference_steps
    assert not contract["perturb_kv"] and not contract["prefix_is_model_input"]
    assert not contract["prefix_is_kv"] and not contract["perturbation_inference"]
    assert "predictor" not in "".join(loaded["model"])
    validate_checkpoint(
        json.loads(json.dumps({k: v for k, v in payload.items() if k != "model"}))
        | {"model": loaded["model"]}
    )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded["model"][name], value)
    with pytest.raises(FileExistsError):
        save_checkpoint(
            path, model, step=40000, provenance=payload["provenance"], sampling={}
        )


def test_earlier_feature_name_loads_the_same_three_record_model(tmp_path):
    _, model, payload = make(tmp_path)
    canonical = "reread-pre-rope-camera-v1"
    assert payload["config"]["feature_schema"] == canonical
    assert payload["execution"]["feature_schema"] == canonical
    sections = ("config", "execution", "sampling")
    for key in sections:
        payload[key]["feature_schema"] = "newm-memory4-pre-rope-camera-v1"
    path = tmp_path / "earlier.pt"
    torch.save(payload, path)
    original = path.read_bytes()
    loaded = load_checkpoint(path)
    assert all(loaded[key]["feature_schema"] == canonical for key in sections)
    assert validate_checkpoint(loaded) == model.config
    assert path.read_bytes() == original
    saved = save_checkpoint(
        tmp_path / "current.pt", model, step=1,
        provenance=payload["provenance"], sampling=payload["sampling"],
    )
    assert saved["sampling"]["feature_schema"] == canonical
    payload["config"]["memory_frames"] = 4
    with pytest.raises(ValueError, match="Memory3"):
        validate_checkpoint(payload)


def test_incompatible_checkpoint_rejected(tmp_path):
    _, _, payload = make(tmp_path)
    payload["schema"] = "not-a-reread-checkpoint"
    with pytest.raises(ValueError, match="Incompatible"):
        validate_checkpoint(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("perturb_kv", True),
        ("final_action_residual", False),
        ("prefix_is_model_input", True),
        ("prefix_is_kv", True),
        ("perturbation_inference", True),
        ("velocity_weight", 0.1),
        ("extra_video_forwards", 1),
        ("corrector_calls_per_decision", 20),
        ("candidate", "pooled_action_tokens"),
        ("feature_schema", "unknown-feature-layout"),
        ("output_clipping", True),
    ],
)
def test_endpoint_execution_contract_tamper_rejected(tmp_path, field, value):
    _, _, payload = make(tmp_path)
    payload["execution"][field] = value
    with pytest.raises(ValueError, match="contract"):
        validate_checkpoint(payload)


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "extra",
        "shape",
        "nan",
        "bf16",
        "bad_sha",
        "frozen_gradients",
        "bool_step",
        "old_config",
    ],
)
def test_invalid_or_partial_checkpoint_rejected(tmp_path, kind):
    _, _, payload = make(tmp_path)
    key = next(k for k, v in payload["model"].items() if v.is_floating_point())
    if kind == "missing":
        payload["model"].pop(key)
    if kind == "extra":
        payload["model"]["predictor.weight"] = torch.zeros(1)
    if kind == "shape":
        payload["model"][key] = torch.zeros(1)
    if kind == "nan":
        payload["model"][key].flatten()[0] = float("nan")
    if kind == "bf16":
        payload["model"][key] = payload["model"][key].bfloat16()
    if kind == "bad_sha":
        payload["provenance"]["base_sha256"] = "bad"
    if kind == "frozen_gradients":
        payload["frozen_backbone_gradients"] = 0
    if kind == "bool_step":
        payload["step"] = True
    if kind == "old_config":
        payload["config"]["plan_perturb_probability"] = 0.2
    with pytest.raises(ValueError):
        validate_checkpoint(payload)
