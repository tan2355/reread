import hashlib
import os
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "experiments/robotwin"))
from reread_policy import color_contract as color
from reread_policy import deploy_policy as deploy
from reread_policy import file_certificate as certified

from reread.checkpoint import execution_contract
from reread.config import ReReadConfig

D20_SOURCE = "a" * 64
RUNTIME = dict(
    vae_encoding="native",
    patch_projection="native_addmm",
    fused_visible_attention=False,
    timestep_cache=False,
    rgb_cache_mib=0,
    text_cache_mib=0,
)


def payload():
    config = ReReadConfig(inference_steps=20)
    return dict(
        schema="newm-endpoint-memory3-weights-v1",
        step=20000,
        config=asdict(config),
        model={},
        execution=execution_contract(config),
        provenance=dict(
            source_manifest_sha256=D20_SOURCE,
            observation_cache_manifest_sha256=("b" * 64),
        ),
        sampling=dict(
            observation_color_contract="native_robotwin_rgb_v1",
            kernel="native_addmm",
            correction_start_decision=0,
            prefix_warmup_decisions=0,
        ),
    )


def test_rgb_contract_uses_checkpoint_metadata_and_rejects_channel_swaps():
    value = payload()
    assert color.validate_checkpoint_rgb_mode("native_rgb", value) == "native_rgb"
    changed = deepcopy(value)
    changed["sampling"]["observation_color_contract"] = "unknown"
    with pytest.raises(ValueError):
        color.validate_checkpoint_rgb_mode("native_rgb", changed)
    with pytest.raises(ValueError):
        color.validate_checkpoint_rgb_mode("swap_rb", value)


def test_d20_deployment_preserves_explicit_bf16_native_runtime_and_execution(
    monkeypatch,
):
    value = payload()
    args = dict(
        value["execution"],
        endpoint_rgb_mode="native_rgb",
        endpoint_runtime=RUNTIME,
        endpoint_corrector_autocast=True,
    )
    monkeypatch.setattr(
        deploy, "validate_checkpoint", lambda p: ReReadConfig.from_dict(p["config"])
    )
    assert deploy.validate_rollout_args(args, value).inference_steps == 20
    assert deploy.deployment_options(args, value) == (RUNTIME, True)
    for key, wrong in (
        ("num_inference_steps", 10),
        ("replan_steps", 10),
        ("endpoint_rgb_mode", "swap_rb"),
        ("endpoint_corrector_autocast", False),
    ):
        with pytest.raises(ValueError):
            deploy.validate_rollout_args(dict(args, **{key: wrong}), value)


class FakeCorrector:
    def __init__(self, config):
        self.config = config

    def to(self, device):
        assert device == "cpu"
        return self

    def load_state_dict(self, state, strict):
        assert strict is True

    def eval(self):
        return self


class FakePromptCache:
    def __init__(self, *args, **kwargs):
        self.resets = 0

    def encode(self, prompt):
        raise AssertionError("The adapter must not encode prompts itself")

    def reset_rollout_stats(self):
        self.resets += 1


class FakeTemporal:
    def __init__(
        self, frozen, corrector, *, runtime, bf16_corrector, memory_bank=None
    ):
        assert memory_bank is None
        assert runtime == RUNTIME and bf16_corrector is True
        self.calls = []
        self.resets = 0

    def act(self, **kwargs):
        self.calls.append(kwargs)

        return dict(action=np.arange(32 * 14, dtype=np.float32).reshape(32, 14))

    def reset(self):
        self.resets += 1


def build_adapter(monkeypatch, supports_video_frames):
    config = ReReadConfig(inference_steps=20)
    monkeypatch.setattr(deploy, "validate_checkpoint", lambda _: config)
    monkeypatch.setattr(deploy, "ReRead", FakeCorrector)
    monkeypatch.setattr(deploy, "FrozenSpatialFastWAM", lambda *a, **kw: object())
    monkeypatch.setattr(deploy, "FrozenPromptCache", FakePromptCache)
    monkeypatch.setattr(deploy, "prepare_current_vae", lambda *a, **kw: object())
    monkeypatch.setattr(deploy, "TemporalMemoryPolicy", FakeTemporal)

    def with_frames(*, input_image, num_video_frames=None):
        pass

    def without_frames(*, input_image):
        pass

    model = SimpleNamespace(
        device="cpu",
        vae=SimpleNamespace(model=object()),
        infer_action=with_frames if supports_video_frames else without_frames,
    )
    image = object()
    normalized = torch.arange(14, dtype=torch.float32).unsqueeze(0)
    base = SimpleNamespace(
        model=model,
        timing_enabled=False,
        episode_count=7,
        step_count=0,
        negative_prompt="",
        text_cfg_scale=1.0,
        seed=0,
        rand_device="cpu",
        tiled=False,
        _num_video_frames=9,
        replan_steps=20,
        resets=0,
    )
    base._build_robotwin_image_tensor = lambda obs: image
    base._normalize_state = lambda state: normalized
    base._denormalize_action = lambda actions: actions[None]

    def reset_base():
        base.resets += 1

    base.reset = reset_base
    real_signature = deploy.inspect.signature
    inspected = []

    def signature(call):
        inspected.append(call)
        return real_signature(call)

    monkeypatch.setattr(deploy.inspect, "signature", signature)
    adapter = deploy.ReReadPolicy(
        base,
        payload(),
        config,
        endpoint_runtime=RUNTIME,
        endpoint_corrector_autocast=True,
    )
    return adapter, base, inspected, image, normalized


@pytest.mark.parametrize("supports_video_frames", [False, True])
def test_signature_resolved_once_while_each_decision_keeps_exact_inputs(
    monkeypatch, supports_video_frames
):
    adapter, base, inspected, image, normalized = build_adapter(
        monkeypatch, supports_video_frames
    )
    assert inspected == [base.model.infer_action]

    monkeypatch.setattr(
        deploy.inspect,
        "signature",
        lambda _: pytest.fail("Repeated signature inspection in hot path"),
    )
    camera = np.array([[[17, 91, 233]]], dtype=np.uint8)
    observation = dict(
        joint_action=dict(vector=np.arange(14)),
        observation=dict(head_camera=dict(rgb=camera)),
    )
    for decision in range(3):
        base.step_count = decision * 20
        result = adapter._infer_action_chunk(observation, "turn switch")
        np.testing.assert_array_equal(
            result, np.arange(32 * 14, dtype=np.float32).reshape(32, 14)
        )
        actual = adapter.temporal.calls[-1]
        expected = dict(
            episode_id=7,
            decision_index=decision,
            observation_step=decision * 20,
            prompt=deploy.DEFAULT_PROMPT.format(task="turn switch"),
            input_image=image,
            proprio=normalized,
            negative_prompt="",
            text_cfg_scale=1.0,
            seed=0,
            rand_device="cpu",
            tiled=False,
        )
        if supports_video_frames:
            expected["num_video_frames"] = 9
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            if key in ("input_image", "proprio"):
                assert actual[key] is value
            else:
                assert actual[key] == value
        assert observation["observation"]["head_camera"]["rgb"] is camera
        assert camera.tolist() == [[[17, 91, 233]]]
    assert adapter.decision_index == 3
    adapter.reset()
    assert adapter.decision_index == 0
    assert base.resets == adapter.temporal.resets == adapter.prompt_cache.resets == 1
    adapter._infer_action_chunk(observation, "turn switch")
    assert adapter.temporal.calls[-1]["decision_index"] == 0
    assert len(inspected) == 1


def certificate_fixture(tmp_path):
    base, stats = tmp_path / "base.pt", tmp_path / "stats.json"
    base.write_bytes(b"frozen base sentinel")
    stats.write_bytes(b'{"mean": 0}')
    provenance = dict(
        base_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
        stats_sha256=hashlib.sha256(stats.read_bytes()).hexdigest(),
    )
    signatures = dict(
        base=certified.file_signature(base), stats=certified.file_signature(stats)
    )
    certificate = certified.issue_file_certificate(
        signatures, deepcopy(signatures), provenance
    )
    return base, stats, provenance, signatures, certificate


def test_local_certificate_detects_wrong_path_sha_and_modified_metadata(tmp_path):
    base, stats, provenance, signatures, certificate = certificate_fixture(tmp_path)
    kwargs = dict(base_path=base, stats_path=stats, provenance=provenance)
    certified.validate_file_certificate(certificate, **kwargs)
    other = tmp_path / "same_bytes_different_path.pt"
    other.write_bytes(base.read_bytes())
    with pytest.raises(ValueError, match="different path"):
        certified.validate_file_certificate(
            certificate, **dict(kwargs, base_path=other)
        )
    changed = deepcopy(provenance)
    changed["base_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="provenance"):
        certified.validate_file_certificate(
            certificate, **dict(kwargs, provenance=changed)
        )
    changed = deepcopy(certificate)
    changed["base"]["size"] = True
    with pytest.raises(ValueError, match="signature values"):
        certified.validate_file_certificate(changed, **kwargs)
    base.write_bytes(b"changed base sentinel")

    os.utime(base, ns=(base.stat().st_atime_ns, signatures["base"]["mtime_ns"]))
    with pytest.raises(ValueError, match="changed"):
        certified.validate_file_certificate(certificate, **kwargs)
    after = dict(
        base=certified.file_signature(base), stats=certified.file_signature(stats)
    )
    with pytest.raises(ValueError, match="during coordinator verification"):
        certified.issue_file_certificate(signatures, after, provenance)


@pytest.mark.parametrize(
    "certificate", [None, {}, {"schema": certified.CERTIFICATE_SCHEMA}]
)
def test_present_but_malformed_certificate_is_rejected(tmp_path, certificate):
    base, stats, provenance, _, _ = certificate_fixture(tmp_path)
    with pytest.raises(ValueError, match="certificate"):
        certified.validate_file_certificate(
            certificate, base_path=base, stats_path=stats, provenance=provenance
        )


def mock_get_model(
    monkeypatch, tmp_path, *, certified_files, mutate_during_construction=False
):
    base_file, stats, provenance, _, certificate = certificate_fixture(tmp_path)
    value = payload()
    value["provenance"].update(provenance)
    args = dict(
        corrector_checkpoint=str(tmp_path / "corrector.pt"),
        ckpt_setting=str(base_file),
        dataset_stats_path=str(stats),
        seed=0,
        endpoint_runtime=RUNTIME,
        endpoint_corrector_autocast=True,
        endpoint_rgb_mode="native_rgb",
        replan_steps=20,
    )
    if certified_files:
        args["_endpoint_preverified_files"] = certificate
    loads, constructed, checks = [], [], []

    def load(path, **kwargs):
        loads.append((path, kwargs))
        return value

    monkeypatch.setattr(deploy, "load_checkpoint", load)
    monkeypatch.setattr(
        deploy,
        "validate_rollout_args",
        lambda a, p: ReReadConfig.from_dict(p["config"]),
    )
    validate = certified.validate_file_certificate

    def check(cert, **kwargs):
        checks.append(kwargs)
        return validate(cert, **kwargs)

    monkeypatch.setattr(deploy, "validate_file_certificate", check)

    class AttrDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    omega = SimpleNamespace(
        to_container=lambda value, resolve: dict(value),
        create=lambda value: AttrDict(value),
    )
    cfg = SimpleNamespace(
        model={"_target_": "fastwam.runtime.create_fastwam"},
        data=SimpleNamespace(
            train=SimpleNamespace(
                processor=object(), num_frames=33, action_video_freq_ratio=4
            )
        ),
    )

    def construct(**kwargs):
        constructed.append(kwargs)
        if mutate_during_construction:
            base_file.write_bytes(b"file replaced during resident model construction")
        return object()

    upstream = SimpleNamespace(
        _compose_sim_cfg=lambda *a: cfg,
        _parse_bool=lambda value: bool(value),
        WorldActionRobotWinPolicy=construct,
    )
    monkeypatch.setitem(sys.modules, "omegaconf", SimpleNamespace(OmegaConf=omega))
    monkeypatch.setitem(
        sys.modules, "fastwam_policy", SimpleNamespace(deploy_policy=upstream)
    )
    sentinel = object()
    monkeypatch.setattr(deploy, "ReReadPolicy", lambda *a, **kw: sentinel)
    return args, loads, constructed, checks, sentinel


@pytest.mark.parametrize("certified_files", [False, True])
def test_get_model_uses_certificate_only_for_local_preverified_inputs(
    monkeypatch, tmp_path, certified_files
):
    args, loads, constructed, checks, sentinel = mock_get_model(
        monkeypatch, tmp_path, certified_files=certified_files
    )
    assert deploy.get_model(args) is sentinel
    assert len(loads) == len(constructed) == 1
    if certified_files:
        assert loads[0][1] == {}
        assert len(checks) == 2
    else:
        assert loads[0][1] == dict(
            base_path=args["ckpt_setting"], stats_path=args["dataset_stats_path"]
        )
        assert checks == []
    assert constructed[0]["num_inference_steps"] == 20
    assert constructed[0]["replan_steps"] == 20


def test_get_model_refuses_file_modified_while_loading_resident_model(
    monkeypatch, tmp_path
):
    args, loads, constructed, checks, _ = mock_get_model(
        monkeypatch, tmp_path, certified_files=True, mutate_during_construction=True
    )
    with pytest.raises(ValueError, match="changed"):
        deploy.get_model(args)
    assert len(loads) == len(constructed) == 1 and len(checks) == 2


def test_get_model_does_not_fall_back_to_hashing_after_certificate_tampering(
    monkeypatch, tmp_path
):
    args, loads, constructed, checks, _ = mock_get_model(
        monkeypatch, tmp_path, certified_files=True
    )
    args["_endpoint_preverified_files"]["stats"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="provenance"):
        deploy.get_model(args)
    assert len(loads) == 1 and loads[0][1] == {}
    assert constructed == [] and len(checks) == 1
