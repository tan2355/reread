import logging
import threading
import time
from contextlib import contextmanager

import torch

from reread.backbone.cuda_graph import GraphedSpatialFastWAM
from reread.backbone.patch_projection import current_frame_addmm_projection

logger = logging.getLogger(__name__)


def graph_mode(value):
    if type(value) is not str or value not in ("native", "auto"):
        raise ValueError("endpoint_frozen_graph must be native or auto")
    return value


def _graph_supported(base):
    return (
        torch.device(base.device).type == "cuda"
        and torch.cuda.is_available()
        and hasattr(torch.cuda, "CUDAGraph")
    )


def _device(base):
    device = torch.device(base.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _same(reference, actual):
    a, kv = reference
    b, other = actual
    if (
        type(kv) is not dict
        or type(other) is not dict
        or kv.keys() != other.keys()
        or kv.keys() != {"k", "v"}
    ):
        return False
    for x, y in ((a, b), (kv["k"], other["k"]), (kv["v"], other["v"])):
        if (
            x.shape != y.shape
            or x.dtype != y.dtype
            or x.device != y.device
            or not bool(torch.isfinite(x).all())
            or not bool(torch.isfinite(y).all())
            or not torch.equal(
                x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)
            )
        ):
            return False
    return True


class LiveGraphedFrozen:

    def __init__(self, frozen, *, mode="auto", audit_decisions=2):
        self.frozen = frozen
        self.mode = graph_mode(mode)
        if type(audit_decisions) is not int or audit_decisions < 2:
            raise ValueError("At least two live native/graph decisions must be audited")
        self.audit_decisions = audit_decisions
        self._lock = threading.Lock()
        self._patch = self._patch_stats = self._graph = None
        self._shape = None
        self._closed = False
        self._fallback = None
        self._audited = self._calls = self._replays = 0
        self._prepare_s = 0.0

    @property
    def backbone(self):
        return self.frozen.backbone

    @property
    def config(self):
        return self.frozen.config

    def stats(self):
        return dict(
            mode=self.mode,
            effective_patch_projection="native_addmm",
            graph_prepared=self._graph is not None and self._fallback is None,
            graph_enabled=self._graph is not None
            and self._fallback is None
            and self._audited >= self.audit_decisions,
            audited_live_decisions=self._audited,
            required_audits=self.audit_decisions,
            full_actions_kv_byte_equal=self._audited >= self.audit_decisions
            and self._fallback is None,
            calibration_scope="first_actual_live_decisions_not_all_possible_inputs",
            replay_decisions=self._replays,
            calls=self._calls,
            prepare_host_seconds=self._prepare_s,
            fallback_reason=self._fallback,
            graph={} if self._graph is None else self._graph.stats(),
        )

    def _ensure_patch(self):
        if self._patch is None:
            scope = current_frame_addmm_projection(self.backbone, validate=True)
            stats = scope.__enter__()
            self._patch, self._patch_stats = scope, stats

    @contextmanager
    def _encoder_outputs(self):
        base, captured, restored = self.backbone, {}, []
        try:
            for method, key in (
                ("_encode_input_image_latents_tensor", "rgb_latents"),
                ("encode_prompt", "text"),
            ):
                original = getattr(base, method)
                restored.append(
                    (method, method in base.__dict__, base.__dict__.get(method))
                )

                def observe(*args, _original=original, _key=key, **kwargs):
                    if _key in captured:
                        raise RuntimeError(
                            "Online graph calibration encoded current input twice"
                        )
                    result = _original(*args, **kwargs)
                    captured[_key] = result
                    return result

                setattr(base, method, observe)
            yield captured
        finally:
            for method, had, previous in reversed(restored):
                if had:
                    setattr(base, method, previous)
                else:
                    delattr(base, method)

    def _pack(self, observation, rgb, text):
        base = self.backbone
        context, mask = text
        proprio = observation["proprio"]
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        proprio = proprio.to(device=base.device, dtype=base.torch_dtype)

        return dict(
            rgb_latents=rgb,
            proprio=proprio,
            context=context,
            context_mask=mask,
            seed=observation["seed"],
        )

    def _compatible(self, values):
        base = self.backbone
        tensors = {k: v for k, v in values.items() if k != "seed"}
        if any(
            not isinstance(v, torch.Tensor) or not v.is_contiguous()
            for v in tensors.values()
        ):
            return False
        patch = base.video_expert.patch_size
        spatial = tuple(self.config.video_grid[i] * patch[i + 1] for i in range(2))
        return (
            tensors["rgb_latents"].shape == (1, base.video_expert.in_dim, 1, *spatial)
            and tensors["proprio"].shape == (1, base.proprio_dim)
            and tensors["context"].ndim == 3
            and tensors["context"].shape[0] == 1
            and tensors["context_mask"].shape == tensors["context"].shape[:2]
            and tensors["context_mask"].dtype == torch.bool
            and all(v.device == _device(base) for v in tensors.values())
            and all(
                v.dtype == base.torch_dtype
                for k, v in tensors.items()
                if k != "context_mask"
            )
        )

    @staticmethod
    def _layout(values):
        return tuple(
            (key, tuple(v.shape), tuple(v.stride()), v.dtype, v.device)
            for key, v in values.items()
            if key != "seed"
        )

    def _disable(self, reason):
        if self._graph is not None:
            self._graph.clear()
        self._fallback = reason
        logger.warning("Endpoint frozen graph permanently disabled: %s", reason)

    def _validate_request(self, observation):
        c = self.config
        fixed = dict(
            seed=0,
            rand_device="cpu",
            text_cfg_scale=1.0,
            negative_prompt="",
            tiled=False,
        )
        fixed.update(
            action_horizon=c.horizon,
            num_inference_steps=c.inference_steps,
            sigma_shift=c.sigma_shift,
            compile_action_infer=False,
        )
        allowed = set(fixed) | {"prompt", "input_image", "proprio"}
        if set(observation) - allowed or any(
            k in observation and observation[k] != v for k, v in fixed.items()
        ):
            raise ValueError(
                "Online graph requires the pinned native seed0 D20/CFG1 single-observation request"
            )
        if (
            "seed" not in observation
            or not isinstance(observation.get("prompt"), str)
            or not isinstance(observation.get("input_image"), torch.Tensor)
            or not isinstance(observation.get("proprio"), torch.Tensor)
            or c.inference_steps != 20
            or c.horizon != 32
        ):
            raise ValueError("Incomplete/unsupported online graph request")

    @torch.no_grad()
    def extract(self, **observation):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("One online graph cannot serve concurrent extraction")
        try:
            if self._closed:
                raise RuntimeError("Online graph extractor was closed")
            self._validate_request(observation)
            self._ensure_patch()
            self._calls += 1
            if self.mode == "native" or self._fallback is not None:
                return self.frozen.extract(**observation)
            if not _graph_supported(self.backbone):
                self._disable("CUDA graph unavailable on this runtime")
                return self.frozen.extract(**observation)

            if self._audited < self.audit_decisions:
                with self._encoder_outputs() as captured:
                    reference = self.frozen.extract(**observation)
                if set(captured) != {"rgb_latents", "text"}:
                    raise RuntimeError("Original live encoder call graph changed")
                values = self._pack(
                    observation, captured["rgb_latents"], captured["text"]
                )
                if not self._compatible(values):
                    self._disable("Original live encoder output layout is unsupported")
                    return reference
                if self._graph is None:
                    self._shape = self._layout(values)
                    self._graph = GraphedSpatialFastWAM(self.frozen)
                    started = time.perf_counter()
                    self._graph.prepare_graph(**values)
                    self._prepare_s += time.perf_counter() - started
                elif self._layout(values) != self._shape:
                    self._disable("Live input layout changed during calibration")
                    return reference
                candidate = self._graph.extract_batch(**values)
                if not _same(reference, candidate):
                    self._disable(
                        "Full live actions/K/V failed byte-equality calibration"
                    )
                    return reference
                self._audited += 1
                logger.info(
                    "Endpoint frozen graph live audit %d/%d: full actions/K/V byte-equal",
                    self._audited,
                    self.audit_decisions,
                )
                return reference
            base = self.backbone
            image = observation["input_image"].to(
                device=base.device, dtype=base.torch_dtype
            )
            rgb = base._encode_input_image_latents_tensor(
                input_image=image, tiled=False
            )
            text = base.encode_prompt(observation["prompt"])
            values = self._pack(observation, rgb, text)
            if not self._compatible(values) or self._layout(values) != self._shape:
                raise RuntimeError(
                    "CUDA graph live encoder layout changed after calibration"
                )
            result = self._graph.extract_batch(**values)
            self._replays += 1
            return result
        finally:
            self._lock.release()

    def close(self):
        with self._lock:
            if self._graph is not None:
                self._graph.clear()
            if self._patch is not None:
                self._patch.__exit__(None, None, None)
                self._patch = None
            self._closed = True
