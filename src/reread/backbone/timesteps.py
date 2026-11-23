import math
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass
class _TimeCache:
    signature: object = None
    shape: object = None
    entries: list = field(default_factory=list)


def _parameter_signature(*modules):
    result = []
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad:
                raise ValueError(
                    "Timestep reuse requires frozen time-projection parameters"
                )
            if parameter.is_inference():
                raise ValueError("Timestep reuse needs versioned frozen parameters")
            result.append(
                (
                    id(parameter),
                    parameter.data_ptr(),
                    parameter._version,
                    tuple(parameter.shape),
                    parameter.device,
                    parameter.dtype,
                )
            )
    return tuple(result)


def _require_same(value, expected):
    if (
        value.shape != expected.shape
        or value.device != expected.device
        or value.dtype != expected.dtype
    ):
        raise RuntimeError("Fixed action-denoising timestep shape/device/dtype changed")
    same = torch.eq(value, expected).all()
    if value.device.type == "cuda":
        torch._assert_async(same, "Fixed action-denoising timestep sequence changed")
    elif not bool(same):
        raise RuntimeError("Fixed action-denoising timestep sequence changed")


@contextmanager
def cached_action_timesteps(
    backbone, enabled=True, *, inference_steps=20, sigma_shift=5.0
):
    stats = {"enabled": bool(enabled), "hits": 0, "misses": 0, "entries": 0}
    if not enabled:
        yield stats
        return
    if (
        isinstance(inference_steps, bool)
        or not isinstance(inference_steps, int)
        or inference_steps < 1
        or not math.isfinite(sigma_shift)
        or sigma_shift <= 0
    ):
        raise ValueError(
            "A positive fixed inference-step count and sigma shift are required"
        )
    expert = backbone.action_expert
    if backbone.training or expert.training:
        raise ValueError(
            "Timestep reuse is limited to the evaluation-mode frozen backbone"
        )
    embedding, projection = expert.time_embedding, expert.time_projection
    if embedding.training or projection.training:
        raise ValueError(
            "Timestep reuse requires evaluation-mode time-projection modules"
        )
    if getattr(backbone, "_reread_time_cache_active", False):
        raise RuntimeError("Concurrent or nested action-timestep reuse is unsupported")
    original_embedding, original_projection = embedding.forward, projection.forward
    signatures = (
        _parameter_signature(embedding, projection),
        getattr(original_embedding, "__func__", original_embedding),
        getattr(original_projection, "__func__", original_projection),
        inference_steps,
        float(sigma_shift),
    )
    cache = getattr(backbone, "_reread_action_time_cache", None)
    if not isinstance(cache, _TimeCache):
        cache = _TimeCache()
        backbone._reread_action_time_cache = cache
    if cache.signature != signatures:
        cache.signature, cache.shape, cache.entries = signatures, None, []
    index, awaiting_projection = 0, None
    restored = [
        (module, "forward" in module.__dict__, module.__dict__.get("forward"))
        for module in (embedding, projection)
    ]

    def embedding_forward(value):
        nonlocal awaiting_projection
        if awaiting_projection is not None or index >= inference_steps:
            raise RuntimeError("Action timestep/projection call order changed")
        if value.ndim != 2 or value.requires_grad:
            raise ValueError("Expected detached fixed timestep embeddings [B,F]")
        shape = (tuple(value.shape), value.device, value.dtype)
        if index == 0 and cache.shape != shape:
            cache.shape, cache.entries = shape, []
        if shape != cache.shape:
            raise RuntimeError("Batch shape changed within one action extraction")
        if index < len(cache.entries):
            entry = cache.entries[index]
            _require_same(value, entry["input"])
            stats["hits"] += 1
        else:
            entry = {
                "input": value.detach().clone(),
                "embedding": original_embedding(value).detach().clone(),
                "projection": None,
            }
            cache.entries.append(entry)
            stats["misses"] += 1
        awaiting_projection = entry
        return entry["embedding"]

    def projection_forward(value):
        nonlocal index, awaiting_projection
        if awaiting_projection is None or value is not awaiting_projection["embedding"]:
            raise RuntimeError(
                "Action time projection did not consume its matching timestep embedding"
            )
        entry = awaiting_projection
        if entry["projection"] is None:
            entry["projection"] = original_projection(value).detach().clone()
        awaiting_projection = None
        index += 1
        return entry["projection"]

    backbone._reread_time_cache_active = True
    try:
        embedding.forward, projection.forward = embedding_forward, projection_forward
        try:
            yield stats
            if index != inference_steps or awaiting_projection is not None:
                raise RuntimeError(
                    "A complete fixed action-denoising sequence was not executed"
                )
            stats["entries"] = len(cache.entries)
        except BaseException:
            cache.shape, cache.entries = None, []
            raise
    finally:
        for module, had_override, original in reversed(restored):
            if had_override:
                module.forward = original
            else:
                delattr(module, "forward")
        backbone._reread_time_cache_active = False
