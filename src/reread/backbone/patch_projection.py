import threading
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


def _geometry(module):
    return (
        module.in_channels,
        module.out_channels,
        tuple(module.kernel_size),
        tuple(module.stride),
        tuple(module.padding),
        tuple(module.dilation),
        module.groups,
        module.padding_mode,
    )


def _parameters(module):
    result = []
    for parameter in (module.weight, module.bias):
        if parameter is None:
            result.append(None)
            continue
        if parameter.requires_grad or parameter.is_inference():
            raise ValueError(
                "Current-frame Conv2d requires versioned frozen patch parameters"
            )
        result.append(
            (
                id(parameter),
                parameter.data_ptr(),
                parameter._version,
                tuple(parameter.shape),
                parameter.dtype,
                parameter.device,
            )
        )
    return tuple(result)


def _precision_flags():
    return (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.enabled,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.mkldnn.enabled,
        torch.get_float32_matmul_precision(),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_autocast_enabled("cuda"),
        torch.get_autocast_dtype("cuda"),
        torch.is_autocast_enabled("cpu"),
        torch.get_autocast_dtype("cpu"),
    )


def _input_metadata(value):
    return (tuple(value.shape), tuple(value.stride()), value.dtype, value.device)


def _input_record(value):
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


def _is_capturing(value):
    return value.device.type == "cuda" and torch.cuda.is_current_stream_capturing()


class AddMMProjectionStats(dict):

    def __init__(self, enabled, automatic):
        super().__init__(
            enabled=bool(enabled),
            calls=0,
            automatic_validation=bool(automatic),
            native_validation_calls=0,
            validation_cache_hits=0,
            validated_shapes=[],
        )
        self._validate = None
        self._active = False

    @torch.no_grad()
    def validate_current(self, value):
        if not self["enabled"]:
            return {"validated": False, "disabled": True, "cache_hit": False}
        if not self._active or self._validate is None:
            raise RuntimeError("AddMM patch validation requires its active context")
        record, _output = self._validate(value)
        return record


@contextmanager
def _current_frame_patch_override(
    backbone,
    enabled,
    project,
    *,
    proof_api=False,
    validate=False,
    validation_cache=None,
):
    stats = (
        AddMMProjectionStats(enabled, validate)
        if proof_api
        else {"enabled": bool(enabled), "calls": 0}
    )
    if not enabled:
        yield stats
        return
    if (
        proof_api
        and validation_cache is not None
        and not isinstance(validation_cache, set)
    ):
        raise TypeError("AddMM validation_cache must be a metadata-only set")
    expert = backbone.video_expert
    module = expert.patch_embedding
    if not isinstance(module, nn.Conv3d):
        raise ValueError(
            "Current-frame Conv2d requires a Conv3d-compatible RGB patch module"
        )
    lock = module.__dict__.setdefault("_reread_patch_conv2d_lock", threading.Lock())
    if not lock.acquire(blocking=False):
        raise RuntimeError(
            "Concurrent or nested current-frame Conv2d override is unsupported"
        )
    had_override = "_conv_forward" in module.__dict__
    previous = module.__dict__.get("_conv_forward")
    installed = False
    try:
        original = module._conv_forward
        if getattr(original, "__func__", None) is not nn.Conv3d._conv_forward:
            raise ValueError("A custom patch _conv_forward cannot be bypassed")
        if expert.training or module.training:
            raise ValueError(
                "Current-frame Conv2d is restricted to evaluation-mode patch modules"
            )
        geometry = _geometry(module)
        if (
            module.groups != 1
            or module.padding != (0, 0, 0)
            or module.dilation != (1, 1, 1)
            or module.stride != module.kernel_size
            or module.kernel_size[0] != 1
            or module.padding_mode != "zeros"
        ):
            raise ValueError(
                "Current-frame Conv2d requires Kt=1, unpadded non-overlapping patches"
            )
        signature = _parameters(module)
        if not module.weight.is_contiguous():
            raise ValueError(
                "Current-frame Conv2d preserves only native contiguous patch weights"
            )
        precision = _precision_flags() if proof_api else None
        proofs = validation_cache if validation_cache is not None else set()
        proof_prefix = (
            (
                geometry,
                signature,
                precision,
                getattr(original, "__func__", original),
                project.__code__,
            )
            if proof_api
            else None
        )

        def check_input(value, weight, bias):
            if torch.is_grad_enabled():
                raise RuntimeError(
                    "Current-frame Conv2d requires no_grad frozen inference"
                )
            if expert.training or module.training or _geometry(module) != geometry:
                raise RuntimeError(
                    "Current-frame Conv2d patch mode or geometry changed"
                )
            if (
                _parameters(module) != signature
                or weight is not module.weight
                or bias is not module.bias
            ):
                raise RuntimeError("Current-frame Conv2d patch parameters changed")
            if proof_api and _precision_flags() != precision:
                raise RuntimeError(
                    "AddMM patch precision/runtime flags changed during its context"
                )
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 5
                or value.shape[2] != 1
                or value.shape[1] != module.in_channels
                or value.requires_grad
                or not value.is_contiguous()
                or not value.is_floating_point()
            ):
                raise ValueError(
                    "Current-frame Conv2d expects detached native N,C,1,H,W input"
                )
            if value.device != weight.device or value.dtype != weight.dtype:
                raise ValueError(
                    "Current-frame Conv2d input must match patch weight device/dtype"
                )
            kh, kw = module.kernel_size[1:]
            if value.shape[3] % kh or value.shape[4] % kw:
                raise ValueError(
                    "Current-frame Conv2d input must be divisible by patch size"
                )
            return kh, kw

        def compute(value, weight, bias, kh, kw):
            result = project(value, weight, bias, kh, kw)
            stats["calls"] += 1
            return result

        def validate_current(value):
            kh, kw = check_input(value, module.weight, module.bias)
            if module.__dict__.get("_conv_forward") is not conv_forward:
                raise RuntimeError(
                    "AddMM validation no longer owns the current patch override"
                )
            key = (proof_prefix, _input_metadata(value))
            record = _input_record(value)
            if key in proofs:
                stats["validation_cache_hits"] += 1
                return dict(
                    record,
                    validated=True,
                    bitwise_equal=True,
                    finite=True,
                    cache_hit=True,
                ), None
            if _is_capturing(value):
                raise RuntimeError(
                    "AddMM patch shape has no pre-capture numerical validation; "
                    "validate it during serial warmup before CUDA capture"
                )
            native = original(value, module.weight, module.bias)
            optimized = compute(value, module.weight, module.bias, kh, kw)
            stats["native_validation_calls"] += 1
            if not bool(torch.isfinite(native).all()) or not bool(
                torch.isfinite(optimized).all()
            ):
                raise RuntimeError(
                    "AddMM patch native/optimized numerical validation produced nonfinite output"
                )
            if not torch.equal(native, optimized):
                raise RuntimeError(
                    "AddMM patch native/optimized numerical validation is not bitwise equal"
                )
            proofs.add(key)
            checked = dict(
                record, validated=True, bitwise_equal=True, finite=True, cache_hit=False
            )
            stats["validated_shapes"].append(checked)
            return checked, optimized

        def conv_forward(value, weight, bias):
            kh, kw = check_input(value, weight, bias)
            if proof_api and validate:
                _record, output = validate_current(value)
                if output is not None:
                    return output
            return compute(value, weight, bias, kh, kw)

        module._conv_forward = conv_forward
        installed = True
        if proof_api:
            stats._validate, stats._active = validate_current, True
        yield stats
    finally:
        if proof_api:
            stats._validate, stats._active = None, False
        if installed:
            if had_override:
                module._conv_forward = previous
            else:
                delattr(module, "_conv_forward")
        lock.release()


@contextmanager
def current_frame_conv2d(backbone, enabled=True):

    def project(value, weight, bias, kh, kw):
        return F.conv2d(
            value.squeeze(2),
            weight.squeeze(2),
            bias,
            stride=(kh, kw),
            padding=0,
            dilation=1,
            groups=1,
        ).unsqueeze(2)

    with _current_frame_patch_override(backbone, enabled, project) as stats:
        yield stats


@contextmanager
def current_frame_linear_separate_bias(backbone, enabled=True):

    def project(value, weight, bias, kh, kw):
        b, c, _t, h, w = value.shape
        patches = value.reshape(b, c, 1, 1, h // kh, kh, w // kw, kw)
        patches = patches.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(
            b, 1, h // kh, w // kw, -1
        )
        result = F.linear(patches, weight.flatten(1), bias=None).to(dtype=value.dtype)
        if bias is not None:
            result = result + bias
        return result.permute(0, 4, 1, 2, 3)

    with _current_frame_patch_override(backbone, enabled, project) as stats:
        yield stats


def _spatial_columns(value, kh, kw):
    b, c, _t, h, w = value.shape

    return (
        value.reshape(b, c, 1, 1, h // kh, kh, w // kw, kw)
        .permute(0, 1, 3, 5, 7, 2, 4, 6)
        .reshape(b, c * kh * kw, (h // kh) * (w // kw))
        .contiguous()
    )


@contextmanager
def current_frame_addmm_projection(
    backbone, enabled=True, *, validate=False, validation_cache=None
):

    def project(value, weight, bias, kh, kw):
        columns = _spatial_columns(value, kh, kw)
        matrix = weight.flatten(1)
        positions = columns.shape[-1]
        initial = bias[:, None].expand(-1, positions) if bias is not None else None
        outputs = [
            torch.addmm(initial, matrix, column, beta=1, alpha=1)
            if initial is not None
            else torch.mm(matrix, column)
            for column in columns.unbind(0)
        ]
        return torch.stack(outputs, dim=0).reshape(
            value.shape[0],
            weight.shape[0],
            1,
            value.shape[3] // kh,
            value.shape[4] // kw,
        )

    with _current_frame_patch_override(
        backbone,
        enabled,
        project,
        proof_api=True,
        validate=validate,
        validation_cache=validation_cache,
    ) as stats:
        yield stats
