import hashlib
import inspect
from contextlib import contextmanager
from functools import lru_cache
from types import FunctionType, MethodType

import torch

_BUILD_HASHES = (
    "e1246fa0303e599d1aeb7dc4135d95c2efc202225863b602f88f97d27edec69e",
    "5ac125a93fa0bfebfcc30ba5e9f38a37ec36b1101266677b97be8a1bd48aa159",
)
_ROPE_SHA256 = "697bd12bada4973c31c2c791dbc21804f4e3fa4a9905195e6ddeaa49d4b86b7f"


def _logical_contract(x, freqs, num_heads):
    if (
        not isinstance(x, torch.Tensor)
        or not isinstance(freqs, torch.Tensor)
        or type(num_heads) is not int
        or num_heads < 1
        or x.ndim != 3
        or x.shape[0] < 1
        or x.shape[1] < 1
        or x.shape[2] < 1
        or x.shape[2] % (2 * num_heads)
        or x.dtype != torch.bfloat16
        or freqs.dtype != torch.complex128
        or freqs.shape != (x.shape[1], 1, x.shape[2] // num_heads // 2)
    ):
        raise ValueError(
            "RoPE requires BF16 [B,S,H*D] and complex128 [S,1,D/2] with even D"
        )
    if (
        x.device != freqs.device
        or not x.is_contiguous()
        or not freqs.is_contiguous()
        or x.is_conj()
        or freqs.is_conj()
        or x.is_neg()
        or freqs.is_neg()
    ):
        raise ValueError(
            "RoPE requires contiguous, non-conjugated tensors on the same device"
        )
    if x.requires_grad or freqs.requires_grad:
        raise ValueError(
            "Diagnostic RoPE supports detached frozen inference tensors only"
        )


def rope_apply_reference(x, freqs, num_heads):
    _logical_contract(x, freqs, num_heads)
    pairs = x.double().reshape(*x.shape[:2], num_heads, -1, 2)
    frequency = torch.view_as_real(freqs)
    real = pairs[..., 0] * frequency[..., 0] - pairs[..., 1] * frequency[..., 1]
    imag = pairs[..., 0] * frequency[..., 1] + pairs[..., 1] * frequency[..., 0]
    return torch.stack((real, imag), dim=-1).flatten(2).float().to(torch.bfloat16)


@lru_cache(maxsize=1)
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def apply(
        X,
        F,
        Y,
        N: tl.constexpr,
        S: tl.constexpr,
        H: tl.constexpr,
        PAIRS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offset < N
        token = (offset // (H * PAIRS)) % S
        pair = offset % PAIRS
        freq_offset = 2 * (token * PAIRS + pair)
        xr = tl.load(X + 2 * offset, valid, 0).to(tl.float64)
        xi = tl.load(X + 2 * offset + 1, valid, 0).to(tl.float64)
        fr = tl.load(F + freq_offset, valid, 0)
        fi = tl.load(F + freq_offset + 1, valid, 0)
        real = xr * fr - xi * fi
        imag = xr * fi + xi * fr
        tl.store(Y + 2 * offset, real.to(tl.float32).to(tl.bfloat16), valid)
        tl.store(Y + 2 * offset + 1, imag.to(tl.float32).to(tl.bfloat16), valid)

    return triton, apply


def fused_rope_apply(x, freqs, num_heads):
    _logical_contract(x, freqs, num_heads)
    if (
        x.device.type != "cuda"
        or x.shape[0] > 64
        or x.shape[1] not in (32, 120)
        or num_heads != 24
        or x.shape[2] != 24 * 128
    ):
        raise ValueError("Fused RoPE requires CUDA B<=64, S=32/120, H=24, D=128")
    if torch.is_grad_enabled():
        raise ValueError("Fused RoPE requires no_grad/inference_mode execution")
    triton, apply = _kernel()
    result = torch.empty_like(x)
    pairs = x.numel() // 2

    real_freqs = torch.view_as_real(freqs)
    with torch.cuda.device(x.device):
        apply[(triton.cdiv(pairs, 256),)](
            x,
            real_freqs,
            result,
            pairs,
            x.shape[1],
            num_heads,
            64,
            256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return result


def _source_contract(function, expected_hashes):
    try:
        source = inspect.getsource(function).replace("\r\n", "\n")
    except (OSError, TypeError) as exc:
        raise ValueError("Cannot verify official RoPE source contract") from exc
    if hashlib.sha256(source.encode()).hexdigest() not in expected_hashes:
        raise ValueError("Official RoPE source contract changed; candidate disabled")


@contextmanager
def fused_fp64_rope(backbone, *, enabled=True):
    if type(enabled) is not bool:
        raise ValueError("enabled must be boolean")
    stats = dict(
        enabled=enabled,
        calls=0,
        video_calls=0,
        action_calls=0,
        elements=0,
        arithmetic="fp64",
        output="bfloat16",
        fp_fusion=False,
    )
    if not enabled:
        yield stats
        return
    from fastwam.models.wan22.mot import MoT
    from fastwam.models.wan22.wan_video_dit import rope_apply

    mot = backbone.mot
    name = "_build_expert_attention_io"
    if getattr(backbone, "_newm_rope_candidate_active", False):
        raise RuntimeError("Nested/concurrent fused RoPE contexts are unsupported")
    if (
        type(mot) is not MoT
        or mot.mixtures["video"] is not backbone.video_expert
        or mot.mixtures["action"] is not backbone.action_expert
        or any(m.training for m in backbone.modules())
        or any(p.requires_grad for p in backbone.parameters())
    ):
        raise ValueError("Fused RoPE requires the evaluation-mode frozen official core")
    original = getattr(mot, name)
    function = MoT._build_expert_attention_io
    if (
        getattr(original, "__func__", None) is not function
        or original.__self__ is not mot
    ):
        raise ValueError("Fused RoPE requires the unmodified official instance method")
    if function.__globals__.get("rope_apply") is not rope_apply:
        raise ValueError("Official MoT RoPE global was replaced")
    _source_contract(function, _BUILD_HASHES)
    _source_contract(rope_apply, (_ROPE_SHA256,))
    if tuple(inspect.signature(function).parameters) != (
        "self",
        "expert",
        "block",
        "x",
        "freqs",
        "t_mod",
    ):
        raise ValueError("Official MoT attention signature changed")
    topology = (
        mot.num_heads,
        mot.attn_head_dim,
        mot.num_layers,
        tuple(backbone.video_expert.blocks),
        tuple(backbone.action_expert.blocks),
    )
    poisoned = False

    def replacement(x, freqs, num_heads):
        nonlocal poisoned
        try:
            if poisoned:
                raise RuntimeError("A failed fused RoPE context cannot be reused")
            if (
                backbone.training
                or mot.training
                or backbone.video_expert.training
                or backbone.action_expert.training
                or torch.is_grad_enabled()
            ):
                raise ValueError(
                    "Fused RoPE requires frozen no_grad/inference_mode execution"
                )
            if (
                MoT._build_expert_attention_io is not function
                or function.__globals__.get("rope_apply") is not rope_apply
                or (
                    mot.num_heads,
                    mot.attn_head_dim,
                    mot.num_layers,
                    tuple(backbone.video_expert.blocks),
                    tuple(backbone.action_expert.blocks),
                )
                != topology
                or num_heads != mot.num_heads
            ):
                raise RuntimeError(
                    "Official MoT method/topology signature changed inside RoPE context"
                )
            output = fused_rope_apply(x, freqs, num_heads)
            stats["calls"] += 1
            stats["video_calls"] += int(x.shape[1] == 120)
            stats["action_calls"] += int(x.shape[1] == 32)
            stats["elements"] += x.numel()
            return output
        except BaseException:
            poisoned = True
            raise

    globals_copy = dict(function.__globals__)
    globals_copy["rope_apply"] = replacement
    clone = FunctionType(
        function.__code__,
        globals_copy,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__annotations__ = dict(function.__annotations__)
    previous = (name in mot.__dict__, mot.__dict__.get(name))
    marker = (
        "_newm_rope_candidate_active" in backbone.__dict__,
        backbone.__dict__.get("_newm_rope_candidate_active"),
    )
    try:
        backbone._newm_rope_candidate_active = True
        setattr(mot, name, MethodType(clone, mot))
        yield stats
        if poisoned:
            raise RuntimeError(
                "A failed fused RoPE extraction was caught inside the context"
            )
    finally:
        if previous[0]:
            setattr(mot, name, previous[1])
        else:
            mot.__dict__.pop(name, None)
        if marker[0]:
            backbone._newm_rope_candidate_active = marker[1]
        else:
            backbone.__dict__.pop("_newm_rope_candidate_active", None)
