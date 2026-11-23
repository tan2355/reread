from contextlib import contextmanager
from types import FunctionType, MethodType

import torch


def _rebind_attention(bound, attention):
    fn = bound.__func__ if isinstance(bound, MethodType) else bound
    if (
        not isinstance(fn, FunctionType)
        or "flash_attention" not in fn.__code__.co_names
    ):
        raise ValueError("Upstream attention function contract changed")
    scope = dict(fn.__globals__)
    scope["flash_attention"] = attention
    replacement = FunctionType(
        fn.__code__, scope, fn.__name__, fn.__defaults__, fn.__closure__
    )
    replacement.__kwdefaults__ = fn.__kwdefaults__
    return (
        MethodType(replacement, bound.__self__)
        if isinstance(bound, MethodType)
        else replacement
    )


class AllVisibleAttention:
    def __init__(self, original):
        self.original = original
        self.known = {}
        self.removed = self.preserved = 0

    def __call__(self, q, k, v, num_heads, ctx_mask=None, compatibility_mode=True):
        if ctx_mask is not None and ctx_mask.dtype == torch.bool:
            version = None if ctx_mask.is_inference() else ctx_mask._version
            key = (
                ctx_mask.untyped_storage().data_ptr(),
                ctx_mask.storage_offset(),
                tuple(ctx_mask.shape),
                tuple(ctx_mask.stride()),
                version,
                ctx_mask.device,
            )

            cacheable = not ctx_mask.is_inference()
            cached = self.known.get(key) if cacheable else None
            if cached is None:
                visible = bool(ctx_mask.all())
                if len(self.known) >= 16:
                    self.known.clear()
                if cacheable:
                    self.known[key] = (ctx_mask, visible)
            else:
                visible = cached[1]
            if visible:
                ctx_mask = None
                self.removed += 1
            else:
                self.preserved += 1
        return self.original(
            q=q,
            k=k,
            v=v,
            num_heads=num_heads,
            ctx_mask=ctx_mask,
            compatibility_mode=compatibility_mode,
        )


@contextmanager
def unmasked_frozen_attention(backbone, enabled=True):
    if not enabled:
        yield None
        return
    roots = (backbone.video_expert, backbone.action_expert)
    if any(module.training for root in roots for module in root.modules()) or any(
        p.requires_grad for root in roots for p in root.parameters()
    ):
        raise ValueError("Fused attention scheduling requires frozen eval experts")
    mot = backbone.mot
    methods = [
        (mot, "prefill_video_cache_tensor"),
        (mot, "forward_action_with_video_cache_tensor"),
    ]
    for expert in roots:
        methods.extend((block.cross_attn, "forward") for block in expert.blocks)
    restored = []
    original = mot.prefill_video_cache_tensor.__func__.__globals__["flash_attention"]
    optimized = AllVisibleAttention(original)
    try:
        for owner, name in methods:
            bound = getattr(owner, name)
            restored.append(
                (owner, name, name in owner.__dict__, owner.__dict__.get(name))
            )
            setattr(owner, name, _rebind_attention(bound, optimized))
        yield optimized
    finally:
        for owner, name, existed, old in reversed(restored):
            if existed:
                setattr(owner, name, old)
            else:
                delattr(owner, name)
