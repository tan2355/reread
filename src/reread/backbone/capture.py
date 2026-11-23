import threading
from contextlib import contextmanager

import torch

from reread.backbone.capture_base import FrozenFastWAM
from reread.config import ReReadConfig
from reread.memory.spatial import pool_video_tokens


class FrozenSpatialFastWAM(FrozenFastWAM):

    def __init__(self, backbone, config=ReReadConfig(), **kwargs):
        if not isinstance(config, ReReadConfig):
            raise TypeError("Spatial extraction requires the versioned ReReadConfig")
        super().__init__(backbone, config, **kwargs)
        if not callable(getattr(backbone.video_expert, "prepare", None)):
            raise ValueError(
                "Upstream video expert must expose its tensor prepare method"
            )
        self._instrumentation_lock = threading.Lock()
        for modality in ("video", "action"):
            for layer in config.layers:
                attention = backbone.mot.mixtures[modality].blocks[layer].self_attn
                if not all(
                    isinstance(getattr(attention, name, None), torch.nn.Module)
                    for name in ("norm_k", "v")
                ):
                    raise ValueError(
                        "Upstream attention must expose normalized K and V modules"
                    )

    @contextmanager
    def _verified_grid(self):
        if not self._instrumentation_lock.acquire(blocking=False):
            raise RuntimeError(
                "One FrozenSpatialFastWAM instance cannot serve concurrent extraction"
            )
        expert, c = self.backbone.video_expert, self.config
        name, prepared = "prepare", 0
        original = getattr(expert, name)
        had_override = name in expert.__dict__
        previous_override = expert.__dict__.get(name)

        def verified(*args, **kwargs):
            nonlocal prepared
            x = kwargs.get("x", args[0] if args else None)
            patch = tuple(getattr(expert, "patch_size", ()))
            if (
                len(patch) != 3
                or patch[0] != 1
                or min(patch) < 1
                or not isinstance(x, torch.Tensor)
                or x.ndim != 5
                or x.shape[2] != 1
                or tuple(x.shape[-2:])
                != (c.video_grid[0] * patch[1], c.video_grid[1] * patch[2])
            ):
                raise ValueError(
                    "Current latent/patch grid does not match the spatial memory contract"
                )
            result = original(*args, **kwargs)
            if (
                not isinstance(result, tuple)
                or len(result) != 10
                or tuple(result[6:9]) != (1, *c.video_grid)
                or result[9] != c.video_grid[0] * c.video_grid[1]
                or result[0].ndim != 3
                or result[0].shape[1] != c.video_grid[0] * c.video_grid[1]
            ):
                raise RuntimeError("Upstream video prepare grid/call contract changed")
            prepared += 1
            if prepared != 1:
                raise RuntimeError(
                    "Current real RGB must be prepared exactly once per decision"
                )
            return result

        try:
            setattr(expert, name, verified)
            yield
            if prepared != 1:
                raise RuntimeError("Current real RGB prepare was skipped")
        finally:
            if had_override:
                setattr(expert, name, previous_override)
            else:
                delattr(expert, name)
            self._instrumentation_lock.release()

    @torch.no_grad()
    def extract(self, **observation):
        with self._verified_grid():
            return super().extract(**observation)

    @torch.no_grad()
    def extract_batch(self, **observation):
        with self._verified_grid():
            return super().extract_batch(**observation)

    @contextmanager
    def _capture(self):
        mot, c = self.backbone.mot, self.config
        name = "_build_expert_attention_io"
        original = getattr(mot, name)
        had_override = name in mot.__dict__
        previous_override = mot.__dict__.get(name)
        saved, counts, handles = {}, {}, []
        active = None

        def observe(key, kind):
            def hook(module, args, output):
                if active is None or active["key"] != key:
                    raise RuntimeError(
                        "Selected attention module ran outside the expected MoT call"
                    )
                active["seen"][kind] += 1
                expected_tokens = (
                    c.video_grid[0] * c.video_grid[1]
                    if key[0] == "video"
                    else c.action_tokens
                )
                if (
                    not isinstance(output, torch.Tensor)
                    or output.ndim != 3
                    or output.shape[1:] != (expected_tokens, c.kv_dim)
                ):
                    raise RuntimeError("Upstream pre-RoPE content shape changed")
                if active["save"]:
                    active["content"][kind] = (
                        pool_video_tokens(output, c)
                        if key[0] == "video"
                        else output.detach().clone()
                    )

            return hook

        def capture(*args, **kwargs):
            nonlocal active
            block = kwargs.get("block", args[1] if len(args) > 1 else None)
            key = self._blocks.get(id(block))
            if key is None:
                return original(*args, **kwargs)
            if active is not None:
                raise RuntimeError("Nested upstream attention capture is unsupported")
            counts[key] = counts.get(key, 0) + 1
            active = {
                "key": key,
                "save": key[0] == "video" or counts[key] == c.inference_steps,
                "seen": {"k": 0, "v": 0},
                "content": {},
            }
            try:
                result = original(*args, **kwargs)
                if active["seen"] != {"k": 1, "v": 1}:
                    raise RuntimeError(
                        "Upstream normalized K/V projection call graph changed"
                    )
                if active["save"]:
                    content = active["content"]
                    if (
                        content["k"].shape != content["v"].shape
                        or content["k"].device != content["v"].device
                    ):
                        raise RuntimeError("Upstream K/V content alignment changed")
                    saved[key] = content
                return result
            finally:
                active = None

        try:
            for modality in ("video", "action"):
                for layer in c.layers:
                    key = (modality, layer)
                    attention = mot.mixtures[modality].blocks[layer].self_attn
                    handles.append(
                        attention.norm_k.register_forward_hook(observe(key, "k"))
                    )
                    handles.append(attention.v.register_forward_hook(observe(key, "v")))
            setattr(mot, name, capture)
            yield saved, counts
        finally:
            for handle in handles:
                handle.remove()
            if had_override:
                setattr(mot, name, previous_override)
            else:
                delattr(mot, name)
