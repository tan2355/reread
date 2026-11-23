import threading
from contextlib import contextmanager

import torch

from reread.backbone.projection_backend import projection_backend
from reread.config import ReReadConfig


class FrozenFastWAM:
    def __init__(
        self,
        backbone,
        config=ReReadConfig(),
        *,
        cache_conditioning=False,
        projection="conv3d",
    ):
        self.backbone, self.config = backbone, config
        self.cache_conditioning = cache_conditioning
        if projection not in ("conv3d", "linear"):
            raise ValueError("Unknown projection backend")
        self.projection = projection
        self._lock = threading.Lock()
        backbone.requires_grad_(False).eval()
        mot = backbone.mot
        if (
            max(config.layers) >= mot.num_layers
            or mot.num_heads * mot.attn_head_dim != config.kv_dim
        ):
            raise ValueError("KV dimensions/layers do not match upstream MoT")
        self._blocks = {
            id(mot.mixtures[modality].blocks[layer]): (modality, layer)
            for modality in ("video", "action")
            for layer in config.layers
        }

    @contextmanager
    def _conditioning_cache(self):
        if not self.cache_conditioning:
            yield
            return
        expert = self.backbone.action_expert
        modules = [expert.text_embedding]
        for block in expert.blocks:
            modules.extend(
                (block.cross_attn.k, block.cross_attn.v, block.cross_attn.norm_k)
            )
        restored = []

        def cached_forward(original):
            cached = None

            def forward(value):
                nonlocal cached
                version = None if value.is_inference() else value._version
                if cached is None:
                    cached = (value, version, original(value))
                elif value is not cached[0] or version != cached[1]:
                    raise RuntimeError(
                        "Denoising condition changed inside one frozen extraction"
                    )
                return cached[2]

            return forward

        try:
            for module in modules:
                had_override = "forward" in module.__dict__
                old_override = module.__dict__.get("forward")
                restored.append((module, had_override, old_override))
                module.forward = cached_forward(module.forward)
            yield
        finally:
            for module, had_override, old_override in reversed(restored):
                if had_override:
                    module.forward = old_override
                else:
                    delattr(module, "forward")

    @torch.no_grad()
    def extract(self, **observation):
        c = self.config
        fixed = dict(
            action_horizon=c.horizon,
            num_inference_steps=c.inference_steps,
            sigma_shift=c.sigma_shift,
            compile_action_infer=False,
        )
        for name, value in fixed.items():
            if name in observation and observation[name] != value:
                raise ValueError(f"{name} is fixed to {value} by feature contract")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError(
                "One FrozenFastWAM instance cannot serve concurrent extraction"
            )
        encoder_name = "_encode_input_image_latents_tensor"
        had_encoder = encoder_name in self.backbone.__dict__
        old_encoder = self.backbone.__dict__.get(encoder_name)
        patched_encoder = False
        try:
            latent = observation.pop("rgb_latents", None)
            if latent is not None:
                if "input_image" in observation:
                    raise ValueError("Provide either current RGB latent or RGB image")
                if latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[2] != 1:
                    raise ValueError(
                        "Only current-frame latent [1,C,1,H,W] is allowed; no future frames"
                    )
                if (
                    latent.shape[1] != self.backbone.video_expert.in_dim
                    or not torch.isfinite(latent).all()
                ):
                    raise ValueError("Invalid cached RGB latent channels or values")
                latent = latent.detach().to(
                    device=self.backbone.device, dtype=self.backbone.torch_dtype
                )

                setattr(self.backbone, encoder_name, lambda **kwargs: latent)
                patched_encoder = True
                observation["input_image"] = torch.empty(
                    1, 3, latent.shape[3] * 16, latent.shape[4] * 16
                )
            self.backbone.eval()
            with (
                projection_backend(self.backbone, self.projection),
                self._conditioning_cache(),
                self._capture() as (saved, counts),
            ):
                result = self.backbone.infer_action(**(observation | fixed))
            for modality in ("video", "action"):
                expected = 1 if modality == "video" else c.inference_steps
                if any(counts.get((modality, layer)) != expected for layer in c.layers):
                    raise RuntimeError(
                        "Upstream inference call graph changed; refusing inconsistent KV"
                    )
            features = {
                name: torch.stack(
                    [
                        torch.cat(
                            [
                                saved[(modality, layer)][name]
                                for modality in ("video", "action")
                            ],
                            dim=1,
                        )
                        for layer in c.layers
                    ],
                    dim=1,
                ).detach()
                for name in ("k", "v")
            }
            action = result["action"].unsqueeze(0).to(features["k"].device).detach()
            if action.shape != (1, c.horizon, c.action_dim):
                raise ValueError("Unexpected upstream action shape")
            return action, features
        finally:
            if patched_encoder:
                if had_encoder:
                    setattr(self.backbone, encoder_name, old_encoder)
                else:
                    delattr(self.backbone, encoder_name)
            self._lock.release()

    @torch.no_grad()
    def extract_batch(
        self, *, rgb_latents, proprio, context, context_mask, seed=0
    ):
        c, base = self.config, self.backbone
        if rgb_latents.ndim != 5 or rgb_latents.shape[2] != 1:
            raise ValueError("Only current RGB latent [B,C,1,H,W] is allowed")
        batch = rgb_latents.shape[0]
        if batch < 1 or rgb_latents.shape[1] != base.video_expert.in_dim:
            raise ValueError("Invalid cached RGB latent batch or channels")
        if (
            proprio.shape != (batch, base.proprio_dim)
            or context.ndim != 3
            or context.shape[0] != batch
            or context_mask.shape != context.shape[:2]
        ):
            raise ValueError("Batched proprio/context shapes do not match RGB batch")
        if not torch.isfinite(rgb_latents).all():
            raise ValueError("Nonfinite cached RGB latent")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError(
                "One FrozenFastWAM instance cannot serve concurrent extraction"
            )
        try:
            base.eval()
            first_frame_latents = rgb_latents.detach().to(
                device=base.device, dtype=base.torch_dtype
            )
            context = context.to(
                device=base.device, dtype=base.torch_dtype, non_blocking=True
            )
            context_mask = context_mask.to(
                device=base.device, dtype=torch.bool, non_blocking=True
            )
            proprio = proprio.to(
                device=base.device, dtype=base.torch_dtype, non_blocking=True
            )
            context, context_mask = base._append_proprio_to_context(
                context, context_mask, proprio
            )
            generator = torch.Generator(device="cpu").manual_seed(seed)

            initial = torch.randn(
                (1, c.horizon, c.action_dim),
                generator=generator,
                device="cpu",
                dtype=torch.float32,
            )
            latents_action = (
                initial.expand(batch, -1, -1)
                .clone()
                .to(device=base.device, dtype=base.torch_dtype)
            )
            timestep_video = torch.zeros(
                (batch,), dtype=first_frame_latents.dtype, device=base.device
            )
            with projection_backend(base, self.projection):
                (
                    video_tokens,
                    _t_video,
                    video_t_mod,
                    video_context,
                    video_context_mask,
                    video_freqs,
                    _f_video,
                    _h_video,
                    _w_video,
                    tokens_per_frame,
                ) = base.video_expert.prepare(
                    x=first_frame_latents,
                    timestep=timestep_video,
                    context=context,
                    context_mask=context_mask,
                    action=None,
                    fuse_vae_embedding_in_latents=bool(
                        getattr(
                            base.video_expert, "fuse_vae_embedding_in_latents", False
                        )
                    ),
                )
            video_seq_len = video_tokens.shape[1]
            attention_mask = base._build_mot_attention_mask(
                video_seq_len=video_seq_len,
                action_seq_len=c.horizon,
                video_tokens_per_frame=tokens_per_frame,
                device=video_tokens.device,
            )
            with self._conditioning_cache(), self._capture() as (saved, counts):
                video_cache_k, video_cache_v = base.mot.prefill_video_cache_tensor(
                    video_tokens=video_tokens,
                    video_freqs=video_freqs,
                    video_t_mod=video_t_mod,
                    video_context=video_context,
                    video_context_mask=video_context_mask,
                    video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
                )
                timesteps, deltas = (
                    base.infer_action_scheduler.build_inference_schedule(
                        num_inference_steps=c.inference_steps,
                        device=base.device,
                        dtype=latents_action.dtype,
                        shift_override=c.sigma_shift,
                    )
                )
                for step_t, step_delta in zip(timesteps, deltas):
                    timestep_action = step_t.expand(batch).to(
                        device=base.device, dtype=latents_action.dtype
                    )
                    prediction = base._denoise_action_with_video_cache(
                        latents_action=latents_action,
                        timestep_action=timestep_action,
                        context=context,
                        context_mask=context_mask,
                        video_cache_k=video_cache_k,
                        video_cache_v=video_cache_v,
                        action_attention_mask=attention_mask[video_seq_len:, :],
                    )
                    latents_action = base.infer_action_scheduler.step(
                        prediction, step_delta, latents_action
                    )
            for modality in ("video", "action"):
                expected = 1 if modality == "video" else c.inference_steps
                if any(counts.get((modality, layer)) != expected for layer in c.layers):
                    raise RuntimeError(
                        "Upstream inference call graph changed; refusing inconsistent KV"
                    )
            features = {
                name: torch.stack(
                    [
                        torch.cat(
                            [
                                saved[(modality, layer)][name]
                                for modality in ("video", "action")
                            ],
                            dim=1,
                        )
                        for layer in c.layers
                    ],
                    dim=1,
                ).detach()
                for name in ("k", "v")
            }
            return latents_action.float().detach(), features
        finally:
            self._lock.release()
