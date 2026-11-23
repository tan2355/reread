from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from reread.backbone.capture import FrozenSpatialFastWAM
from reread.config import ReReadConfig
from reread.memory.spatial import pool_video_tokens


class CountingLinear(nn.Linear):
    def __init__(self, width):
        super().__init__(width, width, bias=False)
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return super().forward(value)


class FakeBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.k = CountingLinear(width)
        self.self_attn.v = CountingLinear(width)
        self.self_attn.norm_k = nn.Identity()
        self.cross_attn = nn.Module()
        self.cross_attn.k = nn.Identity()
        self.cross_attn.v = nn.Identity()
        self.cross_attn.norm_k = nn.Identity()


class FakeVideo(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.in_dim, self.patch_size = 4, (1, 2, 2)
        self.patch_embedding = nn.Conv3d(4, width, self.patch_size, self.patch_size)
        self.blocks = nn.ModuleList([FakeBlock(width) for _ in range(3)])
        self.prepare_calls, self.wrong_grid = 0, False

    def prepare(self, *, x, timestep, context, context_mask, **kwargs):
        self.prepare_calls += 1
        tokens = self.patch_embedding(x).flatten(2).transpose(1, 2)
        h, w = x.shape[3] // 2, x.shape[4] // 2
        if self.wrong_grid:
            h, w = w, h
        return tokens, timestep, None, context, context_mask, None, 1, h, w, h * w


class FakeAction(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.blocks = nn.ModuleList([FakeBlock(width) for _ in range(3)])
        self.text_embedding = nn.Identity()


class FakeMoT(nn.Module):
    def __init__(self, video, action, width):
        super().__init__()
        self.mixtures = nn.ModuleDict({"video": video, "action": action})
        self.num_layers, self.num_heads, self.attn_head_dim = 3, 2, width // 2
        self.contents = {}
        self.skip_norm = False
        self.duplicate_video = False
        self.fail = False

    def _build_expert_attention_io(self, expert, block, x, freqs=None, t_mod=None):
        k = block.self_attn.k(x)
        if not self.skip_norm:
            k = block.self_attn.norm_k(k)
        v = block.self_attn.v(x)

        self.contents[id(block)] = (k.detach().clone(), v.detach().clone())
        k.add_(100)
        if self.fail:
            raise RuntimeError("fake upstream failure")
        return x, k, v

    def prefill_video_cache_tensor(self, *, video_tokens, **kwargs):
        outputs = [
            self._build_expert_attention_io(self.mixtures["video"], b, video_tokens)
            for b in self.mixtures["video"].blocks
        ]
        if self.duplicate_video:
            self._build_expert_attention_io(
                self.mixtures["video"], self.mixtures["video"].blocks[1], video_tokens
            )
        return torch.stack([o[1] for o in outputs]), torch.stack(
            [o[2] for o in outputs]
        )


class FakeBase(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c = config
        self.device, self.torch_dtype, self.proprio_dim = (
            torch.device("cpu"),
            torch.float32,
            2,
        )
        self.video_expert = FakeVideo(config.kv_dim)
        self.action_expert = FakeAction(config.kv_dim)
        self.mot = FakeMoT(self.video_expert, self.action_expert, config.kv_dim)
        self.infer_action_scheduler = SimpleNamespace(
            build_inference_schedule=lambda num_inference_steps, **k: (
                torch.arange(num_inference_steps, dtype=torch.float32),
                torch.full((num_inference_steps,), 0.1),
            ),
            step=lambda prediction, delta, action: action - delta * prediction,
        )
        self.encode_calls = 0
        self.skip_last_action = False

    def _encode_input_image_latents_tensor(self, *, input_image, **kwargs):
        self.encode_calls += 1
        rgb = F.adaptive_avg_pool2d(input_image, (24, 20))
        return torch.cat((rgb, rgb[:, :1]), 1).unsqueeze(2)

    def _append_proprio_to_context(self, context, context_mask, proprio):
        return context, context_mask

    def _build_mot_attention_mask(
        self, *, video_seq_len, action_seq_len, device, **kwargs
    ):
        return torch.ones(
            video_seq_len + action_seq_len,
            video_seq_len + action_seq_len,
            device=device,
        )

    def _denoise_action_with_video_cache(
        self, *, latents_action, video_cache_k, **kwargs
    ):
        x = latents_action.repeat(1, 1, self.c.kv_dim // self.c.action_dim)
        outputs = [
            self.mot._build_expert_attention_io(self.action_expert, b, x)
            for b in self.action_expert.blocks
        ]
        return (
            torch.stack([o[1] + o[2] for o in outputs]).mean(0)[
                ..., : self.c.action_dim
            ]
            * 0.01
        )

    def infer_action(
        self,
        *,
        input_image,
        action_horizon,
        num_inference_steps,
        seed=0,
        context,
        context_mask,
        proprio,
        **kwargs,
    ):
        latent = self._encode_input_image_latents_tensor(input_image=input_image)
        prepared = self.video_expert.prepare(
            x=latent,
            timestep=torch.zeros(1),
            context=context,
            context_mask=context_mask,
        )
        k, v = self.mot.prefill_video_cache_tensor(video_tokens=prepared[0])
        action = torch.randn(
            1,
            action_horizon,
            self.c.action_dim,
            generator=torch.Generator().manual_seed(seed),
        )
        for _ in range(num_inference_steps - int(self.skip_last_action)):
            action = action - 0.1 * self._denoise_action_with_video_cache(
                latents_action=action, video_cache_k=k
            )
        return {"action": action[0]}


@pytest.fixture
def fixture():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    config = replace(
        ReReadConfig(),
        action_dim=2,
        proprio_dim=2,
        horizon=4,
        action_tokens=4,
        kv_dim=16,
        layers=(1, 2),
        inference_steps=3,
        decision_stride=2,
    )
    base = FakeBase(config)
    obs = dict(
        input_image=torch.randn(1, 3, 384, 320),
        context=torch.randn(1, 3, 8),
        context_mask=torch.ones(1, 3, dtype=torch.bool),
        proprio=torch.zeros(2),
        seed=7,
    )
    return config, base, obs


def assert_restored(base):
    assert "_build_expert_attention_io" not in base.mot.__dict__
    assert "prepare" not in base.video_expert.__dict__
    for expert in (base.video_expert, base.action_expert):
        for block in expert.blocks:
            assert not block.self_attn.norm_k._forward_hooks
            assert not block.self_attn.v._forward_hooks


def test_exact_action_readonly_pre_rope_features_call_counts(fixture):
    c, base, obs = fixture
    base.requires_grad_(False)
    with torch.no_grad():
        expected = base.infer_action(
            **obs, action_horizon=c.horizon, num_inference_steps=c.inference_steps
        )["action"]
    counts = {
        (m, i, n): getattr(b.self_attn, n).calls
        for m, e in base.mot.mixtures.items()
        for i, b in enumerate(e.blocks)
        for n in ("k", "v")
    }
    writer = FrozenSpatialFastWAM(base, c, cache_conditioning=True)
    action, content = writer.extract(**obs)
    assert torch.equal(action[0], expected)
    assert set(content) == {"k", "v"}
    assert content["k"].shape == (1, 2, 68, 16)
    for m, e in base.mot.mixtures.items():
        for i, b in enumerate(e.blocks):
            for n in ("k", "v"):
                assert getattr(b.self_attn, n).calls - counts[(m, i, n)] == (
                    1 if m == "video" else 3
                )
    for j, layer in enumerate(c.layers):
        for n, name in enumerate(("k", "v")):
            video = base.mot.contents[id(base.video_expert.blocks[layer])][n]
            plan = base.mot.contents[id(base.action_expert.blocks[layer])][n]
            assert torch.equal(content[name][:, j, :64], pool_video_tokens(video, c))
            assert torch.equal(content[name][:, j, 64:], plan)
            assert not content[name].requires_grad
    assert base.encode_calls == 2 and base.video_expert.prepare_calls == 2
    assert all(not p.requires_grad and p.grad is None for p in base.parameters())
    assert_restored(base)


def test_batch_matches_single_without_duplicate_backbone_projections(fixture):
    c, base, obs = fixture
    writer = FrozenSpatialFastWAM(base, c)
    latent = base._encode_input_image_latents_tensor(input_image=obs["input_image"])
    single_a, single_f = writer.extract(
        **{k: v for k, v in obs.items() if k != "input_image"}, rgb_latents=latent
    )
    batch_a, batch_f = writer.extract_batch(
        rgb_latents=latent.repeat(2, 1, 1, 1, 1),
        proprio=obs["proprio"].repeat(2, 1),
        context=obs["context"].repeat(2, 1, 1),
        context_mask=obs["context_mask"].repeat(2, 1),
        seed=obs["seed"],
    )
    torch.testing.assert_close(
        batch_a, single_a.expand_as(batch_a), rtol=1e-6, atol=1e-6
    )
    for name in single_f:
        torch.testing.assert_close(
            batch_f[name], single_f[name].expand_as(batch_f[name]), rtol=1e-6, atol=1e-6
        )
    assert base.encode_calls == 1 and base.video_expert.prepare_calls == 2
    assert_restored(base)


@pytest.mark.parametrize(
    "failure", ["fail", "skip_norm", "duplicate_video", "wrong_grid"]
)
def test_failure_cleanup_and_recovery(fixture, failure):
    c, base, obs = fixture
    writer = FrozenSpatialFastWAM(base, c, cache_conditioning=True)
    target = base.video_expert if failure == "wrong_grid" else base.mot
    setattr(target, failure, True)
    with pytest.raises((RuntimeError, ValueError)):
        writer.extract(**obs)
    assert_restored(base)
    assert not writer._lock.locked() and not writer._instrumentation_lock.locked()
    setattr(target, failure, False)
    writer.extract(**obs)
    assert_restored(base)


def test_same_token_count_transposed_grid_is_rejected_for_batch(fixture):
    c, base, obs = fixture
    writer = FrozenSpatialFastWAM(base, c)
    base.video_expert.wrong_grid = True
    with pytest.raises(RuntimeError, match="grid"):
        writer.extract_batch(
            rgb_latents=torch.zeros(1, 4, 1, 24, 20),
            proprio=torch.zeros(1, 2),
            context=obs["context"],
            context_mask=obs["context_mask"],
        )
    assert_restored(base)


def test_concurrency_guards_and_changed_input_shape_reject(fixture):
    c, base, obs = fixture
    writer = FrozenSpatialFastWAM(base, c)
    writer._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="concurrent"):
            writer.extract(**obs)
    finally:
        writer._lock.release()
    assert_restored(base)
    writer._instrumentation_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="concurrent"):
            writer.extract(**obs)
    finally:
        writer._instrumentation_lock.release()
    with pytest.raises(ValueError, match="grid"):
        writer.extract(
            **{k: v for k, v in obs.items() if k != "input_image"},
            rgb_latents=torch.zeros(1, 4, 1, 20, 24),
        )
    assert_restored(base)


def test_final_action_call_missing_refuses_partial_memory(fixture):
    c, base, obs = fixture
    writer = FrozenSpatialFastWAM(base, c)
    base.skip_last_action = True
    with pytest.raises(RuntimeError, match="call graph"):
        writer.extract(**obs)
    assert_restored(base)
    base.skip_last_action = False
    writer.extract(**obs)


def test_existing_instance_overrides_and_external_hooks_survive(fixture):
    c, base, obs = fixture
    prepare = base.video_expert.prepare
    build = base.mot._build_expert_attention_io
    base.video_expert.prepare = lambda **kw: prepare(**kw)
    base.mot._build_expert_attention_io = lambda *a, **kw: build(*a, **kw)
    prepare_override = base.video_expert.prepare
    build_override = base.mot._build_expert_attention_io
    called = []
    module = base.video_expert.blocks[1].self_attn.norm_k
    handle = module.register_forward_hook(lambda m, a, out: called.append(out.shape))
    try:
        writer = FrozenSpatialFastWAM(base, c)
        writer.extract(**obs)
        assert base.video_expert.prepare is prepare_override
        assert base.mot._build_expert_attention_io is build_override
        assert len(module._forward_hooks) == 1 and len(called) == 1
    finally:
        handle.remove()
