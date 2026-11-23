from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from reread.config import ReReadConfig
from reread.engine import CausalMemoryBanks, stream_updates
from reread.model import ReRead


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(**kwargs):
    return ReReadConfig(
        **(
            dict(
                action_dim=14,
                proprio_dim=14,
                horizon=4,
                action_tokens=4,
                kv_dim=12,
                layers=(1,),
                video_tokens=6,
                width=24,
                heads=4,
                depth=2,
                ff_dim=48,
                inference_steps=3,
                decision_stride=2,
                history_dropout=0.0,
                video_grid=(3, 4),
                head_rows=1,
                wrist_split=2,
                camera_pools=((1, 2), (1, 2), (1, 2)),
                activation_checkpointing=False,
            )
            | kwargs
        )
    )


def sample(frame=0, decision=0, **overrides):
    result = dict(
        lane=0,
        episode_id="turn_switch/randomized/e0",
        trajectory_pass_id=0,
        frame=frame,
        raw_frame=frame,
        decision_index=decision,
        cache_segment=0,
        cache_variant=1,
        image_augmented=True,
        prompt_id="unseen_id",
        proprio=torch.zeros(14),
        action=torch.ones(4, 14) * 0.1,
        valid=torch.tensor([True, True, True, False]),
        cached_encoded=dict(
            rgb_latents=torch.ones(3, 1, 6, 8) * (frame + 1),
            context=torch.zeros(4, 8),
            context_mask=torch.ones(4, dtype=torch.bool),
        ),
    )
    result.update(overrides)
    return result


def encoded(s):
    return {
        **{k: v[None] for k, v in s["cached_encoded"].items()},
        "proprio": s["proprio"][None],
    }


class Frozen:
    def __init__(self, c):
        self.config = c
        self.backbone = SimpleNamespace(device="cpu")
        self.calls = []

    @torch.no_grad()
    def extract_batch(self, *, rgb_latents, proprio, context, context_mask, seed=0):
        c, b = self.config, len(proprio)
        marker = rgb_latents[:, :, 0, 0, 0].mean(1)
        self.calls.append(dict(marker=marker.clone(), seed=seed))
        actions = (
            marker[:, None, None].expand(b, c.horizon, c.action_dim).clone() * 0.01
        )
        k = (
            marker[:, None, None, None]
            .expand(b, len(c.layers), c.video_tokens + c.action_tokens, c.kv_dim)
            .clone()
        )
        return actions, dict(k=k, v=k * 0.3)


def fixture(**kwargs):
    torch.manual_seed(73)
    c = config(**kwargs)
    return ReRead(c), Frozen(c)


def test_endpoint_training_perturbs_only_candidate_and_shares_unchanged_full_kv(
    monkeypatch,
):
    model, frozen = fixture(action_perturb_probability=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rows = [sample(), sample(2, 1), sample(4, 2)]
    rows[0]["action"][-1] = float("nan")
    histories, metrics = [], []

    def record(_model, args, kwargs):
        candidate, memory = args
        expected_base = 0.01 * float(len(histories) * 2 + 1)
        assert not torch.allclose(
            candidate[:, :3, :6], torch.full_like(candidate[:, :3, :6], expected_base)
        )
        torch.testing.assert_close(
            candidate[:, :3, (6, 13)],
            torch.full_like(candidate[:, :3, (6, 13)], expected_base),
        )
        assert not candidate.requires_grad

        torch.testing.assert_close(
            candidate[:, -1], torch.full_like(candidate[:, -1], expected_base)
        )
        histories.append(memory["k"].clone())

    handle = model.register_forward_pre_hook(record, with_kwargs=True)
    result = stream_updates(
        frozen,
        model,
        optimizer,
        iter([[r] for r in rows]),
        prefetch=False,
        corrector_autocast=False,
        on_step=lambda x: metrics.append(x),
    )
    handle.remove()
    assert result == dict(updates=3, local_decisions=3)
    assert len(frozen.calls) == 3
    assert histories[0].shape == (1, 3, 1, 10, 12)
    assert histories[0][0, :2].count_nonzero() == 0
    assert histories[1][0, 1].unique().item() == 1
    assert histories[2][0, 0].unique().item() == 1
    assert histories[2][0, 1].unique().item() == 3
    assert metrics[0]["action_projection_grad_norm"] == 0
    assert metrics[1]["action_projection_grad_norm"] > 0
    assert all(x["output_grad_norm"] > 0 for x in metrics)
    assert all(
        x["base_forward_calls"] == 3 and x["corrector_forward_calls"] == 1
        for x in metrics
    )
    assert all(torch.isfinite(torch.tensor(x["total_loss"])) for x in metrics)
    assert all(p.grad is None for p in model.parameters())
    assert not model.output.weight.eq(0).all()


@pytest.mark.parametrize("failure", ["loss", "backward", "optimizer"])
def test_failed_training_does_not_commit_feature_bank(monkeypatch, failure):
    import reread.engine as engine

    model, frozen = fixture()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    banks = []

    class Banks(CausalMemoryBanks):
        def __init__(self, c):
            super().__init__(c)
            banks.append(self)

    monkeypatch.setattr(engine, "CausalMemoryBanks", Banks)
    if failure == "loss":
        monkeypatch.setattr(
            engine,
            "action_mse",
            lambda *a, **k: (
                torch.tensor(float("nan"), requires_grad=True),
                dict(total_loss=float("nan")),
            ),
        )
    elif failure == "backward":
        model.output.weight.register_hook(
            lambda grad: torch.full_like(grad, float("nan"))
        )
    else:
        monkeypatch.setattr(
            optimizer,
            "step",
            lambda: (_ for _ in ()).throw(RuntimeError("optimizer failed")),
        )
    with pytest.raises((FloatingPointError, RuntimeError)):
        stream_updates(
            frozen,
            model,
            optimizer,
            iter([[sample()]]),
            prefetch=False,
            corrector_autocast=False,
        )
    assert not banks[0].records_for_lane(0)


def test_history_preparation_is_transactional_owned_and_refuses_foreign_or_stale_publish():
    c = config()
    bank, other = CausalMemoryBanks(c), CausalMemoryBanks(c)
    k = torch.ones(1, 1, 10, 12)
    staged = bank.prepare(dict(k=k, v=k * 0.3), [sample()])
    assert not bank.records_for_lane(0)
    with pytest.raises(ValueError, match="foreign"):
        other.publish(staged)
    bank.publish(staged)
    with pytest.raises(ValueError, match="Stale"):
        bank.publish(staged)
    k.fill_(99)
    assert bank.records_for_lane(0)[0].k.unique().item() == 1
    copied = bank.records_for_lane(0)
    copied[0].k.fill_(88)
    assert bank.records_for_lane(0)[0].k.unique().item() == 1


def test_production_trainer_fixed_budget_and_forbids_legacy_noise_or_optimizations():
    import reread.training as trainer

    args = trainer.parser().parse_args(
        [
            "--launch-id",
            "a" * 32,
            "--cache-root",
            "cache",
            "--output",
            "output",
            "--task-list",
            "tasks.json",
            "--batched-bank",
        ]
    )
    trainer.validate_args(args)
    assert (args.steps, args.batch_size, args.grad_accum) == (20000, 64, 1)
    assert args.phase_weights == [1.0, 10.0]
    assert args.data_workers == args.cpu_threads == 4
    assert args.preencode_next and args.corrector_autocast
    assert not args.prefetch_frozen
    assert not args.frozen_inference_graph and not args.fused_fp64_rope
    trainer.validate_production_config(ReReadConfig())
    assert ReReadConfig().activation_checkpointing is False

    with pytest.raises(ValueError, match="fixed"):
        trainer.validate_production_config(
            replace(ReReadConfig(), activation_checkpointing=True)
        )
    with pytest.raises(ValueError, match="fixed"):
        trainer.validate_production_config(
            replace(ReReadConfig(), action_perturb_probability=0.3)
        )
    for name, value in (
        ("steps", 40000),
        ("batch_size", 32),
        ("grad_accum", 2),
        ("phase_weights", [1.0, 1.0]),
        ("timestep_cache", True),
    ):
        changed = SimpleNamespace(**(vars(args) | {name: value}))
        with pytest.raises(ValueError):
            trainer.validate_args(changed)
    for graph, rope in ((True, False), (False, True), (True, True)):
        changed = SimpleNamespace(
            **(vars(args) | dict(frozen_inference_graph=graph, fused_fp64_rope=rope))
        )
        trainer.validate_args(changed)
        with pytest.raises(ValueError, match="native_addmm"):
            trainer.validate_args(
                SimpleNamespace(**(vars(changed) | dict(patch_projection="native")))
            )
    trainer.validate_args(SimpleNamespace(**(vars(args) | dict(prefetch_frozen=True))))
    with pytest.raises(ValueError, match="preencode-next"):
        trainer.validate_args(
            SimpleNamespace(
                **(vars(args) | dict(prefetch_frozen=True, preencode_next=False))
            )
        )
