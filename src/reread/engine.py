import math
import time
from dataclasses import dataclass
from numbers import Integral

import torch

import reread.backbone.observations as observation_train
from reread.loss import action_mse
from reread.memory.training_bank import TrainingMemoryBanks
from reread.noise import perturb_actions


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative host integer")
    return int(value)


def _identities(samples, previous, config):
    if not isinstance(samples, (list, tuple)) or not samples:
        raise ValueError("Require a nonempty trajectory-lane batch")
    identities = dict(previous)
    seen = set()
    for sample in samples:
        required = (
            "lane",
            "episode_id",
            "trajectory_pass_id",
            "cache_segment",
            "cache_variant",
            "prompt_id",
            "frame",
            "decision_index",
        )
        if not isinstance(sample, dict) or any(name not in sample for name in required):
            raise ValueError(
                "Missing causal episode/pass/segment/variant/prompt identity"
            )
        lane = _integer(sample["lane"], "lane")
        if lane in seen:
            raise ValueError("Duplicate trajectory lane")
        seen.add(lane)
        episode, prompt = sample["episode_id"], sample["prompt_id"]
        if (
            not isinstance(episode, str)
            or not episode
            or not isinstance(prompt, str)
            or not prompt
        ):
            raise ValueError("Episode and prompt identities must be nonempty strings")
        identity = (
            episode,
            _integer(sample["trajectory_pass_id"], "trajectory_pass_id"),
            _integer(sample["cache_segment"], "cache_segment"),
            _integer(sample["cache_variant"], "cache_variant"),
            prompt,
            _integer(sample["frame"], "frame"),
            _integer(sample["decision_index"], "decision_index"),
        )
        if _integer(sample.get("raw_frame", identity[5]), "raw_frame") != identity[5]:
            raise ValueError("History timestamps require original raw frames")
        old = previous.get(lane)
        if old is not None and identity[:2] == old[:2]:
            if identity[2:5] != old[2:5]:
                raise ValueError(
                    "Segment/prompt/variant changed inside a pass; start a new pass"
                )
            if (
                identity[5] - old[5] != config.decision_stride
                or identity[6] != old[6] + 1
            ):
                raise ValueError(
                    "History requires consecutive decisions at exact raw decision stride"
                )
        elif identity[6] != 0:
            raise ValueError("A new episode/pass must begin at decision zero")
        identities[lane] = identity
    return identities


@dataclass(frozen=True)
class MemoryUpdate:
    owner: object
    generation: int
    bank: object
    identities: dict
    memory: dict


class CausalMemoryBanks:

    def __init__(self, config):
        self.config = config
        self._owner, self._generation = object(), 0
        self._bank = TrainingMemoryBanks(config)
        self._identities = {}

    def reset(self):
        self._bank = TrainingMemoryBanks(self.config)
        self._identities = {}
        self._generation += 1

    def records_for_lane(self, lane):
        return self._bank.records_for_lane(lane)

    def prepare(self, current, samples):
        if not isinstance(current, dict) or set(current) != {"k", "v"}:
            raise ValueError("Only frozen raw visual/plan k/v may enter history")
        identities = _identities(samples, self._identities, self.config)
        memory = self._bank.snapshot(current, samples)
        candidate = TrainingMemoryBanks(self.config)
        candidate._lanes = self._bank._lanes.copy()
        candidate.commit(current, samples)
        return MemoryUpdate(
            self._owner, self._generation, candidate, identities, memory
        )

    def publish(self, prepared):
        if (
            not isinstance(prepared, MemoryUpdate)
            or prepared.owner is not self._owner
            or prepared.generation != self._generation
        ):
            raise ValueError("Stale or foreign endpoint-history commit")
        self._bank, self._identities = prepared.bank, prepared.identities
        self._generation += 1


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


class _ClosableSource:

    def __init__(self, source):
        self.source = iter(source)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.source)

    def close(self):
        close = getattr(self.source, "close", None)
        if close is not None:
            close()


def stream_updates(
    frozen,
    model,
    optimizer,
    source,
    *,
    distributed=False,
    corrector_autocast=True,
    prefetch=True,
    inference_seed=0,
    on_step=None,
    on_samples=None,
    extract=None,
    prefetch_frozen=False,
):
    module = _unwrap(model)
    config = module.config
    if frozen.config != config:
        raise ValueError("Frozen endpoint extraction and model configuration differ")
    banks = CausalMemoryBanks(config)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    updates = decisions = 0
    extract = observation_train.extract_current_batch if extract is None else extract
    completed = time.monotonic()
    from reread.data.feature_prefetch import feature_batches

    with feature_batches(
        _ClosableSource(source),
        frozen,
        extract=extract,
        inference_seed=inference_seed,
        prefetch=prefetch,
        prefetch_frozen=prefetch_frozen,
    ) as batches:
        for samples, (base_actions, current) in batches:
            started = time.monotonic()
            staged = banks.prepare(current, samples)
            target = torch.stack([sample["action"] for sample in samples]).to(
                base_actions.device
            )
            valid = torch.stack([sample["valid"] for sample in samples]).to(
                base_actions.device
            )
            candidate, noise = perturb_actions(base_actions, valid, config)
            with torch.autocast(
                base_actions.device.type,
                dtype=torch.bfloat16,
                enabled=corrector_autocast,
            ):
                output = model(candidate, staged.memory, return_aux=True)
            prediction = output["actions"]
            loss, metrics = action_mse(
                prediction, target, valid, config, distributed=distributed
            )
            if not bool(torch.isfinite(loss.detach())) or not math.isfinite(
                metrics["total_loss"]
            ):
                raise FloatingPointError(
                    "Nonfinite endpoint loss; no history was committed"
                )
            forwarded = time.monotonic()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            with torch.no_grad():
                inner_gradient = module.action_proj.weight.grad
                inner_norm = (
                    inner_gradient.float().norm()
                    if inner_gradient is not None
                    else torch.zeros((), device=prediction.device)
                )
                output_gradient = module.output.weight.grad
                head_norm = (
                    output_gradient.float().norm()
                    if output_gradient is not None
                    else torch.zeros((), device=prediction.device)
                )
                residual = prediction.float() - candidate.float()
                mask = valid[..., None].expand_as(residual)
                residual_rms = residual.masked_select(mask).square().mean().sqrt()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            banks.publish(staged)
            updates += 1
            decisions += len(samples)
            if on_samples is not None:
                on_samples(samples)
            metrics.update(
                grad_norm=float(norm),
                action_projection_grad_norm=float(inner_norm),
                output_grad_norm=float(head_norm),
                residual_rms=float(residual_rms),
                near_history_fraction=float(
                    output["history_visible"][:, -2].float().mean()
                ),
                far_history_fraction=float(
                    output["history_visible"][:, 0].float().mean()
                ),
                perturb_fraction=float(noise["mask"].sum() / valid.sum().clamp_min(1)),
                perturb_offset_rms=float(noise["offset"].square().mean().sqrt()),
                base_forward_calls=config.inference_steps,
                corrector_forward_calls=1,
                corrector_forward_loss_wall_seconds=forwarded - started,
                backward_optimizer_wall_seconds=time.monotonic() - forwarded,
                local_decisions=decisions,
                local_step_wall_seconds=time.monotonic() - completed,
            )
            completed = time.monotonic()
            if on_step is not None and on_step(metrics):
                break
    if updates == 0:
        raise ValueError("No complete endpoint training batches consumed")
    return dict(updates=updates, local_decisions=decisions)


class PolicyRuntime:

    def __init__(self, frozen, model, *, autocast=True, seed=0):
        if frozen.config != model.config:
            raise ValueError(
                "Frozen endpoint extraction and model configuration differ"
            )
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("Inference seed must be a nonnegative integer")
        self.frozen, self.model = frozen, model
        self.autocast, self.seed = bool(autocast), seed
        self.banks = CausalMemoryBanks(model.config)

    def reset(self):
        self.banks.reset()

    @torch.no_grad()
    def act_encoded(self, samples, encoded):
        self.model.eval()

        _identities(samples, self.banks._identities, self.model.config)
        base, features = observation_train.extract_encoded_batch(
            self.frozen, encoded, seed=self.seed
        )
        staged = self.banks.prepare(features, samples)
        with torch.autocast(
            base.device.type, dtype=torch.bfloat16, enabled=self.autocast
        ):
            actions = self.model(base, staged.memory)
        if not bool(torch.isfinite(actions).all()):
            raise FloatingPointError(
                "Nonfinite endpoint output; history was not committed"
            )
        self.banks.publish(staged)
        return dict(
            actions=actions.detach().float(),
            base_actions=base.detach().float(),
            residual=actions.detach().float() - base.detach().float(),
            denoising_steps=self.model.config.inference_steps,
            corrector_forward_calls=1,
            image_prediction=False,
            artificial_perturbation=False,
        )
