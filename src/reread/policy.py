from collections.abc import Mapping
from contextlib import ExitStack

import torch

from reread.backbone.attention import unmasked_frozen_attention
from reread.backbone.patch_projection import current_frame_addmm_projection
from reread.backbone.timesteps import cached_action_timesteps
from reread.backbone.vae import MODES
from reread.memory.bank import MemoryBank, _integer


def validate_runtime_options(runtime=None):
    defaults = {
        "vae_encoding": "native",
        "fused_visible_attention": False,
        "timestep_cache": False,
        "rgb_cache_mib": 0,
        "text_cache_mib": 0,
        "patch_projection": "native",
    }
    if runtime is None:
        return defaults
    if not isinstance(runtime, Mapping):
        raise ValueError("ReRead runtime must be a mapping")
    if set(runtime) - set(defaults):
        raise ValueError("Unknown ReRead runtime execution setting")
    result = {**defaults, **runtime}
    if (
        not isinstance(result["vae_encoding"], str)
        or result["vae_encoding"] not in MODES
    ):
        raise ValueError(f"Unknown VAE runtime mode: {result['vae_encoding']!r}")
    if not isinstance(result["patch_projection"], str) or result[
        "patch_projection"
    ] not in ("native", "native_addmm"):
        raise ValueError(
            f"Unknown patch projection runtime mode: {result['patch_projection']!r}"
        )
    for key in ("fused_visible_attention", "timestep_cache"):
        if not isinstance(result[key], bool):
            raise ValueError(f"ReRead runtime {key} must be boolean")
    for key in ("rgb_cache_mib", "text_cache_mib"):
        if (
            isinstance(result[key], bool)
            or not isinstance(result[key], int)
            or result[key] < 0
        ):
            raise ValueError(f"ReRead runtime {key} must be a nonnegative integer")
    return result


class TemporalMemoryPolicy:

    def __init__(
        self,
        frozen,
        corrector,
        *,
        bf16_corrector=False,
        runtime=None,
        memory_bank=None,
    ):
        if frozen.config != corrector.config:
            raise ValueError("Frozen feature and corrector contracts differ")
        self.frozen, self.corrector = frozen, corrector
        self.bf16_corrector = bool(bf16_corrector)
        self.runtime = validate_runtime_options(runtime)

        self._patch_validation_cache = set()
        self.bank = MemoryBank(corrector.config) if memory_bank is None else memory_bank
        if self.bank.config != corrector.config:
            raise ValueError("History bank must retain the trained model configuration")
        self.reset()

    def reset(self):
        self.bank.reset()
        self.episode_id = None
        self.decision_index = None
        self.observation_step = None

    @torch.no_grad()
    def act(self, *, episode_id, decision_index, observation_step, **observation):
        if episode_id is None:
            raise ValueError("episode_id is required")
        try:
            hash(episode_id)
        except TypeError as error:
            raise ValueError(
                "episode_id must be an immutable hashable identifier"
            ) from error
        decision_index = _integer(decision_index, "decision_index")
        observation_step = _integer(observation_step, "observation_step")
        if episode_id != self.episode_id:
            self.reset()
        if self.decision_index is None and decision_index != 0:
            raise ValueError("An episode must start at decision_index=0")
        if (
            self.decision_index is not None
            and decision_index != self.decision_index + 1
        ):
            raise ValueError(
                "Nonconsecutive decision; reset before starting a new sequence"
            )
        if self.observation_step is not None and not (
            self.observation_step
            < observation_step
            <= self.observation_step + self.corrector.config.horizon
        ):
            raise ValueError(
                "Observation step must advance by the actual executed plan prefix"
            )
        if hasattr(self.bank, "validate_call"):
            self.bank.validate_call(observation_step, decision_index)
        state = observation.get("proprio")
        if not isinstance(state, torch.Tensor):
            raise ValueError("Current normalized proprio tensor is required")
        if state.shape == (self.corrector.config.proprio_dim,):
            state = state.unsqueeze(0)
        if (
            state.shape != (1, self.corrector.config.proprio_dim)
            or not state.dtype.is_floating_point
            or not torch.isfinite(state).all()
        ):
            raise ValueError("Policy proprio must be finite normalized [1,proprio_dim]")
        state = state.detach().clone()
        observation["proprio"] = state

        with ExitStack() as execution:
            if self.runtime["patch_projection"] == "native_addmm":
                execution.enter_context(
                    current_frame_addmm_projection(
                        self.frozen.backbone,
                        validate=True,
                        validation_cache=self._patch_validation_cache,
                    )
                )
            if self.runtime["fused_visible_attention"]:
                execution.enter_context(unmasked_frozen_attention(self.frozen.backbone))
            if self.runtime["timestep_cache"]:
                execution.enter_context(
                    cached_action_timesteps(
                        self.frozen.backbone,
                        inference_steps=self.corrector.config.inference_steps,
                        sigma_shift=self.corrector.config.sigma_shift,
                    )
                )
            action, current = self.frozen.extract(**observation)
        expected = (1, self.corrector.config.horizon, self.corrector.config.action_dim)
        if (
            not isinstance(action, torch.Tensor)
            or action.shape != expected
            or not torch.isfinite(action).all()
        ):
            raise ValueError("Frozen backbone returned an invalid action plan")
        memory = self.bank.snapshot(current, state, observation_step, decision_index)
        self.corrector.eval()
        with torch.autocast(
            action.device.type, dtype=torch.bfloat16, enabled=self.bf16_corrector
        ):
            corrected = self.corrector(action, memory)
        if (
            not isinstance(corrected, torch.Tensor)
            or corrected.shape != expected
            or not torch.isfinite(corrected).all()
        ):
            raise ValueError(
                "Corrector returned an invalid action plan; memory was not committed"
            )

        result = {
            "action": corrected[0].detach().float().cpu(),
            "base_action": action[0].detach().float().cpu(),
        }
        self.bank.commit(current, state, observation_step, decision_index)
        self.episode_id = episode_id
        self.decision_index = decision_index
        self.observation_step = observation_step
        return result
