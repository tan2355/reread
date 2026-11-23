import torch

from reread.config import joint_dimensions


def perturb_actions(actions, valid, config, *, generator=None, noise=None, mask=None):
    if actions.ndim != 3 or actions.shape[1:] != (config.horizon, config.action_dim):
        raise ValueError("Actions must be [B,horizon,action_dim]")
    if valid.shape != actions.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("valid must be boolean [B,H]")
    valid = valid.to(device=actions.device)
    candidate = actions.detach().to(dtype=torch.float32)
    if not bool(torch.isfinite(candidate).all()):
        raise FloatingPointError("Complete native candidate actions must be finite")
    enabled = bool(config.action_perturb_probability and config.action_perturb_std)
    if mask is None:
        selected = (
            (
                torch.rand(
                    actions.shape[:2], device=actions.device, generator=generator
                )
                < config.action_perturb_probability
            )
            if enabled
            else torch.zeros_like(valid)
        )
    else:
        if mask.shape != valid.shape or mask.dtype != torch.bool:
            raise ValueError("Perturbation mask must be boolean [B,H]")
        selected = (
            mask.to(device=actions.device).clone()
            if enabled
            else torch.zeros_like(valid)
        )
    selected &= valid
    if noise is not None and noise.shape != actions.shape:
        raise ValueError("Perturbation noise must match actions")
    if enabled:
        gaussian = (
            torch.randn(
                actions.shape,
                device=actions.device,
                dtype=torch.float32,
                generator=generator,
            )
            if noise is None
            else noise.detach().to(device=actions.device, dtype=torch.float32)
        )
        active = selected[..., None].expand_as(candidate).clone()
        keep = torch.zeros(config.action_dim, device=actions.device, dtype=torch.bool)
        keep[list(joint_dimensions(config))] = True
        active &= keep
        gaussian = gaussian.masked_fill(~active, 0)
        if not bool(torch.isfinite(gaussian).all()):
            raise FloatingPointError("Active action perturbation must be finite")
        offset = gaussian * config.action_perturb_std
    else:
        offset = torch.zeros_like(candidate)

    corrupted = torch.where(offset != 0, candidate + offset, candidate)
    return corrupted, {"mask": selected, "offset": offset}
