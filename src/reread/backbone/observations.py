import argparse
import math
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path

import torch

from reread.backbone.vae import VAE_CONTROLLER_ATTR
from reread.data.demonstrations import DEFAULT_RAW_ROOT, normalize_rgb_image
from reread.data.prefetch import AsyncEncodedBatches

DEFAULT_UPSTREAM = "third_party/fastwam"
DEFAULT_BASE = "artifacts/ckpt/fastwam_v0_3/robotwin_uncond_3cam_384.pt"
DEFAULT_STATS = (
    "artifacts/ckpt/fastwam_v0_3/robotwin_uncond_3cam_384_dataset_stats.json"
)
DEFAULT_WAN = "artifacts/ckpt/Wan2.2-TI2V-5B"


def official_configuration(upstream):
    upstream = Path(upstream).resolve()
    sys.path.insert(0, str(upstream / "src"))
    import fastwam

    if not Path(fastwam.__file__).resolve().is_relative_to(upstream):
        raise RuntimeError(
            "Different FastWAM package already imported; use a fresh process"
        )
    from fastwam.utils.config_resolvers import register_default_resolvers
    from hydra import compose, initialize_config_dir

    register_default_resolvers()
    with initialize_config_dir(
        config_dir=str(upstream / "configs"), version_base="1.3"
    ):
        return compose(
            config_name="train", overrides=["task=robotwin_uncond_3cam_384_1e-4"]
        )


def official_processor(cfg, stats):
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    from hydra.utils import instantiate

    processor = instantiate(cfg.data.train.processor)
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats)))
    return processor


def load_training_backbone(cfg, checkpoint, wan, device, *, cached_observations=False):
    from hydra.utils import instantiate

    if cached_observations:
        from reread.backbone.load import load_cached_training_backbone

        return load_cached_training_backbone(
            cfg.model, checkpoint, device, torch_dtype=torch.bfloat16
        )
    base = instantiate(
        cfg.model,
        model_id=str(Path(wan).resolve()),
        tokenizer_model_id=str(Path(wan).resolve()),
        redirect_common_files=False,
        skip_dit_load_from_pretrain=True,
        action_dit_pretrained_path=None,
        load_text_encoder=not cached_observations,
        model_dtype=torch.bfloat16,
        device=device,
    )
    payload = base.load_checkpoint(str(Path(checkpoint).resolve()))
    if "mot" not in payload or set(payload["mot"]) != set(base.mot.state_dict()):
        raise ValueError("Complete matching RGB and Action expert checkpoint required")
    if base.proprio_encoder is not None and "proprio_encoder" not in payload:
        raise ValueError("Trained frozen proprio encoder is missing")
    base.requires_grad_(False).eval()
    return base


@torch.no_grad()
def encode_current_batch(frozen, samples):
    if any("cached_encoded" in sample for sample in samples):
        if not all("cached_encoded" in sample for sample in samples):
            raise ValueError(
                "Raw and sealed encoded observations cannot be mixed in one batch"
            )
        from reread.data.transport import stack_and_transfer

        device = frozen.backbone.device
        result = {
            key: stack_and_transfer(
                [sample["cached_encoded"][key] for sample in samples], device
            )
            for key in ("rgb_latents", "context", "context_mask")
        }
        result["proprio"] = torch.stack([sample["proprio"] for sample in samples])
        return result
    cache = getattr(frozen, "encoding_cache", None)
    if cache is not None:
        return cache.encode(samples)
    base = frozen.backbone
    images = torch.stack([s["input_image"] for s in samples])
    proprio = torch.stack([s["proprio"] for s in samples])
    if images.ndim != 4 or images.shape[1:] != (3, 384, 320):
        raise ValueError("Only current official RGB montage [B,3,384,320] is allowed")
    device, dtype = base.device, base.torch_dtype
    images = normalize_rgb_image(images, device=device, dtype=dtype)
    controller = getattr(base.vae.model, VAE_CONTROLLER_ATTR, None)
    pixels = images.unsqueeze(2)
    if controller is not None:
        pixels = controller.prepare_input(pixels)

    latents = base.vae.model.encode(pixels, base.vae.scale)
    if latents.ndim != 5 or latents.shape[0] != len(samples) or latents.shape[2] != 1:
        raise ValueError("VAE changed current-frame encoding contract")
    context, context_mask = base.encode_prompt([s["prompt"] for s in samples])
    return {
        "rgb_latents": latents.detach(),
        "proprio": proprio,
        "context": context.detach(),
        "context_mask": context_mask.detach(),
    }


@torch.no_grad()
def extract_encoded_batch(frozen, encoded, *, seed=0):
    with ExitStack() as speed:
        if getattr(frozen, "unmasked_attention", False):
            from reread.backbone.attention import unmasked_frozen_attention

            speed.enter_context(unmasked_frozen_attention(frozen.backbone))
        if getattr(frozen, "timestep_cache", False):
            from reread.backbone.timesteps import cached_action_timesteps

            speed.enter_context(
                cached_action_timesteps(
                    frozen.backbone,
                    inference_steps=frozen.config.inference_steps,
                    sigma_shift=frozen.config.sigma_shift,
                )
            )
        route = {}
        if getattr(frozen, "uses_encoding_hit_dispatch", False):
            route["use_graph"] = getattr(encoded, "graph_eligible", False)
        actions, features = frozen.extract_batch(**encoded, seed=seed, **route)
    if actions.requires_grad or any(t.requires_grad for t in features.values()):
        raise RuntimeError("Frozen feature extraction retained a gradient graph")
    return actions, features


@torch.no_grad()
def extract_current_batch(frozen, samples, *, seed=0):
    return extract_encoded_batch(
        frozen, encode_current_batch(frozen, samples), seed=seed
    )


class _PreparedFirstSource:

    def __init__(self, first, source):
        self.first, self.source, self.closed = first, source, False

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration
        if self.first is not None:
            first, self.first = self.first, None
            return first
        return next(self.source)

    def close(self):
        if not self.closed:
            self.closed = True
            self.first = None
            self.source.close()


@contextmanager
def _feature_batches(source, frozen, *, extract, inference_seed, preencode_next):
    encoder = lambda samples: encode_current_batch(frozen, samples)
    prepare = getattr(frozen, "prepare_graph", None)
    if (
        extract is extract_current_batch
        and callable(prepare)
        and not frozen.graph_ready
    ):
        original = source
        try:
            first = next(original)
        except StopIteration:
            original.close()
            yield iter(())
            return
        try:
            first_encoded = encode_current_batch(frozen, first)
            prepare(**first_encoded, seed=inference_seed)
        except BaseException:
            try:
                original.close()
            except BaseException:
                pass
            raise

        source = _PreparedFirstSource(first, original)
        first_pending = True

        def graph_encoder(samples):
            nonlocal first_pending
            if first_pending and samples is first:
                first_pending = False
                return first_encoded
            return encode_current_batch(frozen, samples)

        encoder = graph_encoder

    if preencode_next and extract is extract_current_batch:
        try:
            prepared_batches = AsyncEncodedBatches(
                source, encoder, device=frozen.backbone.device
            )
        except BaseException:
            try:
                source.close()
            except BaseException:
                pass
            raise
        with prepared_batches as batches:
            yield (
                (samples, extract_encoded_batch(frozen, encoded, seed=inference_seed))
                for samples, encoded in batches
            )
    else:
        try:
            if extract is extract_current_batch:
                yield (
                    (
                        samples,
                        extract_encoded_batch(
                            frozen, encoder(samples), seed=inference_seed
                        ),
                    )
                    for samples in source
                )
            else:
                yield (
                    (samples, extract(frozen, samples, seed=inference_seed))
                    for samples in source
                )
        except BaseException:
            try:
                source.close()
            except BaseException:
                pass
            raise
        else:
            source.close()


def scheduled_lr(step, total_steps, warmup_steps, peak_lr, min_lr):
    if warmup_steps and step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps - 1))
    return min_lr + (peak_lr - min_lr) * (1 + math.cos(math.pi * progress)) / 2


def parser():
    p = argparse.ArgumentParser(
        description="Shared backbone loading, feature batches, and training options."
    )
    p.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    p.add_argument(
        "--cache-root",
        help="Sealed RoboTwin current-observation cache; bypass VAE/T5 computation",
    )
    p.add_argument("--tasks", nargs="+")
    p.add_argument(
        "--phases", nargs="+", choices=("clean", "randomized"), default=["clean"]
    )
    p.add_argument("--split", choices=("train", "val", "all"), default="train")
    p.add_argument("--val-episodes", type=int, default=1)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument(
        "--task-list",
        default=str(
            Path(__file__).resolve().parents[3] / "configs/robotwin_tasks.json"
        ),
        help="All 50 tasks, sampled uniformly by decision exposure",
    )
    p.add_argument(
        "--phase-weights",
        nargs="+",
        type=float,
        help="Decision exposure weights in --phases order; training uses clean:randomized = 1:10",
    )
    p.add_argument("--upstream", default=DEFAULT_UPSTREAM)
    p.add_argument("--checkpoint", default=DEFAULT_BASE)
    p.add_argument("--stats", default=DEFAULT_STATS)
    p.add_argument("--wan", default=DEFAULT_WAN)
    p.add_argument("--config")
    p.add_argument("--output")
    p.add_argument(
        "--preflight",
        action="store_true",
        help="CPU-only data/config check; never loads a GPU model",
    )
    p.add_argument(
        "--preflight-report",
        help="Save small CPU-only data/sampling proof, not image or latent caches",
    )
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--data-workers", type=int, default=4)
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument(
        "--preencode-next",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlap next current-frame VAE/T5 on one bounded CUDA stream",
    )
    p.add_argument(
        "--vae-encoding",
        choices=("native", "channels_last_3d", "single_frame_2d"),
        default="native",
    )
    p.add_argument(
        "--fused-visible-attention",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--timestep-cache", action=argparse.BooleanOptionalAction, default=False
    )
    p.add_argument(
        "--patch-projection",
        choices=("native", "native_addmm"),
        default="native_addmm",
        help="Preserve native per-image GEMM, with first-shape bitwise calibration",
    )
    p.add_argument(
        "--frozen-inference-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Prepare one native fixed-batch D20 CUDA graph before preprocessing starts",
    )
    p.add_argument(
        "--batched-bank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pack private training history once per batch, retaining all causal guards",
    )
    p.add_argument(
        "--rgb-cache-mib",
        type=int,
        default=2048,
        help="Bounded GPU-only RGB cache;0 disables",
    )
    p.add_argument(
        "--text-cache-mib",
        type=int,
        default=1024,
        help="Bounded GPU-only ordered T5 cache;0 disables",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--inference-seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="0 means save final corrector weights only",
    )
    p.add_argument("--swanlab-project")
    return p
