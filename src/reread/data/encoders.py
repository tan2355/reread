import hashlib
import sys
from copy import deepcopy
from pathlib import Path

from reread.data.cache_contract import ENCODER_IDENTITY_SCHEMA as IDENTITY_SCHEMA
from reread.data.cache_contract import matches_encoder_source

ENCODE_BATCH = 32
VAE_BATCH_SIZES = (32, 64, 128, 256)
TEXT_BATCH_SIZES = (32, 64, 128, 256)
TOKEN_LENGTH = 128
TEXT_DIM = 4096
RGB_SHAPE = (3, 384, 320)
LATENT_SHAPE = (48, 1, 24, 20)
PROMPT_TEMPLATE = "A video recorded from a robot's point of view executing the following instruction: {task}"
_WEIGHTS = ("Wan2.2_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth")
_TOKENIZER = (
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
)
_SOURCE = (
    "wan_video_vae.py",
    "wan_video_text_encoder.py",
    "helpers/loader.py",
    "helpers/io.py",
    "helpers/state_dict_converters.py",
)


def format_instruction(instruction):
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Instruction must be a nonempty string")
    return PROMPT_TEMPLATE.format(task=instruction)


def _file_identity(path, *, include_sha):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Encoder identity needs a regular file: {path}")
    before = path.stat()
    result = {
        "path": str(path),
        "bytes": before.st_size,
        "mtime_ns": before.st_mtime_ns,
    }
    if include_sha:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Encoder file changed during identity read: {path}")
    return result


def _vae_batch_size(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in VAE_BATCH_SIZES
    ):
        raise ValueError("Native offline VAE batch must be one of 32, 64, 128, 256")
    return value


def _text_batch_size(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in TEXT_BATCH_SIZES
    ):
        raise ValueError("Native offline T5 batch must be one of 32, 64, 128, 256")
    return value


def build_encoder_identity(
    upstream, wan, *, sha256_weights=False, vae_batch_size=32, text_batch_size=32
):
    vae_batch_size = _vae_batch_size(vae_batch_size)
    text_batch_size = _text_batch_size(text_batch_size)
    upstream, wan = Path(upstream).resolve(strict=True), Path(wan).resolve(strict=True)
    source = upstream / "src/fastwam/models/wan22"
    result = {
        "schema": IDENTITY_SCHEMA,
        "upstream": str(upstream),
        "wan": str(wan),
        "weights": {
            name: _file_identity(wan / name, include_sha=sha256_weights)
            for name in _WEIGHTS
        },
        "tokenizer": {
            name: _file_identity(wan / "google/umt5-xxl" / name, include_sha=True)
            for name in _TOKENIZER
        },
        "source": {
            name: _file_identity(source / name, include_sha=True) for name in _SOURCE
        },
        "execution": {
            "dtype": "bfloat16",
            "vae": "native",
            "encode_batch": vae_batch_size,
            "tail_padding": f"duplicate_last_to_{vae_batch_size}_then_slice",
            "rgb_shape": list(RGB_SHAPE),
            "latent_shape": list(LATENT_SHAPE),
            "rgb_normalization": "uint8_to_bfloat16_then_multiply_2_over_255_subtract_1",
            "token_length": TOKEN_LENGTH,
            "text_dim": TEXT_DIM,
            "tokenizer_clean": "whitespace",
            "add_special_tokens": True,
            "text_padding": "zero_context_after_tokenizer_length_then_all_true_mask",
            "prompt_template": PROMPT_TEMPLATE,
        },
    }
    if vae_batch_size != ENCODE_BATCH or text_batch_size != ENCODE_BATCH:
        result["execution"].update(
            text_encode_batch=text_batch_size,
            text_tail_padding=f"duplicate_last_to_{text_batch_size}_then_slice",
        )
    return result


def _validate_shared_identity(identity, upstream, wan):
    if not isinstance(identity, dict) or identity.get("schema") != IDENTITY_SCHEMA:
        raise ValueError("Invalid shared offline encoder identity")
    execution = identity.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("Invalid shared offline encoder execution identity")
    current = build_encoder_identity(
        upstream,
        wan,
        sha256_weights=False,
        vae_batch_size=execution.get("encode_batch"),
        text_batch_size=execution.get("text_encode_batch", ENCODE_BATCH),
    )
    expected = deepcopy(identity)
    for entry in expected.get("weights", {}).values():
        entry.pop("sha256", None)
    recorded_sources = expected.get("source")
    if isinstance(recorded_sources, dict):
        for name, actual in current["source"].items():
            recorded = recorded_sources.get(name)
            if (
                isinstance(recorded, dict)
                and recorded.get("sha256") != actual["sha256"]
                and type(recorded.get("mtime_ns")) is int
                and recorded["mtime_ns"] > 0
                and matches_encoder_source(recorded, actual["bytes"], actual["sha256"])
            ):
                for key in ("bytes", "sha256", "mtime_ns"):
                    recorded[key] = actual[key]
    if expected != current:
        raise ValueError("Encoder files/settings no longer match the shared cache plan")
    return deepcopy(identity)


def _load_native_components(upstream, wan, device):
    import torch

    upstream, wan = Path(upstream).resolve(strict=True), Path(wan).resolve(strict=True)
    sys.path.insert(0, str(upstream / "src"))
    import fastwam

    if not Path(fastwam.__file__).resolve().is_relative_to(upstream):
        raise RuntimeError(
            "Another FastWAM package is already imported; use a fresh process"
        )
    from fastwam.models.wan22.helpers.loader import _load_registered_model
    from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer

    vae = _load_registered_model(
        str(wan / _WEIGHTS[0]),
        "wan_video_vae",
        torch_dtype=torch.bfloat16,
        device=device,
    )
    text = _load_registered_model(
        str(wan / _WEIGHTS[1]),
        "wan_video_text_encoder",
        torch_dtype=torch.bfloat16,
        device=device,
    )
    tokenizer = HuggingfaceTokenizer(
        name=str(wan / "google/umt5-xxl"),
        seq_len=TOKEN_LENGTH,
        clean="whitespace",
        local_files_only=True,
    )
    vae.eval().requires_grad_(False)
    text.eval().requires_grad_(False)
    return vae, text, tokenizer


class OfflineEncoders:

    def __init__(self, vae, text_encoder, tokenizer, *, device, identity):
        self.vae, self.text_encoder, self.tokenizer = vae, text_encoder, tokenizer
        self.device, self._identity = device, deepcopy(identity)
        execution = identity.get("execution", {})
        if not isinstance(execution, dict):
            raise ValueError("Offline encoder execution identity must be a mapping")
        self.vae_batch_size = _vae_batch_size(
            execution.get("encode_batch", ENCODE_BATCH)
        )
        self.text_batch_size = _text_batch_size(
            execution.get("text_encode_batch", ENCODE_BATCH)
        )
        if getattr(tokenizer, "seq_len", None) != TOKEN_LENGTH:
            raise ValueError(
                "FastWAM offline text encoding requires exactly 128 tokens"
            )
        if getattr(tokenizer, "clean", None) != "whitespace":
            raise ValueError("FastWAM offline tokenizer must use whitespace cleaning")
        self._require_frozen()

    @classmethod
    def load(
        cls,
        *,
        upstream,
        wan,
        device,
        cpu_threads=8,
        identity=None,
        expected_gpu_uuid=None,
    ):
        if (
            isinstance(cpu_threads, bool)
            or not isinstance(cpu_threads, int)
            or cpu_threads < 1
        ):
            raise ValueError("cpu_threads must be a positive integer")
        planned = (
            build_encoder_identity(upstream, wan)
            if identity is None
            else _validate_shared_identity(identity, upstream, wan)
        )
        import torch

        chosen = torch.device(device)
        if chosen.type != "cuda" or not torch.cuda.is_available():
            raise ValueError(
                "Offline encoder production factory requires a CUDA device"
            )
        torch.set_num_threads(cpu_threads)
        torch.cuda.set_device(chosen)
        if expected_gpu_uuid is not None:
            from reread.backbone.device import canonical_cuda_uuid

            actual = canonical_cuda_uuid(torch.cuda.get_device_properties(chosen).uuid)
            if actual != canonical_cuda_uuid(expected_gpu_uuid):
                raise RuntimeError(
                    f"CUDA device identity changed: {chosen} is {actual}, expected {expected_gpu_uuid}; encoder weights were not loaded"
                )
        vae, text, tokenizer = _load_native_components(upstream, wan, chosen)
        return cls(vae, text, tokenizer, device=chosen, identity=planned)

    def identity(self):
        return deepcopy(self._identity)

    def release_text(self):
        self.text_encoder, self.tokenizer = None, None

    def _require_frozen(self, *, only=None):
        models = [("VAE", self.vae), ("T5", self.text_encoder)]
        for name, model in models:
            if only is not None and name != only:
                continue
            if model is None:
                raise ValueError(f"{name} encoder was released")
            if any(module.training for module in model.modules()):
                raise ValueError(f"{name} must remain in eval mode")
            if any(parameter.requires_grad for parameter in model.parameters()):
                raise ValueError(f"{name} must remain frozen")

    def encode_rgb(self, images):
        import torch

        if (
            not isinstance(images, torch.Tensor)
            or images.device.type != "cpu"
            or images.dtype != torch.uint8
            or images.ndim != 4
            or tuple(images.shape[1:]) != RGB_SHAPE
            or not 1 <= len(images) <= self.vae_batch_size
        ):
            raise ValueError(
                f"RGB must be CPU uint8 [B,3,384,320], with 1 <= B <= {self.vae_batch_size}"
            )
        self._require_frozen(only="VAE")
        count = len(images)
        if count < self.vae_batch_size:
            images = torch.cat(
                (images, images[-1:].expand(self.vae_batch_size - count, -1, -1, -1))
            )
        with torch.no_grad():
            pixels = images.contiguous().to(device=self.device, dtype=torch.bfloat16)

            pixels = (pixels * (2.0 / 255.0) - 1.0).unsqueeze(2)
            latents = self.vae.model.encode(pixels, self.vae.scale)
            if (
                not isinstance(latents, torch.Tensor)
                or tuple(latents.shape) != (self.vae_batch_size, *LATENT_SHAPE)
                or latents.dtype != torch.bfloat16
            ):
                raise ValueError(
                    "Native VAE output changed its fixed BF16 current-frame contract"
                )
            result = latents[:count].detach().to(device="cpu").contiguous().clone()
            if not torch.isfinite(result).all():
                raise ValueError("Native VAE returned nonfinite current-frame latents")
            return result

    def encode_text(self, prompts):
        import torch

        if (
            not isinstance(prompts, (list, tuple))
            or not 1 <= len(prompts) <= self.text_batch_size
            or any(not isinstance(p, str) or not p.strip() for p in prompts)
        ):
            raise ValueError(
                f"Text requires 1..{self.text_batch_size} nonempty fully formatted prompts"
            )
        self._require_frozen(only="T5")
        count = len(prompts)
        padded = [*prompts, *([prompts[-1]] * (self.text_batch_size - count))]
        with torch.no_grad():
            ids, mask = self.tokenizer(
                padded, return_mask=True, add_special_tokens=True
            )
            if tuple(ids.shape) != (self.text_batch_size, TOKEN_LENGTH) or tuple(
                mask.shape
            ) != tuple(ids.shape):
                raise ValueError(
                    f"Tokenizer changed the fixed [{self.text_batch_size},128] contract"
                )
            ids = ids.to(self.device)
            mask = mask.to(self.device, dtype=torch.bool)
            context = self.text_encoder(ids, mask)
            if (
                not isinstance(context, torch.Tensor)
                or tuple(context.shape)
                != (self.text_batch_size, TOKEN_LENGTH, TEXT_DIM)
                or context.dtype != torch.bfloat16
            ):
                raise ValueError(
                    "Native T5 output changed its fixed BF16 text contract"
                )

            lengths = mask.gt(0).sum(dim=1)
            padding = (
                torch.arange(TOKEN_LENGTH, device=mask.device)[None, :]
                >= lengths[:, None]
            )
            context.masked_fill_(padding[:, :, None], 0)
            result = context[:count].detach().to(device="cpu").contiguous().clone()
            context_mask = torch.ones_like(mask[:count], device="cpu").contiguous()
            if not torch.isfinite(result).all():
                raise ValueError("Native T5 returned nonfinite text context")
            return result, context_mask
