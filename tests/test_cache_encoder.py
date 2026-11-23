import sys
from copy import deepcopy
from types import ModuleType, SimpleNamespace

import pytest
import torch

import reread.data.encoders as enc


class FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor(1.0, dtype=torch.bfloat16), requires_grad=False
        )
        self.calls = []
        self.nonfinite = False

    def encode(self, pixels, scale):
        self.calls.append((pixels.clone(), scale))
        values = pixels[:, 0, 0, 0, 0].reshape(-1, 1, 1, 1, 1)
        output = values.expand(-1, *enc.LATENT_SHAPE).clone()
        if self.nonfinite:
            output[0, 0, 0, 0, 0] = float("nan")
        return output


class FakeT5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor(1.0, dtype=torch.bfloat16), requires_grad=False
        )
        self.calls = []
        self.nonfinite = False

    def forward(self, ids, mask):
        self.calls.append((ids.clone(), mask.clone()))
        output = (
            ids[:, :1, None]
            .to(torch.bfloat16)
            .expand(-1, enc.TOKEN_LENGTH, enc.TEXT_DIM)
            .clone()
        )
        if self.nonfinite:
            output[0, 0, 0] = float("inf")
        return output


class FakeTokenizer:
    seq_len = 128
    clean = "whitespace"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def __call__(self, prompts, **kwargs):
        self.calls.append((tuple(prompts), kwargs))
        ids = (
            torch.arange(1, len(prompts) + 1)[:, None]
            .expand(-1, enc.TOKEN_LENGTH)
            .clone()
        )
        mask = torch.zeros_like(ids)
        mask[:, :7] = 1
        return ids, mask


def components():
    model = FakeVAE().eval()
    vae = torch.nn.Module()
    vae.model = model
    vae.scale = [1.0, 1.0]
    vae.eval()
    return vae, FakeT5().eval(), FakeTokenizer()


def encoder(vae_batch_size=None, text_batch_size=None):
    vae, text, tokenizer = components()
    identity = {"test": True}
    if vae_batch_size is not None:
        identity["execution"] = {"encode_batch": vae_batch_size}
    if text_batch_size is not None:
        identity.setdefault("execution", {})["text_encode_batch"] = text_batch_size
    return enc.OfflineEncoders(vae, text, tokenizer, device="cpu", identity=identity)


def plan_files(tmp_path):
    upstream, wan = tmp_path / "upstream", tmp_path / "wan"
    for name in enc._SOURCE:
        path = upstream / "src/fastwam/models/wan22" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    for name in enc._WEIGHTS:
        path = wan / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    for name in enc._TOKENIZER:
        path = wan / "google/umt5-xxl" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    return upstream, wan


def test_vectorized_text_padding_exactly_matches_native_length_slice():
    instance = encoder(text_batch_size=32)
    lengths = [0, 1, 7, 127, 128] + [19] * 27

    class VariedTokenizer:
        def __call__(self, prompts, **kwargs):
            ids = torch.arange(1, 33)[:, None].expand(-1, 128).clone()
            mask = torch.arange(128)[None, :] < torch.tensor(lengths)[:, None]

            mask[2] = mask[2].roll(3)
            return ids, mask

    instance.tokenizer = VariedTokenizer()
    context, attention = instance.encode_text(["fully templated prompt"] * 32)
    expected = (
        torch.arange(1, 33, dtype=torch.bfloat16)[:, None, None]
        .expand(32, 128, 4096)
        .clone()
    )
    for index, length in enumerate(lengths):
        expected[index, length:] = 0
    assert torch.equal(context, expected)
    assert attention.all() and attention.dtype == torch.bool


def test_identity_uses_128_native_b32_and_large_hashes_are_optional(tmp_path):
    upstream, wan = plan_files(tmp_path)
    identity = enc.build_encoder_identity(upstream, wan)
    assert identity["schema"] == enc.IDENTITY_SCHEMA
    assert identity["execution"]["token_length"] == 128
    assert identity["execution"]["encode_batch"] == 32
    assert identity["execution"]["vae"] == "native"
    assert "text_encode_batch" not in identity["execution"]
    assert "text_tail_padding" not in identity["execution"]
    assert all("sha256" not in info for info in identity["weights"].values())
    assert all(len(info["sha256"]) == 64 for info in identity["source"].values())
    assert all(len(info["sha256"]) == 64 for info in identity["tokenizer"].values())
    hashed = enc.build_encoder_identity(upstream, wan, sha256_weights=True)
    assert all(len(info["sha256"]) == 64 for info in hashed["weights"].values())
    assert enc._validate_shared_identity(hashed, upstream, wan) == hashed


@pytest.mark.parametrize("batch", [64, 128, 256])
def test_larger_vae_identity_keeps_t5_batch_32_and_shared_validation(tmp_path, batch):
    upstream, wan = plan_files(tmp_path)
    identity = enc.build_encoder_identity(
        upstream, wan, sha256_weights=True, vae_batch_size=batch
    )
    assert identity["execution"]["encode_batch"] == batch
    assert (
        identity["execution"]["tail_padding"] == f"duplicate_last_to_{batch}_then_slice"
    )
    assert identity["execution"]["text_encode_batch"] == 32
    assert (
        identity["execution"]["text_tail_padding"] == "duplicate_last_to_32_then_slice"
    )
    assert identity["execution"]["token_length"] == 128
    assert enc._validate_shared_identity(identity, upstream, wan) == identity
    invalid = deepcopy(identity)
    invalid["execution"]["text_encode_batch"] = batch
    with pytest.raises(ValueError, match="no longer match"):
        enc._validate_shared_identity(invalid, upstream, wan)


@pytest.mark.parametrize(
    "vae_batch,text_batch",
    [(32, 64), (32, 128), (32, 256), (64, 128), (128, 128), (256, 256)],
)
def test_text_batch_identity_independent_of_vae_and_validates_both(
    tmp_path, vae_batch, text_batch
):
    upstream, wan = plan_files(tmp_path)
    identity = enc.build_encoder_identity(
        upstream,
        wan,
        sha256_weights=True,
        vae_batch_size=vae_batch,
        text_batch_size=text_batch,
    )
    execution = identity["execution"]
    assert execution["encode_batch"] == vae_batch
    assert execution["tail_padding"] == f"duplicate_last_to_{vae_batch}_then_slice"
    assert execution["text_encode_batch"] == text_batch
    assert (
        execution["text_tail_padding"] == f"duplicate_last_to_{text_batch}_then_slice"
    )
    assert execution["token_length"] == 128
    assert enc._validate_shared_identity(identity, upstream, wan) == identity
    changed = deepcopy(identity)
    changed["execution"]["text_tail_padding"] = "duplicate_last_to_32_then_slice"
    with pytest.raises(ValueError, match="no longer match"):
        enc._validate_shared_identity(changed, upstream, wan)


def test_legacy_b32_and_v2_large_vae_t5_b32_identities_remain_exact(tmp_path):
    upstream, wan = plan_files(tmp_path)
    v1 = enc.build_encoder_identity(upstream, wan, vae_batch_size=32)
    assert "text_encode_batch" not in v1["execution"]
    assert "text_tail_padding" not in v1["execution"]
    assert (
        enc.build_encoder_identity(upstream, wan, vae_batch_size=32, text_batch_size=32)
        == v1
    )
    assert enc._validate_shared_identity(v1, upstream, wan) == v1
    v2 = enc.build_encoder_identity(upstream, wan, vae_batch_size=128)
    assert v2["execution"]["text_encode_batch"] == 32
    assert v2["execution"]["text_tail_padding"] == "duplicate_last_to_32_then_slice"
    assert (
        enc.build_encoder_identity(
            upstream, wan, vae_batch_size=128, text_batch_size=32
        )
        == v2
    )
    assert enc._validate_shared_identity(v2, upstream, wan) == v2


@pytest.mark.parametrize("batch", [0, 16, 33, 512, True, "128", None])
def test_invalid_planned_vae_batch_rejected_before_file_access(tmp_path, batch):
    with pytest.raises(ValueError, match="one of 32, 64, 128, 256"):
        enc.build_encoder_identity(
            tmp_path / "absent", tmp_path / "absent", vae_batch_size=batch
        )


@pytest.mark.parametrize("batch", [0, 16, 33, 512, True, "128", None])
def test_invalid_planned_t5_batch_rejected_before_file_access(tmp_path, batch):
    with pytest.raises(
        ValueError, match="Native offline T5 batch must be one of 32, 64, 128, 256"
    ):
        enc.build_encoder_identity(
            tmp_path / "absent", tmp_path / "absent", text_batch_size=batch
        )


def test_shared_identity_rejects_changed_files_and_settings(tmp_path):
    upstream, wan = plan_files(tmp_path)
    identity = enc.build_encoder_identity(upstream, wan)
    changed = deepcopy(identity)
    changed["execution"]["token_length"] = 512
    with pytest.raises(ValueError, match="no longer match"):
        enc._validate_shared_identity(changed, upstream, wan)
    (wan / enc._WEIGHTS[0]).write_bytes(b"changed weight content")
    with pytest.raises(ValueError, match="no longer match"):
        enc._validate_shared_identity(identity, upstream, wan)


def test_native_component_factory_only_constructs_vae_t5_and_local_tokenizer(
    tmp_path, monkeypatch
):
    upstream, wan = plan_files(tmp_path)
    fake_fastwam = ModuleType("fastwam")
    fake_fastwam.__file__ = str(upstream / "src/fastwam/__init__.py")
    calls = []
    vae, text, _ = components()

    def registered(path, name, *, torch_dtype, device):
        calls.append((path, name, torch_dtype, device))
        return vae if name == "wan_video_vae" else text

    loader = ModuleType("fastwam.models.wan22.helpers.loader")
    loader._load_registered_model = registered
    token_module = ModuleType("fastwam.models.wan22.wan_video_text_encoder")
    token_module.HuggingfaceTokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "fastwam", fake_fastwam)
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    monkeypatch.setitem(sys.modules, token_module.__name__, token_module)
    before = list(sys.path)
    try:
        result = enc._load_native_components(upstream, wan, "cpu")
    finally:
        sys.path[:] = before
    assert [entry[1] for entry in calls] == ["wan_video_vae", "wan_video_text_encoder"]
    assert all(entry[2:] == (torch.bfloat16, "cpu") for entry in calls)
    tokenizer = result[2]
    assert tokenizer.kwargs == {
        "name": str(wan / "google/umt5-xxl"),
        "seq_len": 128,
        "clean": "whitespace",
        "local_files_only": True,
    }
    assert all(not model.training for model in result[:2])


def test_load_checks_identity_and_cuda_then_uses_only_component_factory(
    tmp_path, monkeypatch
):
    upstream, wan = plan_files(tmp_path)
    identity = enc.build_encoder_identity(upstream, wan, sha256_weights=True)
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda device: calls.append(("device", str(device)))
    )
    monkeypatch.setattr(
        torch, "set_num_threads", lambda threads: calls.append(("threads", threads))
    )
    monkeypatch.setattr(
        enc, "_load_native_components", lambda upstream, wan, device: components()
    )
    result = enc.OfflineEncoders.load(
        upstream=upstream, wan=wan, device="cuda:0", cpu_threads=3, identity=identity
    )
    assert calls == [("threads", 3), ("device", "cuda:0")]
    assert result.identity() == identity
    changed = result.identity()
    changed["execution"]["vae"] = "changed"
    assert result.identity() == identity


@pytest.mark.parametrize("matching", [True, False])
def test_load_verifies_explicit_device_physical_uuid_before_loading_weights(
    tmp_path, monkeypatch, matching
):
    upstream, wan = plan_files(tmp_path)
    identity = enc.build_encoder_identity(upstream, wan, sha256_weights=True)
    expected = "GPU-abcdef00-1234-5678-90ab-000000000005"
    actual = (
        expected[4:].upper() if matching else "abcdef00-1234-5678-90ab-000000000007"
    )
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "set_num_threads", lambda threads: None)
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda device: calls.append(("device", str(device)))
    )

    def properties(device):
        calls.append(("properties", str(device)))
        return SimpleNamespace(uuid=actual)

    def native(upstream, wan, device):
        calls.append(("models", str(device)))
        return components()

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    monkeypatch.setattr(enc, "_load_native_components", native)
    options = dict(
        upstream=upstream,
        wan=wan,
        device="cuda:5",
        identity=identity,
        expected_gpu_uuid=expected,
    )
    if matching:
        result = enc.OfflineEncoders.load(**options)
        assert str(result.device) == "cuda:5"
        assert calls == [
            ("device", "cuda:5"),
            ("properties", "cuda:5"),
            ("models", "cuda:5"),
        ]
    else:
        with pytest.raises(RuntimeError, match="identity changed"):
            enc.OfflineEncoders.load(**options)
        assert calls == [("device", "cuda:5"), ("properties", "cuda:5")]


def test_load_derives_vae_batch_from_identity_without_a_second_configuration(
    tmp_path, monkeypatch
):
    upstream, wan = plan_files(tmp_path)
    identity = enc.build_encoder_identity(upstream, wan, vae_batch_size=128)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda threads: None)
    monkeypatch.setattr(
        enc, "_load_native_components", lambda upstream, wan, device: components()
    )
    result = enc.OfflineEncoders.load(
        upstream=upstream, wan=wan, device="cuda:0", identity=identity
    )
    assert result.vae_batch_size == 128


@pytest.mark.parametrize("vae_batch,text_batch", [(32, 128), (128, 128), (128, 256)])
def test_load_derives_both_native_batches_from_one_plan(
    tmp_path, monkeypatch, vae_batch, text_batch
):
    upstream, wan = plan_files(tmp_path)
    identity = enc.build_encoder_identity(
        upstream, wan, vae_batch_size=vae_batch, text_batch_size=text_batch
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda threads: None)
    monkeypatch.setattr(
        enc, "_load_native_components", lambda upstream, wan, device: components()
    )
    result = enc.OfflineEncoders.load(
        upstream=upstream, wan=wan, device="cuda:0", identity=identity
    )
    assert result.vae_batch_size == vae_batch and result.text_batch_size == text_batch
    assert result.identity() == identity


@pytest.mark.parametrize("batch", [1, 3, 32])
def test_rgb_fixed_b32_native_normalization_tail_padding_and_owned_output(batch):
    model = encoder()
    images = torch.arange(batch, dtype=torch.uint8)[:, None, None, None].expand(
        -1, *enc.RGB_SHAPE
    )
    images = images * 29
    result = model.encode_rgb(images)
    pixels, scale = model.vae.model.calls[-1]
    assert pixels.shape == (32, 3, 1, 384, 320)
    assert pixels.dtype == torch.bfloat16 and pixels.is_contiguous()
    expected = images.to(torch.bfloat16) * (2.0 / 255.0) - 1.0
    assert torch.equal(pixels[:batch, :, 0], expected)
    assert torch.equal(
        pixels[batch - 1 :], pixels[batch - 1 :].clone().fill_(expected[-1, 0, 0, 0])
    )
    assert result.shape == (batch, *enc.LATENT_SHAPE)
    assert (
        result.dtype == torch.bfloat16
        and result.device.type == "cpu"
        and result.is_contiguous()
    )
    assert result.requires_grad is False
    assert torch.equal(result[:, 0, 0, 0, 0], expected[:, 0, 0, 0])


@pytest.mark.parametrize("planned,count", [(64, 64), (64, 3), (128, 128), (128, 3)])
def test_rgb_larger_native_batches_pack_full_then_pad_only_tail(planned, count):
    model = encoder(planned)
    images = torch.arange(count, dtype=torch.uint8)[:, None, None, None].expand(
        -1, *enc.RGB_SHAPE
    )
    result = model.encode_rgb(images)
    pixels, _ = model.vae.model.calls[-1]
    assert pixels.shape == (planned, 3, 1, 384, 320)
    expected = images[:, 0, 0, 0].to(torch.bfloat16) * (2.0 / 255.0) - 1.0
    assert torch.equal(pixels[:count, 0, 0, 0, 0], expected)
    assert (pixels[count - 1 :, 0, 0, 0, 0] == expected[-1]).all()
    assert result.shape == (count, *enc.LATENT_SHAPE)
    assert result.is_contiguous() and result.dtype == torch.bfloat16
    assert torch.equal(result[:, 0, 0, 0, 0], expected)
    with pytest.raises(ValueError, match=f"1 <= B <= {planned}"):
        model.encode_rgb(torch.zeros(planned + 1, *enc.RGB_SHAPE, dtype=torch.uint8))


def test_larger_vae_configuration_does_not_change_text_packing():
    model = encoder(128)
    context, mask = model.encode_text(["full prompt"])
    assert len(model.tokenizer.calls[-1][0]) == 32
    assert model.text_encoder.calls[-1][0].shape == (32, 128)
    assert context.shape == (1, 128, 4096) and mask.shape == (1, 128) and mask.all()
    with pytest.raises(ValueError, match="1..32"):
        model.encode_text(["full prompt"] * 33)


@pytest.mark.parametrize(
    "planned,count", [(64, 64), (64, 3), (128, 128), (128, 3), (256, 256), (256, 3)]
)
def test_actual_larger_t5_batches_keep_128_tokens_all_true_mask_and_pad_tail(
    planned, count
):
    model = encoder(32, planned)
    prompts = [f"full prompt {index}" for index in range(count)]
    result, mask = model.encode_text(prompts)
    token_prompts = model.tokenizer.calls[-1][0]
    assert len(token_prompts) == planned
    assert token_prompts[:count] == tuple(prompts)
    assert all(prompt == prompts[-1] for prompt in token_prompts[count:])
    ids, original_mask = model.text_encoder.calls[-1]
    assert ids.shape == original_mask.shape == (planned, 128)
    assert result.shape == (count, 128, 4096) and mask.shape == (count, 128)
    assert (
        result.dtype == torch.bfloat16
        and result.device.type == "cpu"
        and result.is_contiguous()
    )
    assert mask.dtype == torch.bool and mask.device.type == "cpu" and mask.all()
    assert torch.equal(result[:, 0, 0], torch.arange(1, count + 1).to(torch.bfloat16))
    assert not result[:, 7:].count_nonzero()
    assert torch.isfinite(result).all() and not result.requires_grad
    with pytest.raises(ValueError, match=f"1..{planned}"):
        model.encode_text(["full prompt"] * (planned + 1))


def test_large_t5_batch_rejects_nonfinite_output_without_silent_fallback():
    model = encoder(128, 128)
    model.text_encoder.nonfinite = True
    with pytest.raises(ValueError, match="nonfinite text context"):
        model.encode_text(["full prompt"] * 64)
    assert len(model.tokenizer.calls) == len(model.text_encoder.calls) == 1
    assert model.text_encoder.calls[-1][0].shape == (128, 128)


def test_text_preserves_fastwam_zero_padding_visible_mask_and_fixed_b32():
    model = encoder()
    prompts = [
        enc.format_instruction("adjust bottle"),
        enc.format_instruction("another unseen instruction"),
    ]
    result, mask = model.encode_text(prompts)
    sent, options = model.tokenizer.calls[-1]
    assert len(sent) == 32 and sent[:2] == tuple(prompts) and sent[-1] == prompts[-1]
    assert options == {"return_mask": True, "add_special_tokens": True}
    assert result.shape == (2, 128, 4096) and result.dtype == torch.bfloat16
    assert result.device.type == mask.device.type == "cpu"
    assert result.is_contiguous() and mask.is_contiguous() and mask.dtype == torch.bool
    assert mask.all()
    assert result[:, :7].count_nonzero() == 2 * 7 * 4096
    assert not result[:, 7:].count_nonzero()
    assert model.text_encoder.calls[-1][0].shape == (32, 128)


@pytest.mark.parametrize(
    "images",
    [
        torch.zeros(0, 3, 384, 320, dtype=torch.uint8),
        torch.zeros(33, 3, 384, 320, dtype=torch.uint8),
        torch.zeros(2, 3, 384, 320),
        torch.zeros(2, 3, 320, 384, dtype=torch.uint8),
    ],
)
def test_bad_rgb_contract_fails_before_native_call(images):
    model = encoder()
    with pytest.raises(ValueError, match="RGB must"):
        model.encode_rgb(images)
    assert not model.vae.model.calls


@pytest.mark.parametrize("prompts", [[], [""] * 2, ["a"] * 33, "one prompt", [None]])
def test_bad_text_contract_fails_before_native_call(prompts):
    model = encoder()
    with pytest.raises(ValueError, match="1..32"):
        model.encode_text(prompts)
    assert not model.tokenizer.calls


@pytest.mark.parametrize("kind", ["vae", "t5"])
def test_nonfinite_encoder_output_never_cached(kind):
    model = encoder()
    if kind == "vae":
        model.vae.model.nonfinite = True
        operation = lambda: model.encode_rgb(
            torch.zeros(1, *enc.RGB_SHAPE, dtype=torch.uint8)
        )
    else:
        model.text_encoder.nonfinite = True
        operation = lambda: model.encode_text(["one full prompt"])
    with pytest.raises(ValueError, match="nonfinite"):
        operation()


def test_encoder_training_or_wrong_token_length_rejected():
    model = encoder()
    model.vae.model.train()
    with pytest.raises(ValueError, match="eval mode"):
        model.encode_rgb(torch.zeros(1, *enc.RGB_SHAPE, dtype=torch.uint8))
    vae, text, token = components()
    token.seq_len = 512
    with pytest.raises(ValueError, match="128 tokens"):
        enc.OfflineEncoders(vae, text, token, device="cpu", identity={})


def test_prompt_template_matches_training_reader():
    from reread.data.demonstrations import DEFAULT_PROMPT

    assert enc.PROMPT_TEMPLATE == DEFAULT_PROMPT
    assert enc.format_instruction("unseen task") == DEFAULT_PROMPT.format(
        task="unseen task"
    )
    with pytest.raises(ValueError, match="nonempty"):
        enc.format_instruction(" ")


def test_release_text_leaves_rgb_valid_and_forbids_future_text_encoding():
    import weakref

    model = encoder()
    reference = weakref.ref(model.text_encoder)
    model.release_text()
    assert reference() is None
    assert model.text_encoder is model.tokenizer is None
    result = model.encode_rgb(torch.zeros(1, *enc.RGB_SHAPE, dtype=torch.uint8))
    assert result.shape == (1, *enc.LATENT_SHAPE)
    with pytest.raises(ValueError, match="T5 encoder was released"):
        model.encode_text(["full prompt"])
    model.release_text()
