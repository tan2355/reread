import sys as _site_sys
from pathlib import Path as _SitePath

_site_sys.path.insert(0, str(_SitePath(__file__).resolve().parents[1] / "src"))
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INFERENCE_SETTINGS = dict(
    vae_encoding="native",
    patch_projection="native_addmm",
    fused_visible_attention=False,
    timestep_cache=False,
    rgb_cache_mib=0,
    text_cache_mib=0,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cpu_plan(budget=123, reserve=3, *, affinity=None, quota=None, local_workers=16):
    if type(local_workers) is not int or local_workers < 1:
        raise ValueError("Positive integer local worker count required")
    available = sorted(affinity if affinity is not None else os.sched_getaffinity(0))
    if quota is None:
        limits = []
        v2 = Path("/sys/fs/cgroup/cpu.max")
        if v2.is_file():
            amount, period = v2.read_text().split()
            if amount != "max":
                limits.append(math.floor(int(amount) / int(period)))
        for location in ("/sys/fs/cgroup/cpu", "/sys/fs/cgroup/cpu,cpuacct"):
            directory = Path(location)
            if (directory / "cpu.cfs_quota_us").is_file():
                amount = int((directory / "cpu.cfs_quota_us").read_text())
                period = int((directory / "cpu.cfs_period_us").read_text())
                if amount > 0:
                    limits.append(math.floor(amount / period))
        quota = min(limits) if limits else len(available)
    effective = min(len(available), int(quota), budget)
    if reserve < 1 or effective - reserve < local_workers:
        raise ValueError(
            "Each node needs CPU quota for its workers plus coordinator reserve"
        )
    threads, extra = divmod(effective - reserve, local_workers)
    worker_threads = [threads + int(i < extra) for i in range(local_workers)]
    affinities, cursor = [], 0
    for count in worker_threads:
        affinities.append(available[cursor : cursor + count])
        cursor += count
    return dict(
        effective_cpus=effective,
        cpu_threads_per_worker=threads,
        worker_threads=worker_threads,
        worker_affinity=affinities,
        coordinator_affinity=available[cursor:effective],
        reserved_cpus=effective - cursor,
        interop_threads=1,
        openblas_threads=1,
        video_threads=1,
    )


def verify_release(root, expected=None):
    root = Path(root).resolve(strict=True)
    digest = sha256(root / "manifest.json")
    if expected and digest != expected:
        raise ValueError("Evaluation release manifest SHA mismatch")
    manifest = json.loads((root / "manifest.json").read_text())
    known = set()
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        path = root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or path.is_symlink()
            or not path.resolve().is_relative_to(root)
            or entry["path"] in known
        ):
            raise ValueError("Unsafe/duplicate release path")
        known.add(entry["path"])
        if path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
            raise ValueError("Evaluation source changed: " + entry["path"])
    required = {
        "scripts/evaluate.py",
        "scripts/eval_utils.py",
        "src/reread/evaluation/speed_summary.py",
        "src/reread/evaluation/shared_queue.py",
        "src/reread/evaluation/worker.py",
        "src/reread/evaluation/runtime_setup.py",
        "src/reread/evaluation/scheduler.py",
        "src/reread/checkpoint.py",
        "src/reread/model.py",
        "src/reread/config.py",
        "src/reread/policy.py",
        "third_party/RoboTwin/script/eval_protocol.py",
        "third_party/RoboTwin/script/eval_policy.py",
        "experiments/robotwin/reread_policy/deploy_policy.py",
        "experiments/robotwin/reread_policy/prompt_cache.py",
        "experiments/robotwin/reread_policy/file_certificate.py",
        "experiments/robotwin/reread_policy/live_graph.py",
        "experiments/robotwin/reread_policy/color_contract.py",
    }
    if not required <= known:
        raise ValueError("Incomplete evaluation release")
    return digest


def validate_identity(payload):
    from reread.config import ReReadConfig

    if (
        payload.get("schema") != "newm-endpoint-memory3-weights-v1"
        or type(payload.get("step")) is not int
        or payload["step"] != 20000
        or ReReadConfig.from_dict(payload["config"])
        != ReReadConfig()
        or payload.get("frozen_backbone_gradients") is not False
    ):
        raise ValueError(
            "Requires the endpoint-v1 Memory3 20K checkpoint with D20 and correction from the first decision"
        )
    provenance, sampling, execution = (
        payload["provenance"],
        payload["sampling"],
        payload["execution"],
    )
    for key in ("source_manifest_sha256", "observation_cache_manifest_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(provenance.get(key, ""))):
            raise ValueError("Missing training source/cache SHA256 provenance")
    required_sampling = dict(
        observation_color_contract="native_robotwin_rgb_v1",
        kernel="native_addmm",
        correction_start_decision=0,
        supervised_start_decision=0,
        prefix_warmup_decisions=0,
        perturb_kv=False,
        encoders_loaded=False,
        steps=20000,
        batch_per_rank=64,
        world_size=16,
        global_batch=1024,
    )
    if any(
        type(sampling.get(k)) is not type(v) or sampling.get(k) != v
        for k, v in required_sampling.items()
    ):
        raise ValueError(
            "Checkpoint RGB / start0 / training-topology contract mismatch"
        )

    if (
        "corrector_autocast" in sampling
        and sampling["corrector_autocast"] is not True
        or "runtime" in sampling
        and sampling["runtime"] != INFERENCE_SETTINGS
    ):
        raise ValueError("Conflicting explicit precision/runtime metadata")
    required = dict(
        action_horizon=32,
        replan_steps=20,
        num_inference_steps=20,
        perturb_kv=False,
        perturbation_inference=False,
        inference_filter=False,
        output_clipping=False,
        image_prediction=False,
    )
    if any(
        type(execution.get(k)) is not type(v) or execution.get(k) != v
        for k, v in required.items()
    ):
        raise ValueError(
            "Checkpoint execution settings differ from start0 D20 endpoint inference"
        )


def checkpoint_preflight(args):
    path = Path(args.corrector)
    complete_path = path.parent / "complete.json"
    if (
        path.name != "endpoint_step_20000.pt"
        or not path.is_file()
        or not complete_path.is_file()
    ):
        raise ValueError(
            "Training not finalized: endpoint_step_20000.pt and complete.json are required; no evaluation started"
        )
    if list(path.parent.glob("failed*.json")):
        raise ValueError("Training failure marker exists; inspect it before evaluating")
    sys.path.insert(0, str(ROOT / "experiments/robotwin"))
    from reread_policy.file_certificate import (
        file_signature,
        issue_corrector_certificate,
        issue_file_certificate,
    )

    corrector_before = file_signature(path)
    complete = json.loads(complete_path.read_text())
    digest = sha256(path)
    if (
        complete.get("complete") is not True
        or complete.get("step") != 20000
        or complete.get("checkpoint") != path.name
        or complete.get("checkpoint_sha256") != digest
        or complete.get("checkpoint_bytes") != path.stat().st_size
        or complete.get("frozen_backbone_gradients") is not False
    ):
        raise ValueError("Training completion marker / final weight identity mismatch")
    if args.expected_checkpoint_sha and digest != args.expected_checkpoint_sha:
        raise ValueError("Final checkpoint differs from requested SHA")
    import torch

    from reread.checkpoint import load_checkpoint

    before = dict(
        base=file_signature(args.checkpoint), stats=file_signature(args.stats)
    )
    torch.set_num_threads(2)
    payload = load_checkpoint(path, base_path=args.checkpoint, stats_path=args.stats)
    corrector_certificate = issue_corrector_certificate(
        corrector_before, file_signature(path), digest
    )
    after = dict(base=file_signature(args.checkpoint), stats=file_signature(args.stats))
    certificate = issue_file_certificate(before, after, payload["provenance"])
    validate_identity(payload)
    params = sum(t.numel() for t in payload["model"].values())
    if params != 198_523_406:
        raise ValueError("Expected exactly 198523406 corrector parameters")
    sys.path.insert(0, str(ROOT / "experiments/robotwin"))
    adapter = importlib.import_module("reread_policy.deploy_policy")
    if (
        Path(adapter.__file__).resolve()
        != ROOT / "experiments/robotwin/reread_policy/deploy_policy.py"
    ):
        raise ValueError("Wrong deployment adapter imported")
    adapter.validate_rollout_args(policy_settings(args, payload), payload)
    result = {
        k: payload[k]
        for k in ("schema", "step", "config", "execution", "provenance", "sampling")
    }
    result.update(
        checkpoint_sha256=digest,
        checkpoint_bytes=path.stat().st_size,
        parameters=params,
        completion_verified=True,
        local_preverified_files=certificate,
        local_preverified_corrector=corrector_certificate,
        precision_runtime_source="explicit_checkpoint_metadata",
    )
    return result


def policy_settings(args, identity):
    execution = identity["execution"]
    settings = dict(
        ckpt_setting=args.checkpoint,
        corrector_checkpoint=args.corrector,
        seed=execution["seed"],
        sim_cfg_path=str(Path(args.upstream) / "configs/sim_robotwin.yaml"),
        sim_task="robotwin_uncond_3cam_384_1e-4",
        mixed_precision="bf16",
        dataset_stats_path=args.stats,
        action_horizon=32,
        replan_steps=20,
        num_inference_steps=20,
        sigma_shift=execution["sigma_shift"],
        text_cfg_scale=execution["text_cfg_scale"],
        negative_prompt=execution["negative_prompt"],
        rand_device=execution["rand_device"],
        tiled=False,
        timing_enabled=True,
        skip_get_obs_within_replan=False,
        wan=args.wan,
        endpoint_corrector_autocast=True,
        endpoint_runtime=dict(INFERENCE_SETTINGS),
        endpoint_text_cache_mib=args.text_cache_mib,
        endpoint_frozen_graph=getattr(args, "frozen_graph", "auto"),
        endpoint_rgb_mode="native_rgb",
    )
    if "checkpoint_sha256" in identity:
        settings["endpoint_corrector_sha256"] = identity["checkpoint_sha256"]
    return settings


def configure_environment(args, threads):
    runtime = Path(args.runtime)
    paths = [
        ROOT / "src",
        Path(args.upstream) / "src",
        runtime / "curobo/src",
        runtime / "sim",
        runtime / "geometry",
        runtime / "flash_attn",
        Path(args.extra_deps),
    ]
    os.environ["PYTHONPATH"] = os.pathsep.join(map(str, paths))
    for path in reversed(paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ["CUDA_HOME"] = str(runtime / "cuda")

    os.environ["LD_LIBRARY_PATH"] = (
        str(runtime / "cuda/lib64") + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    )
    os.environ["PATH"] = os.pathsep.join(
        (str(runtime / "cuda/bin"), str(runtime / "bin"), os.environ["PATH"])
    )
    os.environ["PYTHONUNBUFFERED"] = os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MAX_JOBS"] = "1"
    os.environ.pop("FSWM_RENDER_PCI", None)
    os.environ.pop("TORCH_CUDA_ARCH_LIST", None)


def runtime_preflight(args):
    import importlib.metadata as metadata
    import importlib.util

    import torch

    if sys.version_info[:2] != (3, 12) or torch.__version__ != "2.9.0+cu126":
        raise RuntimeError("Packaged runtime requires Python3.12 / torch2.9.0+cu126")
    versions = {
        name: metadata.version(name)
        for name in ("sapien", "numpy", "mplib", "toppra", "trimesh")
    }
    if versions["sapien"] != "3.0.0b1" or not versions["numpy"].startswith("1."):
        raise RuntimeError("Incompatible SAPIEN/NumPy runtime")
    from reread.evaluation.runtime_setup import validate_sapien_renderer_api

    stub = (
        Path(importlib.util.find_spec("sapien").origin).parent / "pysapien/render.pyi"
    )
    validate_sapien_renderer_api(stub.read_text())
    for path in (
        Path(args.assets) / "embodiments/aloha-agilex/config.yml",
        Path(args.upstream) / "configs/sim_robotwin.yaml",
        Path(args.runtime) / "cuda/lib64",
        Path(args.runtime) / "curobo/src/curobo",
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    encoders = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    if "libx264" not in encoders.stdout:
        raise RuntimeError("ffmpeg requires libx264 for failure videos")
    return dict(
        torch=torch.__version__,
        cuda=torch.version.cuda,
        packages=versions,
        ffmpeg=shutil.which("ffmpeg"),
    )
