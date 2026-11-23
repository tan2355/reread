from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path


def normalize_pci(value: str) -> str:
    match = re.fullmatch(
        r"([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])", value.strip()
    )
    if match is None:
        raise ValueError(f"Invalid CUDA PCI bus address: {value!r}")
    return f"{int(match[1], 16):04x}:{match[2].lower()}:{match[3].lower()}.{match[4]}"


def cuda_pci_bus_id(index: int) -> str:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("CUDA ordinal must be a nonnegative integer")
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    driver.cuDeviceGet.restype = ctypes.c_int
    driver.cuDeviceGetPCIBusId.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    driver.cuDeviceGetPCIBusId.restype = ctypes.c_int

    def check(code, operation):
        if code != 0:
            raise RuntimeError(f"CUDA driver {operation} failed with error {code}")

    check(driver.cuInit(0), "cuInit")
    device = ctypes.c_int()
    check(driver.cuDeviceGet(ctypes.byref(device), index), "cuDeviceGet")
    buffer = ctypes.create_string_buffer(32)
    check(
        driver.cuDeviceGetPCIBusId(buffer, len(buffer), device.value),
        "cuDeviceGetPCIBusId",
    )
    return normalize_pci(buffer.value.decode("ascii"))


def worker_cuda_token(index: int, original_visibility: str | None) -> str:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("CUDA ordinal must be a nonnegative integer")
    if original_visibility is None:
        return str(index)
    tokens = [token.strip() for token in original_visibility.split(",")]
    if (
        index >= len(tokens)
        or any(not token or token == "-1" for token in tokens)
        or len(set(tokens)) != len(tokens)
    ):
        raise ValueError("Invalid parent CUDA visibility mapping")
    return tokens[index]


def bind_worker_visibility(config) -> None:
    token = config.get("cuda_visible_devices")
    if (
        not isinstance(token, str)
        or not token.strip()
        or token != token.strip()
        or "," in token
        or token == "-1"
    ):
        raise ValueError(
            "Worker requires exactly one parent-authenticated CUDA visibility token"
        )
    if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
        raise RuntimeError(
            "Worker CUDA visibility must be bound before importing Torch"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = token
    os.environ["NEWM_EVAL_CUDA_INDEX"] = "0"


def probe_gpus(minimum_free_mib: int = 70000) -> list[dict]:
    if (
        isinstance(minimum_free_mib, bool)
        or not isinstance(minimum_free_mib, int)
        or minimum_free_mib <= 0
    ):
        raise ValueError("minimum_free_mib must be a positive integer")
    import torch

    torch.cuda.init()
    count = torch.cuda.device_count()
    if count != 8:
        raise RuntimeError(
            f"This evaluation requires exactly eight visible GPUs per node, found {count}"
        )
    result = []
    for index in range(count):
        torch.cuda.set_device(index)
        properties = torch.cuda.get_device_properties(index)
        free, total = map(int, torch.cuda.mem_get_info(index))
        if (
            not re.search(r"\b(?:A800|A100|H800|H100)\b", properties.name.upper())
            or total < 75000 * 1024**2
            or properties.major < 8
        ):
            raise RuntimeError(
                f"CUDA {index} is {properties.name!r}; requires a supported 80GB Ampere/Hopper GPU"
            )
        if free < minimum_free_mib * 1024**2:
            raise RuntimeError(
                f"CUDA {index} has only {free / 1024**2:.0f} MiB free; "
                f"requires {minimum_free_mib} MiB. Do not overlap evaluation with training."
            )
        result.append(
            dict(
                index=index,
                name=properties.name,
                pci=cuda_pci_bus_id(index),
                free=free,
                total=total,
                memory_free=free,
                memory_total=total,
                free_mib=free // 1024**2,
                total_mib=total // 1024**2,
            )
        )
    if len({row["pci"] for row in result}) != count:
        raise RuntimeError(
            "Runtime ordinals do not identify eight distinct physical GPUs"
        )
    return result


GPU_SCENE_SOURCE = '''"""Private evaluation renderer: bind to the worker's actual CUDA ordinal."""
import os


def create_scene(engine, config):
    import torch
    import sapien
    from reread.evaluation.runtime_setup import cuda_pci_bus_id, normalize_pci
    index = torch.cuda.current_device()
    expected_index = int(os.environ["NEWM_EVAL_CUDA_INDEX"])
    if index != expected_index:
        raise RuntimeError(f"CUDA device changed inside worker: {index} != {expected_index}")
    pci = normalize_pci(os.environ["FSWM_RENDER_PCI"])
    if cuda_pci_bus_id(index) != pci:
        raise RuntimeError("Renderer PCI no longer matches worker CUDA device")
    sapien.physx.set_scene_config(config)
    system = sapien.render.RenderSystem(sapien.Device("pci:" + pci))
    actual = system.device
    if actual.is_cpu() or not actual.can_render() or normalize_pci(actual.pci_string) != pci:
        raise RuntimeError(f"Simulator renderer GPU mismatch: requested {pci}, got {actual}")
    if actual.cuda_id != index:
        raise RuntimeError(f"Renderer CUDA ordinal mismatch: expected {index}, got {actual.cuda_id}")
    return sapien.Scene([sapien.physx.PhysxCpuSystem(), system])
'''


def patch_planner(source: str) -> str:
    substitutions = (
        (
            "    from curobo.types.math import Pose as CuroboPose",
            "    from curobo.types.math import Pose as CuroboPose\n    from curobo.types.base import TensorDeviceType",
            1,
        ),
        (
            "                world_config,\n                interpolation_dt=",
            "                world_config,\n                tensor_args=TensorDeviceType(device=torch.device('cuda', torch.cuda.current_device())),\n                interpolation_dt=",
            2,
        ),
        (
            "CuroboPose.from_list(list(target_pose_p) + list(target_pose_q))",
            "CuroboPose.from_list(list(target_pose_p) + list(target_pose_q), tensor_args=self.motion_gen.tensor_args)",
            1,
        ),
        (
            "torch.tensor(joint_angles).cuda().reshape(1, -1)",
            "self.motion_gen.tensor_args.to_device(joint_angles).reshape(1, -1)",
            1,
        ),
        (
            "torch.tensor(poses_list, dtype=torch.float32).cuda()",
            "self.motion_gen_batch.tensor_args.to_device(poses_list)",
            1,
        ),
        (
            "torch.tensor(joint_angles, dtype=torch.float32).cuda().reshape(1, -1)",
            "self.motion_gen_batch.tensor_args.to_device(joint_angles).reshape(1, -1)",
            1,
        ),
    )
    for old, new, count in substitutions:
        actual = source.count(old)
        if actual != count:
            raise ValueError(
                f"Unknown CuRobo planner source: expected {count} matches for {old!r}, found {actual}"
            )
        source = source.replace(old, new)
    compile(source, "private/envs/robot/planner.py", "exec")
    return source


def validate_sapien_renderer_api(stub: str) -> None:
    signature = (
        r"class SapienRenderer:\s+def __init__\(self, "
        r"device: sapien\.pysapien\.Device \| None = None\) -> None:"
    )
    if re.search(signature, stub) is None:
        raise RuntimeError(
            "Packaged SAPIEN native renderer lacks the verified explicit Device constructor"
        )


def patch_base_task(source: str) -> str:
    old = "self.renderer = sapien.SapienRenderer()"
    new = (
        "self.renderer = sapien.render.SapienRenderer("
        "sapien.Device('pci:' + os.environ['FSWM_RENDER_PCI']))"
    )
    if source.count(old) != 1:
        raise ValueError("Unknown RoboTwin BaseTask legacy renderer source")
    source = source.replace(old, new)
    compile(source, "private/envs/_base_task.py", "exec")
    return source


def prepare_simulator(output, source, assets, upstream, policy) -> Path:
    output = Path(output).resolve(strict=True)
    source, assets, upstream, policy = (
        Path(path).resolve(strict=True) for path in (source, assets, upstream, policy)
    )
    if not output.is_dir():
        raise NotADirectoryError(output)
    sim = output / "simulator"
    if sim.exists() or sim.is_symlink():
        raise FileExistsError(sim)
    required = (
        source / "envs/robot/planner.py",
        source / "envs/_base_task.py",
        source / "script/eval_policy.py",
        source / "script/eval_protocol.py",
        source / "task_config/_eval_step_limit.yml",
        upstream / "experiments/robotwin/fastwam_policy/deploy_policy.py",
        policy / "deploy_policy.py",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    planner = patch_planner(
        (source / "envs/robot/planner.py").read_text(encoding="utf-8")
    )
    base_task = patch_base_task(
        (source / "envs/_base_task.py").read_text(encoding="utf-8")
    )
    shutil.copytree(
        source,
        sim,
        ignore=shutil.ignore_patterns(
            "assets",
            "data",
            "eval_result",
            "__pycache__",
            "*.pyc",
            ".git",
            "fastwam_policy",
            "reread_policy",
        ),
    )
    policy_root = sim / "policy"
    policy_root.mkdir(exist_ok=True)
    (policy_root / "fastwam_policy").symlink_to(
        upstream / "experiments/robotwin/fastwam_policy", target_is_directory=True
    )
    (policy_root / "reread_policy").symlink_to(policy, target_is_directory=True)
    asset_view = sim / "assets"
    asset_view.mkdir()
    for entry in assets.iterdir():
        if entry.name != "embodiments" and entry.is_dir():
            (asset_view / entry.name).symlink_to(entry, target_is_directory=True)
    robot = asset_view / "embodiments/aloha-agilex"
    shutil.copytree(assets / "embodiments/aloha-agilex", robot)
    import yaml

    for side in ("left", "right"):
        path = robot / f"curobo_{side}.yml"
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        kin = cfg["robot_cfg"]["kinematics"]
        kin["urdf_path"] = str(robot / "urdf/arx5_description_isaac.urdf")
        kin["collision_spheres"] = str(robot / f"collision_aloha_{side}.yml")
        for key in ("urdf_path", "collision_spheres"):
            if not Path(kin[key]).is_file():
                raise FileNotFoundError(kin[key])
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (sim / "envs/robot/planner.py").write_text(planner, encoding="utf-8")
    (sim / "envs/_base_task.py").write_text(base_task, encoding="utf-8")
    (sim / "envs/gpu_scene.py").write_text(GPU_SCENE_SOURCE, encoding="utf-8")
    suspicious = []
    for path in (sim / "envs").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if re.search(
            r"['\"]cuda:0['\"]|torch\.device\(\s*['\"]cuda['\"]\s*,\s*0\s*\)", text
        ):
            suspicious.append(str(path.relative_to(sim)))
    if suspicious:
        raise ValueError(
            f"Unreviewed fixed CUDA device in private simulator: {suspicious}"
        )
    hashes = {
        name: hashlib.sha256((sim / name).read_bytes()).hexdigest()
        for name in ("envs/gpu_scene.py", "envs/robot/planner.py", "envs/_base_task.py")
    }
    with (output / "simulator_setup.json").open("x", encoding="utf-8") as stream:
        json.dump(
            dict(
                source=str(source),
                assets=str(assets),
                upstream=str(upstream),
                policy=str(policy),
                patched_sha256=hashes,
                device_binding="preserved_cuda_visibility_runtime_ordinal_and_pci",
            ),
            stream,
            indent=2,
        )
    return sim
