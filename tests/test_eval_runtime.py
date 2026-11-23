import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import reread.evaluation.runtime_setup as runtime

PLANNER = """def enclosing():
    from curobo.types.math import Pose as CuroboPose
    class Planner:
        def method(self):
            cfg = MotionGenConfig.load_from_robot_config(
                self.yml_path,
                world_config,
                interpolation_dt=1 / 250,
            )
            cfg = MotionGenConfig.load_from_robot_config(
                self.yml_path,
                world_config,
                interpolation_dt=1 / 250,
            )
            goal = CuroboPose.from_list(list(target_pose_p) + list(target_pose_q))
            joint = torch.tensor(joint_angles).cuda().reshape(1, -1)
            poses = torch.tensor(poses_list, dtype=torch.float32).cuda()
            batch_goal = CuroboPose(poses[:, :3], poses[:, 3:])
            joint_batch = torch.tensor(joint_angles, dtype=torch.float32).cuda().reshape(1, -1)
"""


@pytest.mark.parametrize(
    "value, expected",
    [
        ("00000000:8A:00.0", "0000:8a:00.0"),
        (" 0001:af:02.3 ", "0001:af:02.3"),
    ],
)
def test_normalize_pci(value, expected):
    assert runtime.normalize_pci(value) == expected


@pytest.mark.parametrize(
    "value", ["GPU-abc", "0:8a:00.0", "0000:8a:00.9", "0000:xx:00.0"]
)
def test_bad_pci(value):
    with pytest.raises(ValueError):
        runtime.normalize_pci(value)


@pytest.mark.parametrize(
    "index,visibility,expected",
    [
        (6, None, "6"),
        (2, "7,4,1,6,5,0,3,2", "1"),
        (1, "GPU-platform-a,GPU-platform-b", "GPU-platform-b"),
        (0, " 3 , 1 ", "3"),
    ],
)
def test_worker_mapping_selects_parent_visible_token(index, visibility, expected):
    assert runtime.worker_cuda_token(index, visibility) == expected


@pytest.mark.parametrize(
    "index,visibility",
    [
        (-1, None),
        (True, None),
        (0, ""),
        (0, "-1"),
        (0, "0,,1"),
        (1, "0,0"),
        (2, "0,1"),
    ],
)
def test_invalid_parent_visibility_cannot_bind_a_worker(index, visibility):
    with pytest.raises(ValueError):
        runtime.worker_cuda_token(index, visibility)


def test_bind_single_worker_before_torch_import_preserves_non_cuda_visibility(
    monkeypatch,
):
    monkeypatch.setattr(runtime, "sys", SimpleNamespace(modules={}))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7,4,1,6,5,0,3,2")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "platform_inventory")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    runtime.bind_worker_visibility({"cuda_visible_devices": "1", "gpu_index": 2})
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert os.environ["NEWM_EVAL_CUDA_INDEX"] == "0"
    assert os.environ["NVIDIA_VISIBLE_DEVICES"] == "platform_inventory"
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


@pytest.mark.parametrize("token", ["", "0,1", "-1", None, " 0 "])
def test_bind_rejects_invalid_worker_visibility(monkeypatch, token):
    monkeypatch.setattr(runtime, "sys", SimpleNamespace(modules={}))
    with pytest.raises(ValueError):
        runtime.bind_worker_visibility({"cuda_visible_devices": token})


def test_bind_rejects_torch_import_even_before_cuda_initialization(monkeypatch):
    monkeypatch.setattr(runtime, "sys", SimpleNamespace(modules={"torch": object()}))
    with pytest.raises(RuntimeError, match="before importing Torch"):
        runtime.bind_worker_visibility({"cuda_visible_devices": "0"})


def fake_cuda(monkeypatch, *, count=8, name="NVIDIA H800", free=79000, total=81000):
    selected = []
    cuda = SimpleNamespace(
        init=lambda: None,
        device_count=lambda: count,
        set_device=lambda index: selected.append(index),
        get_device_properties=lambda index: SimpleNamespace(name=name, major=8),
        mem_get_info=lambda index: (free * 1024**2, total * 1024**2),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(
        runtime, "cuda_pci_bus_id", lambda index: f"0000:{index + 10:02x}:00.0"
    )
    return selected


def test_probe_runtime_ordinal_and_no_visibility_mutation(monkeypatch):
    selected = fake_cuda(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7,6,5,4,3,2,1,0")
    rows = runtime.probe_gpus()
    assert [row["index"] for row in rows] == list(range(8)) == selected
    assert len({row["pci"] for row in rows}) == 8
    assert rows[0]["free"] == 79000 * 1024**2
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "7,6,5,4,3,2,1,0"


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"count": 7}, "exactly eight"),
        ({"name": "NVIDIA V100"}, "supported 80GB"),
        ({"total": 40000}, "supported 80GB"),
        ({"free": 20000}, "Do not overlap"),
    ],
)
def test_probe_refuses_wrong_or_busy_gpus(monkeypatch, kwargs, error):
    fake_cuda(monkeypatch, **kwargs)
    with pytest.raises(RuntimeError, match=error):
        runtime.probe_gpus()


@pytest.mark.parametrize(
    "name",
    ["NVIDIA A800", "NVIDIA A100-SXM4-80GB", "NVIDIA H800", "NVIDIA H100 80GB HBM3"],
)
def test_supported_ampere_and_hopper_not_hardbound_h800(monkeypatch, name):
    fake_cuda(monkeypatch, name=name)
    assert len(runtime.probe_gpus()) == 8


def test_duplicate_physical_device_refused(monkeypatch):
    fake_cuda(monkeypatch)
    monkeypatch.setattr(runtime, "cuda_pci_bus_id", lambda index: "0000:8a:00.0")
    with pytest.raises(RuntimeError, match="distinct physical"):
        runtime.probe_gpus()


@pytest.mark.parametrize("value", [0, -1, True, 0.5])
def test_bad_free_memory_threshold(value):
    with pytest.raises(ValueError):
        runtime.probe_gpus(value)


def test_planner_default_device_patches_are_exact():
    patched = runtime.patch_planner(PLANNER)
    assert (
        patched.count(
            "tensor_args=TensorDeviceType(device=torch.device('cuda', torch.cuda.current_device()))"
        )
        == 2
    )
    assert "tensor_args=self.motion_gen.tensor_args" in patched
    assert patched.count(".tensor_args.to_device(") == 3
    assert ".cuda()" not in patched

    assert "CuroboPose(poses[:, :3], poses[:, 3:])" in patched
    ast.parse(patched)
    with pytest.raises(ValueError, match="Unknown CuRobo planner source"):
        runtime.patch_planner(patched)
    with pytest.raises(ValueError, match="Unknown CuRobo planner source"):
        runtime.patch_planner(PLANNER.replace("interpolation_dt=", "other="))


def test_renderer_uses_worker_device_and_no_aux_module():
    ast.parse(runtime.GPU_SCENE_SOURCE)
    assert "NEWM_EVAL_CUDA_INDEX" in runtime.GPU_SCENE_SOURCE
    assert "actual.cuda_id != index" in runtime.GPU_SCENE_SOURCE
    assert "cuda_pci_bus_id(index) != pci" in runtime.GPU_SCENE_SOURCE


def test_native_renderer_constructor_api_validation():
    runtime.validate_sapien_renderer_api(
        "class SapienRenderer:\n    def __init__(self, device: sapien.pysapien.Device | None = None) -> None:\n        ..."
    )
    with pytest.raises(RuntimeError, match="explicit Device"):
        runtime.validate_sapien_renderer_api(
            "class SapienRenderer:\n    def __init__(self, **kwargs): pass"
        )


def test_legacy_renderer_patch_calls_native_with_selected_pci(monkeypatch):
    old = "def setup(self):\n    self.renderer = sapien.SapienRenderer()\n    self.engine.set_renderer(self.renderer)\n"
    patched = runtime.patch_base_task(old)
    assert "sapien.SapienRenderer()" not in patched
    with pytest.raises(ValueError):
        runtime.patch_base_task(patched)
    monkeypatch.setenv("FSWM_RENDER_PCI", "0000:a5:00.0")
    passed = []
    fake = SimpleNamespace(
        render=SimpleNamespace(
            SapienRenderer=lambda device: passed.append(device) or "native_renderer"
        ),
        Device=lambda device: device,
    )
    namespace = dict(sapien=fake, os=os)
    exec(patched, namespace)
    instance = SimpleNamespace(
        engine=SimpleNamespace(set_renderer=lambda renderer: passed.append(renderer))
    )
    namespace["setup"](instance)
    assert passed == ["pci:0000:a5:00.0", "native_renderer"]


def test_actual_bundled_gpu_patches_compile():
    root = Path(__file__).resolve().parents[1] / "third_party/RoboTwin"
    planner = (root / "envs/robot/planner.py").read_text(encoding="utf-8")
    base_task = (root / "envs/_base_task.py").read_text(encoding="utf-8")
    compile(runtime.patch_planner(planner), "planner.py", "exec")
    compile(runtime.patch_base_task(base_task), "_base_task.py", "exec")


def test_driver_query_uses_visible_ordinal_not_nvml(monkeypatch):
    calls = []

    class Function:
        def __init__(self, name, function):
            self.name, self.function = name, function

        def __call__(self, *args):
            calls.append(self.name)
            return self.function(*args)

    def device_get(pointer, ordinal):
        assert ordinal == 6
        pointer._obj.value = 42
        return 0

    def get_pci(buffer, size, device):
        assert device == 42 and size == 32
        buffer.value = b"00000000:A5:00.0"
        return 0

    driver = SimpleNamespace(
        cuInit=Function("init", lambda flags: 0),
        cuDeviceGet=Function("device", device_get),
        cuDeviceGetPCIBusId=Function("pci", get_pci),
    )
    monkeypatch.setattr(runtime.ctypes, "CDLL", lambda name: driver)
    assert runtime.cuda_pci_bus_id(6) == "0000:a5:00.0"
    assert calls == ["init", "device", "pci"]


@pytest.mark.parametrize("renderer_index, should_pass", [(6, True), (0, False)])
def test_renderer_accepts_nonzero_worker_ordinal_only(
    monkeypatch, renderer_index, should_pass
):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(current_device=lambda: 6)),
    )
    actual = SimpleNamespace(
        is_cpu=lambda: False,
        can_render=lambda: True,
        pci_string="0000:a5:00.0",
        cuda_id=renderer_index,
    )
    fake = SimpleNamespace(
        physx=SimpleNamespace(
            set_scene_config=lambda config: None, PhysxCpuSystem=lambda: "cpu"
        ),
        render=SimpleNamespace(
            RenderSystem=lambda device: SimpleNamespace(device=actual)
        ),
        Device=lambda address: address,
        Scene=lambda systems: tuple(systems),
    )
    monkeypatch.setitem(sys.modules, "sapien", fake)
    monkeypatch.setenv("NEWM_EVAL_CUDA_INDEX", "6")
    monkeypatch.setenv("FSWM_RENDER_PCI", "0000:a5:00.0")
    monkeypatch.setattr(runtime, "cuda_pci_bus_id", lambda index: "0000:a5:00.0")
    namespace = {}
    exec(runtime.GPU_SCENE_SOURCE, namespace)
    if should_pass:
        assert namespace["create_scene"](None, None)[0] == "cpu"
    else:
        with pytest.raises(RuntimeError, match="ordinal mismatch"):
            namespace["create_scene"](None, None)


def fake_tree(root):
    source, assets, upstream, policy, output = [
        root / name for name in ("source", "assets", "upstream", "policy", "output")
    ]
    contents = {
        source / "envs/robot/planner.py": PLANNER,
        source / "envs/gpu_scene.py": "old_renderer = True\n",
        source
        / "envs/_base_task.py": "def setup(self):\n    self.renderer = sapien.SapienRenderer()\n",
        source / "script/eval_policy.py": "pass\n",
        source / "script/eval_protocol.py": "pass\n",
        source / "task_config/_eval_step_limit.yml": "turn_switch: 400\n",
        source / "data/private.txt": "excluded",
        source / "__pycache__/private.txt": "excluded",
        upstream / "experiments/robotwin/fastwam_policy/deploy_policy.py": "pass\n",
        policy / "deploy_policy.py": "pass\n",
        assets
        / "embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf": "<robot/>\n",
        assets / "objects/example.json": "{}",
    }
    for side in ("left", "right"):
        contents[assets / f"embodiments/aloha-agilex/curobo_{side}.yml"] = (
            "robot_cfg:\n  kinematics:\n    urdf_path: /old/robot.urdf\n    collision_spheres: /old/sphere.yml\n"
        )
        contents[assets / f"embodiments/aloha-agilex/collision_aloha_{side}.yml"] = "{}"
    for path, content in contents.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    output.mkdir()
    return output, source, assets, upstream, policy


def test_private_simulator_portable_nonoverwriting(tmp_path):
    pytest.importorskip("yaml")
    arguments = fake_tree(tmp_path)
    output, source, assets, upstream, policy = arguments
    before = (source / "envs/robot/planner.py").read_bytes()
    try:
        sim = runtime.prepare_simulator(*arguments)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip(
                "Windows symlink privilege unavailable; full tree test runs on Linux CPU"
            )
        raise
    assert sim == output / "simulator"
    assert (sim / "policy/reread_policy").resolve() == policy
    assert (sim / "assets/objects").resolve() == assets / "objects"
    assert not (sim / "data").exists()
    assert not (sim / "__pycache__").exists()
    assert (source / "envs/robot/planner.py").read_bytes() == before
    assert "old_renderer" not in (sim / "envs/gpu_scene.py").read_text()
    assert (
        str(sim / "assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf")
        in (sim / "assets/embodiments/aloha-agilex/curobo_left.yml").read_text()
    )
    assert (
        "/old/robot.urdf"
        in (assets / "embodiments/aloha-agilex/curobo_left.yml").read_text()
    )
    assert (output / "simulator_setup.json").is_file()
    with pytest.raises(FileExistsError):
        runtime.prepare_simulator(*arguments)


def test_unknown_planner_fails_before_creating_private_tree(tmp_path):
    arguments = fake_tree(tmp_path)
    (arguments[1] / "envs/robot/planner.py").write_text("pass\n")
    with pytest.raises(ValueError):
        runtime.prepare_simulator(*arguments)
    assert not (arguments[0] / "simulator").exists()
