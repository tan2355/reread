import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from reread.evaluation.worker import (
    FailureVideoBuffer,
    evaluate_episode,
    is_expert_infrastructure_error,
    worker_process,
)


class Unstable(Exception):
    pass


class Protocol:
    REPLAY_SCHEMA = 1

    @staticmethod
    def jsonable(value):
        return value

    @staticmethod
    def rng_state():
        return {"python": random.getstate()}

    @staticmethod
    def restore_rng(state):
        def tuples(value):
            return (
                tuple(map(tuples, value)) if isinstance(value, (list, tuple)) else value
            )

        random.setstate(tuples(state["python"]))

    @staticmethod
    def observation_signature(observation):
        return {"observed_step": observation["step"]}

    @staticmethod
    def write_record(path, value):
        Path(path).write_text(json.dumps(value))


class Environment:
    def __init__(self, runtime):
        self.runtime = runtime
        self.take_action_cnt = 0
        self.step_lim = 3
        self.eval_success = False
        self.render_freq = 0
        self.closes = []
        self.setup_count = {}

    def setup_demo(self, *, now_ep_num, seed, **args):
        self.runtime.setups.append((now_ep_num, seed, args))
        self.seed = seed
        self.take_action_cnt = 0
        self.eval_success = False
        self.setup_count[seed] = self.setup_count.get(seed, 0) + 1
        if self.setup_count[seed] == 2 and self.runtime.rollout_error:
            raise self.runtime.rollout_error

    def play_once(self):
        outcome = self.runtime.screen.get(self.seed, "accepted")
        if isinstance(outcome, Exception):
            raise outcome
        self.plan_success = outcome != "rejected"
        return {"info": {"object": "bottle"}}

    def check_success(self):
        return self.plan_success

    def close_env(self, clear_cache=False):
        self.closes.append(clear_cache)
        if (
            self.runtime.cleanup_error
            and len(self.closes) > self.runtime.cleanup_error_after
        ):
            raise self.runtime.cleanup_error

    def set_instruction(self, instruction):
        self.instruction = instruction

    def _set_eval_video_ffmpeg(self, sink):
        self.sink = sink

    def _del_eval_video_ffmpeg(self):
        self.sink.stdin.close()
        self.sink.wait()

    def get_obs(self):
        self.runtime.observations += 1
        return {"step": self.take_action_cnt}


class Runtime:
    protocol = Protocol
    unstable_error = Unstable

    def __init__(
        self,
        *,
        success=True,
        screen=None,
        rollout_error=None,
        policy_error=None,
        cleanup_error=None,
        cleanup_error_after=0,
    ):
        self.success = success
        self.screen = screen or {}
        self.rollout_error = rollout_error
        self.policy_error = policy_error
        self.cleanup_error = cleanup_error
        self.cleanup_error_after = cleanup_error_after
        self.setups = []
        self.observations = 0
        self.resets = 0
        self.instruction_types = []

    @staticmethod
    def is_expert_infrastructure_error(error):
        return is_expert_infrastructure_error(error)

    def seed_numpy(self, seed):
        pass

    def new_environment(self, task):
        self.env = Environment(self)
        return self.env

    def environment_args(self, task, phase):
        return {"render_freq": 0, "clear_cache_freq": 5}

    def sample_instruction(self, task, info, kind, *, seed):
        self.instruction_types.append(kind)
        instruction = f"{kind} {random.random()}"
        return instruction, {
            "instruction": instruction,
            "instruction_source": "official_" + kind,
            "description_count": 25,
            "instruction_seed": seed,
            "episode_info": info,
        }

    def protocol_identity(self, task, phase):
        return {"env.py": "a" * 64}

    def video_dimensions(self, args):
        return 2, 2

    def reset_model(self):
        self.resets += 1

    def policy_step(self, env, observation):
        if self.policy_error:
            raise self.policy_error
        env.sink.stdin.write(b"\0" * 12)
        env.take_action_cnt += 1
        if self.success and env.take_action_cnt == 2:
            env.eval_success = True

    def timing(self):
        return {"infer_s": 0.1}


@pytest.fixture
def config(tmp_path):
    return {
        "output_root": str(tmp_path),
        "worker_id": 3,
        "gpu_index": 1,
        "slot": 1,
        "ffmpeg_threads": 1,
        "policy_args": {"seed": 0},
    }


@pytest.fixture
def job():
    return {
        "job_id": "turn_switch/demo_clean/000",
        "task": "turn_switch",
        "phase": "demo_clean",
        "episode_index": 0,
        "seed": 100000,
        "max_seed_attempts": 10,
        "candidate_stride": 25,
        "rng_state": None,
    }


def test_success_never_encodes_or_writes_video(config, job, monkeypatch):
    monkeypatch.setattr(
        FailureVideoBuffer,
        "save_failure",
        lambda *a, **k: pytest.fail("encoded success"),
    )
    runtime = Runtime()
    result = evaluate_episode(job, runtime, config)
    assert result["success"] is True and result["status"] == "success"
    assert result["video"] is None
    assert result["seed"] == 100000 and result["next_seed"] == 100025
    assert result["inference_seed"] == 0
    assert runtime.resets == 1 and runtime.observations == 2
    assert not list(Path(config["output_root"]).rglob("*.mp4"))
    assert len(runtime.setups) == 2
    assert all(values[2]["eval_video_save_dir"] is None for values in runtime.setups)


def test_only_policy_failures_encode_video(config, job, monkeypatch):
    called = []

    def encode(buffer, path, *, threads):
        called.append((len(buffer.frames), threads))
        Path(path).write_bytes(b"video")
        buffer.discard()
        return str(path)

    monkeypatch.setattr(FailureVideoBuffer, "save_failure", encode)
    result = evaluate_episode(job, Runtime(success=False), config)
    assert result["status"] == "failure" and result["success"] is False
    assert Path(result["video"]).read_bytes() == b"video"
    assert called == [(3, 1)]


def test_rejects_invalid_expert_seeds_with_disjoint_lane_stride(config, job):
    runtime = Runtime(
        screen={
            100000: "rejected",
            100025: Unstable("unstable"),
            100050: ValueError("expert planning failure"),
        }
    )
    result = evaluate_episode(job, runtime, config)
    assert result["seed"] == 100075 and result["seed_attempts"] == 4
    assert result["next_seed"] == 100100
    assert [entry[1] for entry in runtime.setups] == [
        100000,
        100025,
        100050,
        100075,
        100075,
    ]
    assert runtime.resets == 1


@pytest.mark.parametrize(
    "error",
    [OSError("asset missing"), MemoryError("memory"), RuntimeError("CUDA failure")],
)
def test_infrastructure_error_is_fatal_not_rejected_seed(config, job, error):
    runtime = Runtime(screen={100000: error})
    with pytest.raises(type(error), match=str(error)):
        evaluate_episode(job, runtime, config)
    assert len(runtime.setups) == 1 and runtime.resets == 0
    assert list(Path(config["output_root"]).rglob("error.json"))
    assert not list(Path(config["output_root"]).rglob("result.json"))


WARP_DEVICE_ERROR = (
    "Error launching kernel 'compute_spheres': trying to launch on device=cuda:5, "
    "but input array 'retract_config' is on device=cuda:0"
)


@pytest.mark.parametrize(
    "message",
    [
        WARP_DEVICE_ERROR,
        "Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0!",
        "CUDA error: invalid device ordinal",
        "CUDA_ERROR_INVALID_DEVICE",
        "Error launching kernel compute_spheres on cuda:7",
        "device mismatch for retract_config",
    ],
)
def test_device_and_kernel_failures_abort_first_seed(config, job, message):
    original = RuntimeError(message)
    runtime = Runtime(screen={100000: original})
    with pytest.raises(RuntimeError) as caught:
        evaluate_episode(job, runtime, config)
    assert caught.value is original
    assert len(runtime.setups) == len(runtime.env.closes) == 1
    assert runtime.resets == 0
    attempts = [
        json.loads(line)
        for line in next(Path(config["output_root"]).rglob("seed_attempts.jsonl"))
        .read_text()
        .splitlines()
    ]
    assert [(entry["seed"], entry["status"]) for entry in attempts] == [
        (100000, "infrastructure_error")
    ]
    assert not list(Path(config["output_root"]).rglob("result.json"))


def test_partial_setup_cleanup_preserves_original_warp_error_and_stops_seed_retry(
    config, job
):
    original = RuntimeError(WARP_DEVICE_ERROR)
    runtime = Runtime(
        screen={100000: original},
        cleanup_error=AttributeError("'Robot' object has no attribute 'left_planner'"),
    )
    with pytest.raises(RuntimeError) as caught:
        evaluate_episode(job, runtime, config)
    assert caught.value is original
    assert len(runtime.setups) == len(runtime.env.closes) == 1
    diagnostic = json.loads(
        next(Path(config["output_root"]).rglob("error.json")).read_text()
    )
    assert diagnostic["error_type"] == "RuntimeError"
    assert diagnostic["error"] == WARP_DEVICE_ERROR
    assert "retract_config" in diagnostic["traceback"]
    assert len(diagnostic["cleanup_errors"]) == 1
    assert diagnostic["cleanup_errors"][0]["stage"] == "expert_exception"
    assert diagnostic["cleanup_errors"][0]["error_type"] == "AttributeError"
    assert "left_planner" in diagnostic["cleanup_errors"][0]["error"]


def test_unstable_wrapper_cannot_hide_infrastructure_cause_or_retry_seed(config, job):
    original = Unstable("planner setup failed")
    original.__cause__ = RuntimeError(WARP_DEVICE_ERROR)
    runtime = Runtime(screen={100000: original})
    with pytest.raises(Unstable) as caught:
        evaluate_episode(job, runtime, config)
    assert caught.value is original
    assert len(runtime.setups) == len(runtime.env.closes) == 1
    attempts = [
        json.loads(line)
        for line in next(Path(config["output_root"]).rglob("seed_attempts.jsonl"))
        .read_text()
        .splitlines()
    ]
    assert len(attempts) == 1 and attempts[0]["status"] == "infrastructure_error"
    assert "retract_config" in attempts[0]["traceback"]


def test_exception_classifier_terminates_for_nonfatal_context_cycle():
    error = ValueError("expert planning failure")
    other = Unstable("scene unstable")
    error.__context__ = other
    other.__context__ = error
    assert not is_expert_infrastructure_error(error)


@pytest.mark.parametrize(
    "original", [Unstable("scene unstable"), ValueError("expert planning failure")]
)
def test_rejected_seed_with_failed_cleanup_aborts_without_masking_original(
    config, job, original
):
    runtime = Runtime(
        screen={100000: original}, cleanup_error=AttributeError("left_planner missing")
    )
    with pytest.raises(type(original)) as caught:
        evaluate_episode(job, runtime, config)
    assert caught.value is original
    assert len(runtime.setups) == len(runtime.env.closes) == 1
    diagnostic = json.loads(
        next(Path(config["output_root"]).rglob("error.json")).read_text()
    )
    assert diagnostic["error"] == str(original)
    assert diagnostic["cleanup_errors"][0]["error"] == "left_planner missing"
    assert not list(Path(config["output_root"]).rglob("result.json"))


def test_rollout_error_retains_its_cause_when_cleanup_also_raises(config, job):
    original = RuntimeError("policy GPU failed")
    runtime = Runtime(
        policy_error=original,
        cleanup_error=AttributeError("left_planner missing"),
        cleanup_error_after=1,
    )
    with pytest.raises(RuntimeError) as caught:
        evaluate_episode(job, runtime, config)
    assert caught.value is original
    assert len(runtime.setups) == len(runtime.env.closes) == 2
    diagnostic = json.loads(
        next(Path(config["output_root"]).rglob("error.json")).read_text()
    )
    assert diagnostic["error"] == "policy GPU failed"
    assert diagnostic["cleanup_errors"][0]["stage"] == "episode"
    assert diagnostic["cleanup_errors"][0]["error_type"] == "AttributeError"


@pytest.mark.parametrize(
    "original",
    [
        AttributeError("left_planner missing"),
        RuntimeError("renderer teardown failed"),
        Unstable("teardown failed without a CUDA marker"),
    ],
)
def test_expert_normal_close_failure_is_not_retried_or_double_closed(
    config, job, original
):
    runtime = Runtime(cleanup_error=original)
    with pytest.raises(type(original)) as caught:
        evaluate_episode(job, runtime, config)
    assert caught.value is original
    assert len(runtime.setups) == len(runtime.env.closes) == 1
    diagnostic = json.loads(
        next(Path(config["output_root"]).rglob("error.json")).read_text()
    )
    assert diagnostic["error"] == str(original)
    assert diagnostic["cleanup_errors"] == []
    attempts = [
        json.loads(line)
        for line in next(Path(config["output_root"]).rglob("seed_attempts.jsonl"))
        .read_text()
        .splitlines()
    ]
    assert len(attempts) == 1 and attempts[0]["status"] == "infrastructure_error"


def test_rollout_setup_error_does_not_skip_accepted_seed(config, job):
    runtime = Runtime(rollout_error=Unstable("policy setup unstable"))
    with pytest.raises(Unstable):
        evaluate_episode(job, runtime, config)
    assert [entry[1] for entry in runtime.setups] == [100000, 100000]


def test_policy_error_is_not_success_rate_failure(config, job):
    with pytest.raises(RuntimeError, match="policy broke"):
        evaluate_episode(
            job, Runtime(policy_error=RuntimeError("policy broke")), config
        )
    assert not list(Path(config["output_root"]).rglob("result.json"))


def test_screening_budget_exhaustion(config, job):
    job["max_seed_attempts"] = 2
    with pytest.raises(RuntimeError, match="screening budget"):
        evaluate_episode(
            job, Runtime(screen={100000: "rejected", 100025: "rejected"}), config
        )


def test_independent_job_rng_does_not_depend_on_worker_or_completion_order(config, job):
    first = evaluate_episode(job, Runtime(), config)
    random.seed(9999)
    for _ in range(17):
        random.random()
    other_output = str(Path(config["output_root"]) / "independent_run")
    second = evaluate_episode(
        json.loads(json.dumps(job)),
        Runtime(),
        dict(config, worker_id=7, output_root=other_output),
    )
    assert second["instruction"] == first["instruction"]
    assert second["scene_episode_index"] == 0
    assert second["worker_id"] == 7


def test_independent_lane_cannot_inherit_another_episode_rng(config, job):
    job["rng_state"] = Protocol.rng_state()
    with pytest.raises(ValueError, match="must not inherit"):
        evaluate_episode(job, Runtime(), config)


def test_clean_seen_random_unseen_and_training_disclosure(config, job):
    first = evaluate_episode(job, Runtime(), config)
    second = evaluate_episode(dict(job, phase="demo_randomized"), Runtime(), config)
    assert first["instruction_source"] == "official_seen"
    assert second["instruction_source"] == "official_unseen"
    assert first["randomized_unseen_was_in_training"] is False
    assert second["randomized_unseen_was_in_training"] is True


def test_duplicate_episode_refuses_to_overwrite(config, job):
    evaluate_episode(job, Runtime(), config)
    with pytest.raises(FileExistsError):
        evaluate_episode(job, Runtime(), config)


@pytest.mark.parametrize(
    "key,value", [("task", "../escape"), ("phase", "../escape"), ("episode_index", -1)]
)
def test_unsafe_job_path_rejected(config, job, key, value):
    job[key] = value
    with pytest.raises(ValueError):
        evaluate_episode(job, Runtime(), config)


def test_no_action_progress_is_fatal(config, job):
    runtime = Runtime()
    runtime.policy_step = lambda env, obs: None
    with pytest.raises(RuntimeError, match="no action progress"):
        evaluate_episode(job, runtime, config)


def test_video_frame_size_and_bound():
    buffer = FailureVideoBuffer(2, 2, 1)
    with pytest.raises(ValueError, match="one RGB frame"):
        buffer.write(b"x")
    assert buffer.write(b"x" * 12) == 12
    with pytest.raises(MemoryError, match="bound"):
        buffer.write(b"x" * 12)
    buffer.close()
    with pytest.raises(ValueError, match="closed"):
        buffer.write(b"x" * 12)
    buffer.discard()
    assert not buffer.frames


@pytest.mark.parametrize("dimensions", [(0, 2, 1), (3, 2, 1), (2, 3, 1), (2, 2, 0)])
def test_invalid_video_dimensions(dimensions):
    with pytest.raises(ValueError):
        FailureVideoBuffer(*dimensions)


class FakeEncoder:
    def __init__(self, command, **kwargs):
        self.command = command
        self.input_bytes = bytearray()
        self.stdin = SimpleNamespace(write=self.input_bytes.extend, close=lambda: None)
        self.destination = Path(command[-1])

    def wait(self, timeout):
        self.destination.write_bytes(b"encoded")
        return 0

    def poll(self):
        return 0


def test_failure_encoder_thread_limit_and_atomic_completion(tmp_path):
    created = []

    def popen(command, **kwargs):
        instance = FakeEncoder(command, **kwargs)
        created.append(instance)
        return instance

    buffer = FailureVideoBuffer(2, 2, 2)
    buffer.write(b"x" * 12)
    buffer.write(b"y" * 12)
    result = buffer.save_failure(tmp_path / "failure.mp4", threads=1, popen=popen)
    assert Path(result).read_bytes() == b"encoded"
    command = created[0].command
    assert command[command.index("-threads") + 1] == "1"
    assert "-n" in command and "-y" not in command
    assert created[0].input_bytes == b"x" * 12 + b"y" * 12
    assert not list(tmp_path.glob("*.incomplete.mp4"))


def test_no_frames_or_existing_video_is_error(tmp_path):
    buffer = FailureVideoBuffer(2, 2, 1)
    with pytest.raises(RuntimeError, match="no video frames"):
        buffer.save_failure(tmp_path / "failure.mp4")
    buffer.write(b"x" * 12)
    (tmp_path / "failure.mp4").write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        buffer.save_failure(tmp_path / "failure.mp4")
    assert (tmp_path / "failure.mp4").read_bytes() == b"existing"


def test_worker_loads_model_once_for_many_jobs(monkeypatch):
    import reread.evaluation.worker as worker

    calls = []
    monkeypatch.setattr(worker, "configure_worker", lambda config: None)
    monkeypatch.setattr(
        worker, "RobotWinRuntime", lambda config: calls.append("load") or object()
    )
    monkeypatch.setattr(
        worker,
        "evaluate_episode",
        lambda job, runtime, config, progress: {"job_id": job["job_id"]},
    )
    monkeypatch.setattr(
        worker,
        "acknowledge_worker_exit",
        lambda connection, message: connection.send(message),
    )
    messages = iter(
        [
            {"type": "job", "job": {"job_id": "a"}},
            {"type": "job", "job": {"job_id": "b"}},
            {"type": "stop"},
        ]
    )
    sent = []
    connection = SimpleNamespace(
        recv=lambda: next(messages), send=sent.append, close=lambda: None
    )
    worker_process(connection, {"worker_id": 0, "gpu_index": 0})
    assert calls == ["load"]
    assert [m["type"] for m in sent] == ["ready", "result", "result", "stopped"]


def test_worker_configuration_binds_runtime_gpu_and_limits_cpu_threads(
    tmp_path, monkeypatch
):
    import os
    import sys

    import reread.evaluation.runtime_setup as runtime_setup
    import reread.evaluation.worker as worker

    calls = []
    torch = SimpleNamespace(
        set_num_threads=lambda n: calls.append(("torch_threads", n)),
        set_num_interop_threads=lambda n: calls.append(("interop", n)),
        cuda=SimpleNamespace(
            device_count=lambda: 1, set_device=lambda n: calls.append(("device", n))
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(setNumThreads=lambda n: calls.append(("cv2", n))),
    )

    def pci_bus_id(index):
        assert index == 0
        return "0000:31:00.0"

    monkeypatch.setattr(runtime_setup, "cuda_pci_bus_id", pci_bus_id)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent-runtime-token-for-gpu-1")
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "MAX_JOBS",
        "TOKENIZERS_PARALLELISM",
        "PYTHONDONTWRITEBYTECODE",
        "NEWM_EVAL_CUDA_INDEX",
        "WARP_CACHE_PATH",
        "TORCH_EXTENSIONS_DIR",
        "CUDA_CACHE_PATH",
        "FSWM_RENDER_PCI",
        "XDG_CACHE_HOME",
        "MPLCONFIGDIR",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
    ):
        monkeypatch.setenv(key, "previous")
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    monkeypatch.chdir(tmp_path)
    config = dict(
        output_root=str(tmp_path),
        sim_root=str(tmp_path),
        worker_id=9,
        gpu_index=1,
        cpu_threads=10,
        cpu_affinity=[],
        render_pci="0000:31:00.0",
        policy_args={},
        cuda_visible_devices="parent-runtime-token-for-gpu-1",
    )
    worker.configure_worker(config)
    assert calls == [("torch_threads", 10), ("interop", 1), ("device", 0), ("cv2", 1)]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == config["cuda_visible_devices"]
    assert os.environ["NEWM_EVAL_CUDA_INDEX"] == "0"
    assert os.environ["OMP_NUM_THREADS"] == "10"
    assert os.environ["OPENBLAS_NUM_THREADS"] == os.environ["MAX_JOBS"] == "1"
    cache_root = tmp_path / "runtime_cache/worker_09"
    for key in (
        "WARP_CACHE_PATH",
        "CUDA_CACHE_PATH",
        "XDG_CACHE_HOME",
        "MPLCONFIGDIR",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
    ):
        cache_path = Path(os.environ[key])
        assert cache_path.is_relative_to(cache_root)
        assert cache_path.is_dir()
    assert (
        Path(os.environ["TORCH_EXTENSIONS_DIR"])
        == tmp_path / "runtime_cache/node_0/shared_torch_extensions"
    )
    assert Path(os.environ["TORCH_EXTENSIONS_DIR"]).is_dir()
    assert sys.dont_write_bytecode is True
    assert config["policy_args"]["device"] == "cuda:0"
    assert config["gpu_index"] == 1


@pytest.mark.parametrize(
    "configured,visible",
    [
        (None, "0,1,2,3,4,5,6,7"),
        ("3", "0,1,2,3,4,5,6,7"),
        ("3", "4"),
    ],
)
def test_worker_rejects_missing_or_changed_visibility_before_torch_import(
    tmp_path, monkeypatch, configured, visible
):
    import os

    from reread.evaluation.worker import configure_worker

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    config = dict(cuda_visible_devices=configured, output_root=str(tmp_path))
    with pytest.raises(RuntimeError, match="visibility was not bound"):
        configure_worker(config)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == visible
    assert not (tmp_path / "runtime_cache").exists()


def test_worker_rejects_multiple_visible_devices_before_cuda_selection(
    tmp_path, monkeypatch
):
    import os
    import sys

    from reread.evaluation.worker import configure_worker

    calls = []
    torch = SimpleNamespace(
        set_num_threads=lambda n: None,
        set_num_interop_threads=lambda n: None,
        cuda=SimpleNamespace(
            device_count=lambda: 8, set_device=lambda n: calls.append(n)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.chdir(tmp_path)
    config = dict(
        cuda_visible_devices="3",
        output_root=str(tmp_path),
        sim_root=str(tmp_path),
        worker_id=0,
        gpu_index=3,
        cpu_threads=10,
        cpu_affinity=[],
    )
    with pytest.raises(RuntimeError, match="exactly one GPU"):
        configure_worker(config)
    assert calls == []


def test_runtime_instruction_sampler_uses_accepted_seed_and_requested_25_count():
    from reread.evaluation.worker import RobotWinRuntime

    runtime = object.__new__(RobotWinRuntime)
    generator = object()
    calls = []
    runtime.instructions = SimpleNamespace(generate_episode_descriptions=generator)
    runtime.protocol = SimpleNamespace(
        sample_instruction=lambda *a, **k: calls.append((a, k)) or ("prompt", {})
    )
    result = runtime.sample_instruction(
        "turn_switch", {"a": "left"}, "unseen", seed=100075
    )
    assert result == ("prompt", {})
    assert calls == [
        (
            (generator, "turn_switch", {"a": "left"}, "unseen"),
            {"seed": 100075, "description_count": 25},
        )
    ]


def test_bundled_official_instruction_helpers_integrate_without_simulator_or_gpu():
    import importlib.util

    from reread.evaluation.worker import RobotWinRuntime

    root = Path(__file__).resolve().parents[1] / "third_party/RoboTwin"

    def module(name, relative):
        spec = importlib.util.spec_from_file_location(name, root / relative)
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result

    runtime = object.__new__(RobotWinRuntime)
    runtime.protocol = module("worker_test_protocol", "script/eval_protocol.py")
    runtime.instructions = module(
        "worker_test_instructions", "description/utils/generate_episode_instructions.py"
    )

    infos = {
        "open_microwave": {"{A}": "044_microwave/base0", "{a}": "left"},
        "hanging_mug": {"{A}": "039_mug/base0", "{B}": "040_rack/base0"},
        "turn_switch": {"{A}": "056_switch/base0", "{a}": "left"},
        "place_can_basket": {
            "{A}": "071_can/base0",
            "{B}": "110_basket/base0",
            "{a}": "left",
        },
        "move_stapler_pad": {"{A}": "048_stapler/base0", "{B}": "red", "{a}": "left"},
        "stack_bowls_three": {"{A}": "002_bowl/base3"},
        "pick_diverse_bottles": {"{A}": "001_bottle/base0", "{B}": "001_bottle/base1"},
        "place_mouse_pad": {"{A}": "047_mouse/base0", "{B}": "red", "{a}": "left"},
        "handover_block": {},
        "place_object_basket": {
            "{A}": "057_toycar/base0",
            "{B}": "110_basket/base0",
            "{a}": "left",
            "{b}": "right",
        },
        "move_can_pot": {
            "{A}": "060_kitchenpot/base0",
            "{B}": "105_sauce-can/base0",
            "{a}": "left",
        },
    }
    from reread.evaluation.scheduler import SELECTED_TASKS

    assert set(infos) == set(SELECTED_TASKS)
    for task, info in infos.items():
        for kind in ("seen", "unseen"):
            instruction, audit = runtime.sample_instruction(
                task, info, kind, seed=100075
            )
            assert instruction and instruction != task, (task, kind, audit)
            assert audit["description_count"] == 25
            assert audit["instruction_seed"] == 100075
            assert audit["instruction_source"] == "official_" + kind
            assert audit["instruction_generation_error"] is None
            random.seed(812)
            assert (
                runtime.sample_instruction(task, info, kind, seed=100075)[0]
                == instruction
            )


@pytest.mark.parametrize(
    "audit",
    [
        {"instruction_source": "official_task_name_fallback"},
        {
            "instruction_source": "official_seen",
            "instruction_generation_error": "FileNotFoundError: language assets",
        },
        {"instruction_source": "official_unseen"},
    ],
)
def test_instruction_fallback_or_wrong_bank_aborts_without_policy_score(
    config, job, audit
):
    runtime = Runtime()
    runtime.sample_instruction = lambda *a, **k: ("turn_switch", audit)
    with pytest.raises(RuntimeError, match="refusing task-name fallback"):
        evaluate_episode(job, runtime, config)
    assert runtime.resets == 0
    assert not list(Path(config["output_root"]).rglob("result.json"))
    assert list(Path(config["output_root"]).rglob("instruction_audit.json"))


@pytest.mark.parametrize(
    "error",
    [
        AttributeError("missing CUDA API"),
        NameError("undefined helper"),
        TypeError("unsupported runtime API argument"),
    ],
)
def test_deployment_programming_errors_are_not_rejected_seeds(error):
    from reread.evaluation.worker import is_expert_infrastructure_error

    assert is_expert_infrastructure_error(error)


def test_worker_and_policy_direct_project_imports_exist_in_bundled_source():
    import ast

    root = Path(__file__).resolve().parents[1]
    source_roots = {
        "reread": root / "src",
        "fastwam": root / "third_party/fastwam/src",
        "fastwam_policy": root / "third_party/fastwam/experiments/robotwin",
    }
    paths = [
        root / "src/reread/evaluation/worker.py",
        root / "experiments/robotwin/reread_policy/deploy_policy.py",
        root
        / "third_party/fastwam/experiments/robotwin/fastwam_policy/deploy_policy.py",
    ]
    checked = []
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            else:
                continue
            for name in modules:
                base = source_roots.get(name.split(".")[0])
                if base is None:
                    continue
                module = base.joinpath(*name.split("."))
                assert (
                    module.with_suffix(".py").is_file()
                    or (module / "__init__.py").is_file()
                ), (path, name)
                checked.append(name)
    assert "fastwam.datasets.lerobot.processors.fastwam_processor" in checked
    assert "reread.policy" in checked


def test_official_attention_function_imports_and_executes_on_cpu(monkeypatch):
    import importlib.util

    import torch

    root = Path(__file__).resolve().parents[1]
    upstream = root / "third_party/fastwam/src"
    monkeypatch.syspath_prepend(str(upstream))
    source = upstream / "fastwam/models/wan22/wan_video_dit.py"
    spec = importlib.util.spec_from_file_location(
        "worker_cpu_official_attention", source
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tensor = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8) / 10
    result = module.flash_attention(tensor, tensor, tensor, num_heads=2)
    assert result.shape == tensor.shape
    assert result.device.type == "cpu" and torch.isfinite(result).all()
