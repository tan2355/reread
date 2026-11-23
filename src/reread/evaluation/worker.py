from __future__ import annotations

import copy
import importlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from reread.evaluation.resource_lifecycle import (
    acknowledge_worker_exit,
    finalize_environment,
    memory_snapshot,
)


class EpisodeStageTiming:

    def __init__(self, clock=None):
        self.clock = time.perf_counter if clock is None else clock
        self.seconds = {}
        self.calls = {}
        self._active = None

    @contextmanager
    def measure(self, name):
        if self._active is not None:
            raise RuntimeError("Evaluation stage timing must remain exclusive")
        self._active = name
        started = self.clock()
        try:
            yield
        finally:
            elapsed = self.clock() - started
            self.seconds[name] = self.seconds.get(name, 0.0) + max(0.0, elapsed)
            self.calls[name] = self.calls.get(name, 0) + 1
            self._active = None

    def snapshot(self, elapsed_seconds):
        accounted = sum(self.seconds.values())
        return dict(
            schema="newm-evaluation-stage-timing-v1",
            clock="host_perf_counter_no_cuda_synchronize",
            exclusive=True,
            seconds=dict(self.seconds),
            calls=dict(self.calls),
            accounted_seconds=accounted,
            unattributed_seconds=max(0.0, elapsed_seconds - accounted),
            policy_note="rollout_policy_s includes legacy infer_s and sim_s; do not add them twice",
        )


def write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()


def is_expert_infrastructure_error(error):
    markers = (
        "cuda error",
        "cuda failure",
        "cuda out of memory",
        "cuda kernel build failed",
        "nvrtc_error",
        "cublas_status",
        "cudnn_status",
        "device-side assert",
        "vk_error",
        "failed to find vulkan",
        "failed to load vulkan",
        "cuda device changed",
        "renderer gpu mismatch",
        "renderer cuda ordinal mismatch",
        "error launching kernel",
        "trying to launch on device=",
        "device mismatch",
        "expected all tensors to be on the same device",
        "found at least two devices",
        "invalid device",
        "cudaerrorinvaliddevice",
        "cuda_error_invalid_device",
        "cuda driver error",
    )
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(
            error,
            (OSError, ImportError, MemoryError, AttributeError, NameError, TypeError),
        ):
            return True
        if any(marker in str(error).lower() for marker in markers):
            return True
        error = error.__cause__ or error.__context__
    return False


class FailureVideoBuffer:

    def __init__(self, width, height, max_frames):
        if min(width, height, max_frames) <= 0 or width % 2 or height % 2:
            raise ValueError(
                "Video dimensions must be positive/even, frame bound positive"
            )
        self.width, self.height = int(width), int(height)
        self.frame_bytes = self.width * self.height * 3
        self.max_frames = int(max_frames)
        self.frames = []
        self.stdin = self
        self.closed = False

    def write(self, data):
        if self.closed:
            raise ValueError("Video buffer is closed")
        if len(data) != self.frame_bytes:
            raise ValueError(
                f"Expected one RGB frame ({self.frame_bytes} bytes), got {len(data)}"
            )
        if len(self.frames) >= self.max_frames:
            raise MemoryError(
                "Video frame bound exceeded; refusing unbounded RAM growth"
            )
        self.frames.append(bytes(data))
        return len(data)

    def close(self):
        self.closed = True

    def wait(self):
        return 0

    def discard(self):
        self.frames.clear()
        self.closed = True

    def save_failure(self, path, *, threads=1, popen=subprocess.Popen):
        path = Path(path)
        if not self.frames:
            raise RuntimeError("Failed rollout has no video frames")
        if threads < 1:
            raise ValueError("ffmpeg threads must be positive")
        path.parent.mkdir(parents=True, exist_ok=True)
        incomplete = path.with_name(path.stem + ".incomplete.mp4")
        if path.exists() or incomplete.exists():
            raise FileExistsError(f"Refusing to overwrite video: {path}")
        command = [
            "ffmpeg",
            "-nostdin",
            "-n",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            "10",
            "-i",
            "-",
            "-an",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            "-preset",
            "veryfast",
            "-threads",
            str(threads),
            "-filter_threads",
            "1",
            "-movflags",
            "+faststart",
            str(incomplete),
        ]
        with path.with_suffix(".ffmpeg.log").open("xb") as error_log:
            process = popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=error_log,
            )
            try:
                for frame in self.frames:
                    process.stdin.write(frame)
                process.stdin.close()
                code = process.wait(timeout=180)
                if code:
                    raise RuntimeError(
                        f"ffmpeg failed with exit code {code}; see {error_log.name}"
                    )
            except BaseException:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=30)
                raise
        if not incomplete.is_file() or incomplete.stat().st_size == 0:
            raise RuntimeError("ffmpeg returned without a nonempty video")

        incomplete.rename(path)
        self.discard()
        return str(path)


def _safe_job_directory(output_root, job):
    task, phase = job["task"], job["phase"]
    if (
        not isinstance(task, str)
        or not task
        or any(c not in "abcdefghijklmnopqrstuvwxyz_0123456789" for c in task)
        or phase not in ("demo_clean", "demo_randomized")
    ):
        raise ValueError("Invalid task/phase")
    index = job["episode_index"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("Invalid episode index")
    directory = Path(output_root) / task / phase / f"episode_{index:03d}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def evaluate_episode(job, runtime, config, *, progress=None):
    directory = _safe_job_directory(config["output_root"], job)
    write_json_new(directory / "job.json", job)
    protocol = runtime.protocol
    if job.get("rng_state") is not None:
        raise ValueError(
            "Independent seed lanes must not inherit another episode RNG state"
        )

    random.seed(int(job.get("seed", 100000)))
    runtime.seed_numpy(int(job.get("seed", 100000)))
    env = observation = None
    episode_memory_start = memory_snapshot()
    started = time.perf_counter()
    stages = EpisodeStageTiming()
    seed = int(job["seed"])
    initial_seed = seed
    attempt_budget = int(job.get("max_seed_attempts", 2500))
    candidate_stride = int(job.get("candidate_stride", 25))
    if attempt_budget <= 0 or seed < 0 or candidate_stride < 1:
        raise ValueError(
            "Nonnegative seeds, positive stride and screening budget required"
        )
    seed_log = directory / "seed_attempts.jsonl"
    capture = None
    active = False
    cleanup_errors = []

    def cleanup_after_error(stage):
        nonlocal active
        if not active:
            return False
        active = False
        try:
            with stages.measure(
                "expert_cleanup_s" if stage.startswith("expert") else "cleanup_s"
            ):
                env.close_env()
        except Exception as cleanup_error:
            cleanup_errors.append(
                dict(
                    stage=stage,
                    seed=seed,
                    error_type=type(cleanup_error).__name__,
                    error=str(cleanup_error),
                    traceback=traceback.format_exc(),
                )
            )
            traceback.print_exc()
            return True
        return False

    try:
        env = runtime.new_environment(job["task"])
        args = copy.deepcopy(runtime.environment_args(job["task"], job["phase"]))
        args.update(eval_mode=True, eval_video_save_dir=None)
        env.suc = 0
        env.test_num = job["episode_index"]
        for attempt in range(1, attempt_budget + 1):
            before_expert = protocol.rng_state()
            screen_args = dict(args, render_freq=0)
            if progress:
                progress(dict(stage="expert", seed=seed, attempt=attempt))
            expert_closing = False
            try:
                active = True
                with stages.measure("expert_setup_s"):
                    env.setup_demo(
                        now_ep_num=job["episode_index"],
                        seed=seed,
                        is_test=True,
                        **screen_args,
                    )
                with stages.measure("expert_rollout_s"):
                    episode_info = env.play_once()
                active = False
                expert_closing = True
                with stages.measure("expert_cleanup_s"):
                    env.close_env()
                expert_closing = False
                with stages.measure("expert_check_s"):
                    accepted = bool(env.plan_success and env.check_success())
            except runtime.unstable_error as error:
                fatal = expert_closing or runtime.is_expert_infrastructure_error(error)
                append_json(
                    seed_log,
                    dict(
                        seed=seed,
                        status="infrastructure_error" if fatal else "unstable",
                        error_type=type(error).__name__,
                        error=str(error),
                        traceback=traceback.format_exc(),
                    ),
                )
                cleanup_failed = cleanup_after_error("expert_unstable")
                if fatal or cleanup_failed:
                    raise
                seed += candidate_stride
                continue
            except Exception as error:
                fatal = expert_closing or runtime.is_expert_infrastructure_error(error)
                append_json(
                    seed_log,
                    dict(
                        seed=seed,
                        status="infrastructure_error" if fatal else "expert_exception",
                        error_type=type(error).__name__,
                        error=str(error),
                        traceback=traceback.format_exc(),
                    ),
                )
                cleanup_failed = cleanup_after_error("expert_exception")
                if fatal or cleanup_failed:
                    raise
                seed += candidate_stride
                continue
            if not accepted:
                append_json(seed_log, dict(seed=seed, status="expert_rejected"))
                seed += candidate_stride
                continue
            break
        else:
            raise RuntimeError(
                "Seed screening budget exhausted; evaluation is incomplete"
            )

        before_rollout_setup = protocol.rng_state()
        active = True

        with stages.measure("rollout_setup_s"):
            env.setup_demo(
                now_ep_num=job["episode_index"], seed=seed, is_test=True, **args
            )
        instruction_type = "seen" if job["phase"] == "demo_clean" else "unseen"
        with stages.measure("instruction_s"):
            instruction, audit = runtime.sample_instruction(
                job["task"], episode_info.get("info", {}), instruction_type, seed=seed
            )
        write_json_new(directory / "instruction_audit.json", protocol.jsonable(audit))
        if (
            audit.get("instruction_source") != "official_" + instruction_type
            or audit.get("instruction_generation_error")
            or not isinstance(instruction, str)
            or not instruction.strip()
        ):
            raise RuntimeError(
                "Requested instruction bank was not sampled; refusing task-name fallback: "
                + str(
                    audit.get("instruction_generation_error")
                    or audit.get("instruction_source")
                )
            )
        replay = dict(
            schema=protocol.REPLAY_SCHEMA,
            task=job["task"],
            task_config=job["phase"],
            seed=seed,
            episode=job["episode_index"],
            before_expert=before_expert,
            before_rollout_setup=before_rollout_setup,
            protocol_source_sha256=runtime.protocol_identity(job["task"], job["phase"]),
            **audit,
        )
        env.set_instruction(instruction=instruction)
        width, height = runtime.video_dimensions(args)
        capture = FailureVideoBuffer(width, height, int(env.step_lim) + 1)
        env.eval_video_path = str(directory)
        env._set_eval_video_ffmpeg(capture)
        with stages.measure("model_reset_s"):
            runtime.reset_model()
        if progress:
            progress(
                dict(
                    stage="rollout", seed=seed, actions=0, step_limit=int(env.step_lim)
                )
            )
        success = False
        first_observation = True
        last_progress = 0
        while env.take_action_cnt < env.step_lim:
            with stages.measure("observation_s"):
                observation = env.get_obs()
            if first_observation:
                replay["initial_observation"] = protocol.observation_signature(
                    observation
                )
                protocol.write_record(directory / "replay.json", replay)
                first_observation = False
            before_actions = env.take_action_cnt
            with stages.measure("rollout_policy_s"):
                runtime.policy_step(env, observation)
            if env.take_action_cnt <= before_actions and not env.eval_success:
                raise RuntimeError("Policy step made no action progress")
            if progress and env.take_action_cnt - last_progress >= 50:
                last_progress = env.take_action_cnt
                progress(
                    dict(
                        stage="rollout",
                        seed=seed,
                        actions=int(last_progress),
                        step_limit=int(env.step_lim),
                    )
                )
            if env.eval_success:
                success = True
                break
        with stages.measure("video_finalize_s"):
            env._del_eval_video_ffmpeg()
        frames = len(capture.frames)
        video = None
        if success:
            with stages.measure("video_discard_s"):
                capture.discard()
        else:
            if progress:
                progress(dict(stage="failure_video", seed=seed, frames=frames))
            with stages.measure("video_encode_s"):
                video = capture.save_failure(
                    directory / f"seed_{seed}_failure.mp4",
                    threads=int(config.get("ffmpeg_threads", 1)),
                )
        record = dict(
            job_id=job["job_id"],
            task=job["task"],
            phase=job["phase"],
            episode_index=job["episode_index"],
            scene_episode_index=job["episode_index"],
            status="success" if success else "failure",
            success=success,
            seed=seed,
            initial_seed=initial_seed,
            seed_attempts=attempt,
            next_seed=seed + candidate_stride,
            candidate_stride=candidate_stride,
            evaluation_kind="official_expert_admission_strided_subset",
            actions=int(env.take_action_cnt),
            step_limit=int(env.step_lim),
            instruction=instruction,
            instruction_type=instruction_type,
            instruction_source=audit["instruction_source"],
            description_count=audit["description_count"],
            randomized_unseen_was_in_training=job["phase"] == "demo_randomized",
            video=video,
            captured_video_frames=frames,
            replay_record=str(directory / "replay.json"),
            worker_id=config["worker_id"],
            gpu_index=config["gpu_index"],
            slot=config["slot"],
            inference_seed=int(config["policy_args"]["seed"]),
        )
        record.update(runtime.timing())

        clear_every = int(args["clear_cache_freq"])
        active = False
        with stages.measure("cleanup_s"):
            env.close_env(clear_cache=((job["episode_index"] + 2) % clear_every == 0))
            if getattr(env, "render_freq", 0):
                env.viewer.close()
        record["rng_state"] = protocol.rng_state()
        record["elapsed_seconds"] = time.perf_counter() - started
        record["stage_timing"] = stages.snapshot(record["elapsed_seconds"])
        append_json(seed_log, dict(seed=seed, status="evaluated"))
        write_json_new(directory / "result.json", record)
        return record
    except BaseException as error:
        original_traceback = traceback.format_exc()
        cleanup_after_error("episode")
        elapsed_seconds = time.perf_counter() - started
        write_json_new(
            directory / "error.json",
            dict(
                job_id=job["job_id"],
                seed=seed,
                error_type=type(error).__name__,
                error=str(error),
                traceback=original_traceback,
                cleanup_errors=cleanup_errors,
                elapsed_seconds=elapsed_seconds,
                stage_timing=stages.snapshot(elapsed_seconds),
            ),
        )
        raise
    finally:
        original_error = sys.exc_info()[1]

        def drop_episode_references():
            nonlocal env, observation, capture
            old_capture = capture
            env = observation = capture = None
            if old_capture is not None:
                old_capture.discard()

        lifecycle = finalize_environment(
            env, drop_local_references=drop_episode_references
        )
        lifecycle.update(
            start=episode_memory_start,
            worker_id=config["worker_id"],
            worker_generation=config.get("worker_generation", 0),
            job_id=job["job_id"],
            original_episode_error=original_error is not None,
            timing_note="terminal resource cleanup is outside legacy result/stage timings",
        )
        try:
            write_json_new(directory / "resource_lifecycle.json", lifecycle)
        except Exception:
            if original_error is None:
                raise

            traceback.print_exc()
        if lifecycle["errors"] and original_error is None:
            raise RuntimeError(
                "Episode resource cleanup failed; see resource_lifecycle.json"
            )


class RobotWinRuntime:

    def __init__(self, config):
        self.config = config
        self.root = Path(config["sim_root"]).resolve()
        for path in (
            self.root,
            self.root / "policy",
            self.root / "script",
            self.root / "description/utils",
        ):
            sys.path.insert(0, str(path))
        self.official = importlib.import_module("eval_policy")
        self.protocol = importlib.import_module("eval_protocol")
        self.instructions = importlib.import_module("generate_episode_instructions")
        self.yaml = importlib.import_module("yaml")
        self.numpy = importlib.import_module("numpy")
        self.unstable_error = self.official.UnStableError
        self.is_expert_infrastructure_error = is_expert_infrastructure_error
        self.adapter = importlib.import_module("reread_policy.deploy_policy")
        self._identities = {}
        self._args = {}

        offload_enabled = config.get("text_encoder_offload", True)
        if type(offload_enabled) is not bool:
            raise ValueError("text_encoder_offload must be boolean")
        from reread.evaluation.text_offload import (
            gpu_text_lease,
            install_text_encoder_offload,
            text_lease_path,
        )

        with gpu_text_lease(text_lease_path(config)):
            self.model = self.adapter.get_model(dict(config["policy_args"]))
            self.text_offload = None
            if offload_enabled:
                self.text_offload = install_text_encoder_offload(self.model, config)

    def seed_numpy(self, seed):
        self.numpy.random.seed(seed)

    def new_environment(self, task):
        return self.official.class_decorator(task)

    def environment_args(self, task, phase):
        key = task, phase
        if key in self._args:
            return self._args[key]
        args = self.yaml.safe_load(
            (self.root / "task_config" / f"{phase}.yml").read_text()
        )
        args.update(
            task_name=task,
            task_config=phase,
            ckpt_setting=self.config["policy_args"]["ckpt_setting"],
            policy_name="reread_policy.deploy_policy",
            eval_video_log=True,
            eval_mode=True,
        )
        embodiments = self.yaml.safe_load(
            (self.root / "task_config/_embodiment_config.yml").read_text()
        )
        cameras = self.yaml.safe_load(
            (self.root / "task_config/_camera_config.yml").read_text()
        )
        camera = cameras[args["camera"]["head_camera_type"]]
        args.update(head_camera_h=camera["h"], head_camera_w=camera["w"])
        types = args["embodiment"]
        if len(types) == 1:
            left = right = embodiments[types[0]]["file_path"]
            args["dual_arm_embodied"] = True
        elif len(types) == 3:
            left, right = (embodiments[name]["file_path"] for name in types[:2])
            args.update(embodiment_dis=types[2], dual_arm_embodied=False)
        else:
            raise ValueError("Embodiment configuration must have one or three entries")
        args.update(
            left_robot_file=left,
            right_robot_file=right,
            left_embodiment_config=self.official.get_embodiment_config(left),
            right_embodiment_config=self.official.get_embodiment_config(right),
        )
        if int(args["clear_cache_freq"]) < 1:
            raise ValueError("clear_cache_freq must be positive")
        self._args[key] = args
        return args

    def protocol_identity(self, task, phase):
        key = task, phase
        if key not in self._identities:
            self._identities[key] = self.protocol.protocol_source_identity(
                self.root, task, phase
            )
        return self._identities[key]

    def sample_instruction(self, task, info, instruction_type, *, seed):
        return self.protocol.sample_instruction(
            self.instructions.generate_episode_descriptions,
            task,
            info,
            instruction_type,
            seed=seed,
            description_count=25,
        )

    def video_dimensions(self, args):
        return tuple(map(int, self.official.get_eval_video_size(args).split("x")))

    def reset_model(self):
        self.adapter.reset_model(self.model)

    def policy_step(self, env, observation):
        self.adapter.eval(env, self.model, observation)

    def timing(self):
        result = (
            dict(self.model.get_timing_rollout())
            if hasattr(self.model, "get_timing_rollout")
            else {}
        )
        if self.text_offload is not None:
            result["text_encoder_offload"] = self.text_offload.stats()
        return result


def configure_worker(config):
    expected_visibility = config.get("cuda_visible_devices")
    if (
        not expected_visibility
        or os.environ.get("CUDA_VISIBLE_DEVICES") != expected_visibility
    ):
        raise RuntimeError(
            "Worker CUDA visibility was not bound before importing the worker"
        )
    affinity = config.get("cpu_affinity")
    if affinity and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(map(int, affinity)))
    threads = int(config["cpu_threads"])
    if threads < 1:
        raise ValueError("cpu_threads must be positive")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = str(threads)
    for key in ("OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "MAX_JOBS"):
        os.environ[key] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    os.environ["NEWM_EVAL_CUDA_INDEX"] = "0"
    generation = config.get("worker_generation", 0)
    if type(generation) is not int or generation < 0:
        raise ValueError("worker_generation must be a nonnegative integer")
    worker_cache = f"worker_{config['worker_id']:02d}"
    if generation:
        worker_cache += f"_generation_{generation:03d}"
    cache = Path(config["output_root"]) / "runtime_cache" / worker_cache

    shared_extensions = (
        Path(config["output_root"])
        / "runtime_cache"
        / f"node_{config.get('node_rank', 0)}"
        / "shared_torch_extensions"
    )
    shared_extensions.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(shared_extensions)
    for key, name in (
        ("WARP_CACHE_PATH", "warp"),
        ("CUDA_CACHE_PATH", "cuda"),
        ("XDG_CACHE_HOME", "xdg"),
        ("MPLCONFIGDIR", "matplotlib"),
        ("TRITON_CACHE_DIR", "triton"),
        ("TORCHINDUCTOR_CACHE_DIR", "torch_inductor"),
    ):
        directory = cache / name
        directory.mkdir(parents=True, exist_ok=False)
        os.environ[key] = str(directory)
    if config.get("render_pci"):
        os.environ["FSWM_RENDER_PCI"] = config["render_pci"]
    os.chdir(config["sim_root"])
    import torch

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Worker CUDA visibility must expose exactly one GPU")
    torch.cuda.set_device(0)
    from reread.evaluation.runtime_setup import cuda_pci_bus_id, normalize_pci

    actual_pci = cuda_pci_bus_id(0)
    if not config.get("render_pci") or actual_pci != normalize_pci(
        config["render_pci"]
    ):
        raise RuntimeError("Worker runtime CUDA PCI differs from coordinator inventory")
    config["policy_args"]["device"] = "cuda:0"

    try:
        import cv2

        cv2.setNumThreads(1)
    except ImportError:
        pass


def worker_process(connection, config):
    current_job = None
    try:
        configure_worker(config)

        runtime = RobotWinRuntime(config)
        connection.send(
            dict(
                type="ready",
                worker_id=config["worker_id"],
                gpu_index=config["gpu_index"],
            )
        )
        while True:
            message = connection.recv()
            if message.get("type") == "stop":
                acknowledge_worker_exit(
                    connection, dict(type="stopped", worker_id=config["worker_id"])
                )
                return
            if message.get("type") != "job":
                raise ValueError("Expected job or stop message")
            current_job = message["job"]

            def progress(fields):
                connection.send(
                    dict(
                        type="progress",
                        worker_id=config["worker_id"],
                        job_id=current_job["job_id"],
                        **fields,
                    )
                )

            result = evaluate_episode(current_job, runtime, config, progress=progress)
            connection.send(
                dict(type="result", worker_id=config["worker_id"], result=result)
            )
            current_job = None
    except BaseException as error:
        record = dict(
            type="error",
            worker_id=config.get("worker_id"),
            job_id=current_job["job_id"] if current_job else None,
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        print(json.dumps(record), flush=True)
        try:
            connection.send(record)
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        connection.close()
