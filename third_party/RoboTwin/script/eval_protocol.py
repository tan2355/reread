import hashlib
import json
from pathlib import Path
import random

import numpy as np

DESCRIPTION_COUNT = 100
REPLAY_SCHEMA = 2
OFFICIAL_PROTOCOL = {
    "repository": "https://github.com/RoboTwin-Platform/RoboTwin",
    "commit": "ea8b21121ebb3cd201ff5b3fe361944ac94eda3f",
    "path": "scripts/eval_policy_xpolicylab.py",
    "source_sha256": "75c4f20285936abcfc9163bdab881479852f24012b7086baaaba90db4aa1cf28",
    "description_generator_sha256": "ea469d08a1cccee700a153d4ab4742c862f7e9ac5b78821fcb0fb42d3134875a",
    "mode": "sequential_expert_screened",
}


def protocol_source_identity(root, task, phase):
    root = Path(root)
    paths = [root / "envs" / (task + ".py"), root / "envs/_base_task.py",
             root / "description/utils/generate_episode_instructions.py",
             root / "script/eval_protocol.py", root / "script/eval_policy.py",
             root / "script/official_protocol_reference.json"]
    paths += sorted((root / "task_config").glob("*.yml"))
    paths += sorted((root / "description").rglob("*.json"))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def rng_state():
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=[
        state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])])


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def write_record(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(jsonable(record), indent=2, allow_nan=False))
    tmp.replace(path)


def sample_instruction(generate, task, episode_info, instruction_type, *, seed,
                       description_count=DESCRIPTION_COUNT):
    if description_count <= 0:
        raise ValueError("description_count must be positive")
    before = rng_state()
    sampled = task
    source = "official_task_name_fallback"
    generation_error = None
    if instruction_type:
        try:
            results = generate(task, [episode_info], description_count, seed=seed)
            candidates = results[0].get(instruction_type)
            if candidates:
                sampled = random.Random(f"{seed}:select").choice(candidates)
                source = "official_" + instruction_type
        except Exception as error:
            generation_error = f"{type(error).__name__}: {error}"
            print("Failed to generate episode instruction; using task name as instruction.")
    after = rng_state()
    return sampled, dict(description_count=description_count,
        instruction_type=instruction_type, instruction_seed=int(seed), official_protocol=OFFICIAL_PROTOCOL,
        instruction_source=source,
        instruction_generation_error=generation_error,
        instruction=sampled, sampled_instruction=sampled, episode_info=episode_info,
        before_instruction=before, after_instruction=after)


def observation_signature(observation):
    arrays = {}
    def walk(value, key):
        if isinstance(value, np.ndarray):
            arr = np.ascontiguousarray(value)
            arrays[key] = dict(shape=list(arr.shape), dtype=str(arr.dtype),
                              sha256=hashlib.sha256(arr.tobytes()).hexdigest())
        elif isinstance(value, dict):
            for name, item in value.items():
                walk(item, key + "/" + str(name))
    walk(observation, "")
    return dict(arrays=arrays,
                state=jsonable({k: observation[k] for k in ("joint_action", "endpose") if k in observation}))
