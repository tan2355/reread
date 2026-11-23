# ReRead

ReRead corrects a frozen Fast-WAM policy by re-reading its native K/V. The
corrector uses the current and two preceding decisions to predict an action
residual. This package covers observation caching, training, and closed-loop
RoboTwin evaluation.

Model weights are available on [Hugging Face](https://huggingface.co/dihabso/ReRead/tree/main).

Training samples all 50 RoboTwin 2.0 tasks uniformly (1/50 each), with a
clean:randomized decision ratio of 1:10. Trajectories advance chronologically.
Clean scenes use seen instructions in both training and closed-loop evaluation.
For randomized scenes, training uses both seen and unseen instructions, and
closed-loop evaluation uses unseen instructions.

## Quick start

Use Python 3.12 and install PyTorch for your machine. From this directory:

~~~bash
python -m pip install -r requirements.txt
python examples/minimal.py
python tools/run_tests.py
~~~

The example runs a small CPU forward/backward pass on synthetic inputs.

~~~python
from reread import ReRead, ReReadConfig

config = ReReadConfig()
model = ReRead(config)
~~~

The default corrector has 198,523,406 trainable parameters.
See [implementation](docs/implementation.md) for tensor shapes and
[reproduction](docs/reproduce.md) for assets, environment, training, and evaluation.

Evaluation defaults to Hard11, with 25 clean and 25 randomized episodes per task.
To evaluate all 50 tasks with 100 clean and 100 randomized episodes each, run
the following command on each evaluation node with the same run ID and output root.

~~~bash
python scripts/evaluate.py --task-set all50 \
  --run-id all50_eval --output-root /data/eval \
  --corrector /weights/reread_run/endpoint_step_20000.pt \
  --checkpoint /weights/robotwin_uncond_3cam_384.pt \
  --stats /weights/robotwin_uncond_3cam_384_dataset_stats.json \
  --wan /weights/Wan2.2-TI2V-5B --assets /data/robotwin_assets \
  --runtime /opt/robotwin_runtime --extra-deps /opt/robotwin_dependencies
~~~

## Source map

| Path | Purpose |
|---|---|
| src/reread/model.py | Action queries, memory reading, and residual output |
| src/reread/config.py | Model and execution settings |
| src/reread/loss.py | Valid-element action MSE |
| src/reread/engine.py | Training updates and causal memory commits |
| src/reread/memory/ | Per-camera pooling and history storage |
| src/reread/backbone/ | Frozen Fast-WAM loading and native K/V capture |
| src/reread/data/ | Observation caches and uniform task sampling |
| src/reread/train.py | Distributed training entry point |
| experiments/robotwin/reread_policy/ | RoboTwin policy adapter |
| scripts/evaluate.py | Hard11 and full 50-task evaluation |
| configs/reread.json | Default model configuration |
| tests/ | Synthetic model, data, and runtime regression tests |
| third_party/ | Required upstream sources and attribution |

The feature schema describes K/V from one decision. The three-decision window
is recorded separately in `memory_frames`. Earlier checkpoints remain compatible.
Weights, demonstrations, and simulator assets are supplied separately.

## Source verification

~~~bash
python tools/check.py
python tools/check_anonymity.py
~~~

After editing the source, regenerate its checksum manifest with
python tools/refresh_manifest.py. Create a source archive with
python tools/build_source.py ../reread_source.zip.

See LICENSE and THIRD_PARTY_NOTICES.md for licensing and upstream attribution.
