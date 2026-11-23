# Third-party notices

The top-level Apache-2.0 license applies to original ReRead code. It does not
replace the licenses of the following bundled components.

| Component | Source | License files |
|---|---|---|
| Fast-WAM | https://github.com/yuantianyuan01/FastWAM | [third_party/fastwam/LICENSE](third_party/fastwam/LICENSE) |
| RoboTwin | https://github.com/RoboTwin-Platform/RoboTwin | [third_party/RoboTwin/LICENSE](third_party/RoboTwin/LICENSE) |

These component license files are MIT licenses. Preserve their copyright and
permission notices when redistributing their source or derived components.
Refer to the included files for the complete terms and attribution.

The bundled Fast-WAM files implement the frozen backbone and its RoboTwin
observation and action processing. The RoboTwin files provide the simulator,
tasks, and instructions for closed-loop evaluation.
Fast-WAM's source revision and file hashes are recorded in its
[upstream manifest](third_party/fastwam/UPSTREAM_MANIFEST.json).
The root [manifest.json](manifest.json) records the size and SHA-256 of every
distributed source file, including both components.

## Assets and installed dependencies

Fast-WAM checkpoints and normalization statistics, Wan encoder weights,
tokenizers, RoboTwin simulator assets, and training datasets are separate from
the original ReRead code. Obtain them from their publishers and follow the
terms applicable to the selected versions. This repository's Apache-2.0
license does not establish permission to redistribute those assets.

Python, PyTorch, CUDA/NCCL, and other installed packages retain their own
licenses.
