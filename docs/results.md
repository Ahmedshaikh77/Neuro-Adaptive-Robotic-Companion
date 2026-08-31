# Results and evidence record

The repository has one committed numeric result artifact: [`results/latency_power.csv`](../results/latency_power.csv). It contains compute latency and board-power rows only. The table below preserves that boundary rather than converting configuration constants or simulated output into measured model results.

## 1. Per-modality task accuracy

No committed accuracy artifact is available for the face-emotion, keyword-spotting, or gesture task. The repository contains training and evaluation entry points, but no retained dataset version, trained checkpoint, test output, confusion matrix, or per-class report from which a task-accuracy result can be reported.

The source constants audio reliability `0.906`, face reliability `0.635`, and gesture reliability `0.50` are configuration values used by the gate. They are not validated metrics in the current repository. In particular, the gesture value is marked as a placeholder in source. Do not use any of these values as a current measured result.

## 2. Compute latency and board power

**Committed provenance:** [the 12-row CSV](../results/latency_power.csv) records three compute paths across power modes `0` through `3`. Its source path uses dummy inputs; audio and gesture were run without checkpoints, and the face modality is explicitly named `face_mediapipe_resnet18_randweights`. The CSV contains no Jetson identity, run date, sample count, checkpoint hash, or power-sampling metadata. These measurements therefore do not establish task accuracy, response quality, or end-to-end companion performance.

| Modality | Power mode | p50 latency (ms) | p95 latency (ms) | Mean latency (ms) | Average board power (W) | Peak board power (W) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| audio | 0 | 7.473 | 18.150 | 9.745 | 5.532 | 6.259 |
| gesture | 0 | 2.314 | 3.735 | 2.487 | 6.546 | 6.658 |
| face_mediapipe_resnet18_randweights | 0 | 19.852 | 22.663 | 19.796 | 7.809 | 8.133 |
| audio | 1 | 9.709 | 34.665 | 14.440 | 5.922 | 7.028 |
| gesture | 1 | 3.044 | 5.647 | 3.398 | 7.028 | 7.495 |
| face_mediapipe_resnet18_randweights | 1 | 18.510 | 21.156 | 19.001 | 7.860 | 8.200 |
| audio | 2 | 5.895 | 30.817 | 10.051 | 7.486 | 7.974 |
| gesture | 2 | 1.929 | 2.065 | 1.960 | 7.539 | 7.575 |
| face_mediapipe_resnet18_randweights | 2 | 15.552 | 17.373 | 15.829 | 8.867 | 9.275 |
| audio | 3 | 10.803 | 57.479 | 18.850 | 4.853 | 5.462 |
| gesture | 3 | 3.193 | 3.515 | 3.283 | 5.296 | 5.382 |
| face_mediapipe_resnet18_randweights | 3 | 25.243 | 40.280 | 28.251 | 6.343 | 6.589 |

## 3. Real adaptive-strategy performance

No committed real adaptive-strategy result is available. `src/eval_strategies.py` contains a harness for `single`, `all_on`, and `adaptive`, but it requires paired, real events in a shared command vocabulary. The repository has no such retained event set, trial pairing, strategy summary, degraded-condition result, or raw comparison artifact.

For the command task, only audio and gesture are mapped to the shared command vocabulary. Face emotion does not vote on command intent, so no facial-emotion result should be interpreted as a command-strategy outcome.

## 4. Simulation

No simulation output artifact is committed. `selftest.py` and `src/demo_adaptive.py` can generate seeded synthetic events and demonstrate gate behavior, including fault-injection conditions. Those scripts test the implementation and configured simulated modality behavior; they are not a substitute for real model evaluation, paired strategy trials, or a board-power run.

## Future result-record schema

Every future result release should retain the following fields alongside the raw artifact.

| Field | Record requirement |
| --- | --- |
| Dataset | Dataset name, source, version, license status, split definition, and split identifiers or hashes |
| Checkpoint | Checkpoint file hash, training configuration, and whether weights are trained, random, or absent |
| Preprocessing | Input modality, shapes, sample rate or frame count, transforms, and normalization |
| Reproducibility | Random seed, software environment, date, and repository commit SHA |
| Calibration | Calibration method, held-out data, and confidence metrics when applicable |
| Subgroup and bias analysis | Defined subgroups, inclusion/exclusion criteria, coverage, results, and limitations |
| Adaptive trial design | Shared command vocabulary, paired-event definition, condition assignment, and trial pairing across strategies |
| Latency and power environment | Device identity, JetPack, CUDA/cuDNN, package versions, power mode, clock state, sample interval, warmups, run count, and input shapes |
| Raw evidence | Unmodified metric output, latency/power samples or CSV, confusion matrix, logs, and artifact locations |

Use the [reproduction guide](reproduction.md) to collect the required workflows and metadata. The README's survey links are not a source of performance results; their accessibility and governance status remain unestablished here.
