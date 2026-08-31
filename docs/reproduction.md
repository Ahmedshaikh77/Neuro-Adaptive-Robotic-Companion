# Reproduction guide

`requirements.txt` contains minimum dependency versions, not a pinned environment. Use an isolated environment, grant camera and microphone permissions only when a procedure needs them, and do not place credentials in tracked files. The workflows below are existing commands and code paths, not a claim that the uncommitted models or datasets are available in this checkout.

## 1. Source-only self-test

- **Requirements:** Python and NumPy. No dataset, checkpoint, camera, microphone, GPU, API access, or Jetson hardware is required.
- **Command:**

  ```bash
  python selftest.py
  ```

- **Expected output:** A printed sequence of `PASS` or `FAIL` checks for imports, simulated modalities, fusion, benchmark helpers, fault injection, and simulated smart-switch behavior, followed by an all-checks-passed summary when successful.
- **Data boundary:** Fully local. It creates seeded synthetic events and does not contact a cloud provider.
- **Cleanup:** It does not create session logs, checkpoints, or credentials.
- **Evidence status:** Functional path and simulation. It is a software smoke test, not model-quality or hardware-cost evidence.

## 2. Webcam-only emotion demo

- **Requirements:** A local face checkpoint at `artifacts/best_fer_resnet.pt`, a webcam, camera permission, and the local vision dependencies from `requirements.txt`. No inference dataset is required after the checkpoint exists.
- **Command:**

  ```bash
  python -m src.demo_webcam_only
  ```

- **Expected output:** A live webcam window with face/emotion display; press `q` to quit. If the checkpoint is absent, the program prints its missing-model error and exits.
- **Data boundary:** Webcam frames are processed locally by the shown path. This demo has no Google or OpenAI call and writes no session log.
- **Cleanup:** Close the window with `q`, then remove any separately created local checkpoint only if it is no longer needed. Do not upload webcam material or a checkpoint unless its provenance permits it.
- **Evidence status:** Functional path. It does not establish emotion-recognition quality because neither checkpoint nor evaluation artifact is committed.

## 3. Voice session with optional cloud conversation

- **Requirements:** The local face checkpoint at `artifacts/best_fer_resnet.pt`; webcam and microphone access; PyAudio/SpeechRecognition and local vision dependencies. Google Speech Recognition network access is required for transcription. `OPENAI_API_KEY` and network access are required only for the cloud conversation branch.
- **Command:**

  ```bash
  python -m src.demo_voice_session
  ```

- **Expected output:** A webcam window and spoken companion replies. Each turn captures facial context, sends microphone audio to Google for transcription, and writes an emotion CSV and a conversation CSV. On exit it attempts to create a JSON session report. Without `OPENAI_API_KEY`, or after an OpenAI exception, the conversation manager uses its local fallback reply. A matching fixed crisis-keyword list returns the local crisis-resource reply before any cloud response.
- **Data boundary:** Raw microphone audio goes to Google Speech Recognition. On the configured OpenAI branch, recognized text, recent history, and optional emotion label/valence/arousal context go to OpenAI. The raw webcam frame is not sent to OpenAI by this code. Local artifacts are `logs/sessions/`, `logs/reports/`, and `conv_logs/`.
- **Cleanup:** After retaining only authorized evidence, delete sensitive local session CSV/JSON files from `logs/` and `conv_logs/`. Remove `OPENAI_API_KEY` from the shell or local `.env` file when finished; never commit it. Review provider-side controls separately because this repository does not manage them.
- **Evidence status:** Functional path. It is not a model-quality, privacy, or crisis-service evaluation.

## 4. Simulated adaptive-gate demo

- **Requirements:** Python and NumPy. No dataset, checkpoint, sensors, API access, or Jetson hardware is required.
- **Command:**

  ```bash
  python -m src.demo_adaptive
  ```

- **Expected output:** Printed Pareto-style comparison tables for `single`, `all_on`, and `adaptive` under simulated `clean` and `low_light` conditions.
- **Data boundary:** Fully local seeded synthetic events. No external provider is used and no result file is written.
- **Cleanup:** No credentials or session logs are created.
- **Evidence status:** Simulation. Its generated numbers describe `SimulatedModality` settings, not an observed real multimodal trial.

## 5. FER evaluation

- **Requirements:** A folder-based FER-2013-style dataset at `data/archive` with `train/` and `test/` label folders; a trained local checkpoint at `artifacts/best_fer_resnet.pt`; PyTorch, torchvision, scikit-learn, matplotlib, seaborn, and related local dependencies.
- **Command:**

  ```bash
  python -m src.eval_fer --data-root data/archive --checkpoint artifacts/best_fer_resnet.pt
  ```

- **Expected output:** Printed test accuracy and per-class report, plus `artifacts/confusion_matrix.npy` and `artifacts/confusion_matrix.png`.
- **Data boundary:** Dataset, checkpoint, evaluation, and generated confusion-matrix artifacts remain local in this command. No cloud provider is called.
- **Cleanup:** Treat the dataset, checkpoint, and generated confusion matrices as local research artifacts. Retain their license/provenance if reporting them; otherwise remove local copies that are no longer authorized or needed.
- **Evidence status:** Model-quality evaluation when run with retained dataset and checkpoint provenance. No corresponding committed output artifact exists in this repository.

## 6. Jetson compute benchmark

- **Requirements:** A Jetson environment with JetPack and Jetson-compatible `torch`, `torchvision`, and `torchaudio` installed through the NVIDIA JetPack path; remaining dependencies from `requirements.txt`; `tegrastats` available for board-power sampling; and local checkpoints for the checkpointed evaluator command. The committed benchmark CSV was instead produced from dummy inputs with audio and gesture uncheckpointed and a face path labelled random weights.
- **Existing evaluator command:**

  ```bash
  python -m src.eval_strategies --face artifacts/best_fer_resnet.pt --kws artifacts/kws.pt --gesture artifacts/gesture.pt --bench-n 200
  ```

- **Existing multi-mode driver commands:**

  ```bash
  sudo nvpmodel -m N && sudo jetson_clocks
  python3 bench_modes.py
  ```

- **Expected output:** The evaluator prints modality cost summaries and writes `latency_power.csv` in its `--out` directory, which defaults to `results`. The multi-mode driver detects the current `nvpmodel` mode, appends missing modality/mode rows to `results/latency_power.csv`, and prints the full CSV. `tegrastats` absence is reported as unavailable by the helper and yields zero sampled power rather than a board-power measurement.
- **Data boundary:** Benchmark inputs are local NumPy dummy waveform, frame-stack, and image values. No participant data or cloud provider is involved.
- **Cleanup:** Do not overwrite the committed CSV when collecting a new run without first copying it to a run-specific evidence location. Remove local benchmark outputs that should not be retained, and restore the device's intended `nvpmodel` and clock state after testing. The command does not use cloud credentials.
- **Evidence status:** Compute benchmark. It supports latency and board-power recording for the specific recorded environment, inputs, and model state, not task accuracy or adaptive-strategy quality.

### Required metadata for every future Jetson run

Record the Jetson model; JetPack; CUDA and cuDNN versions; Python and package versions; `nvpmodel` mode; `jetson_clocks` state; `tegrastats` sample interval; warmup count; run count; input shapes; checkpoint status and hash; date; repository commit SHA; and random seeds. Also retain the raw output CSV and state whether board-power samples were actually available.

## Cross-workflow notes

The adaptive command evaluator needs paired, real audio and gesture events mapped into the shared command vocabulary before `run_condition()` can report a real strategy comparison. Facial emotion is intentionally excluded from that command vote in [`src/eval_strategies.py`](../src/eval_strategies.py). The README's Duke Qualtrics pre- and post-survey links are preserved, but survey accessibility and governance are not established by these workflows.

For local source and result locations, see [architecture and data boundaries](architecture.md). For what the committed record does and does not show, see [results](results.md).
