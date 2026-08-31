# NeuroBot: Neuro-Adaptive Robotic Companion

NeuroBot is a human-robot interaction and embedded-perception prototype that combines local facial-affect inference, voice interaction, and adaptive multimodal sensing under practical latency and power constraints.

![NeuroBot companion prototype illustration](assets/NeuroBot.png)

## Prototype status

| Area | Status | What the repository establishes | Current boundary |
| --- | --- | --- | --- |
| Companion path | Implemented | Webcam face detection, ResNet18 FER-style inference, five-frame smoothing, optional conversation, speech output, and session logging are implemented in source. | The face checkpoint is not committed; this is not a validated clinical or participant-facing system. |
| Adaptive sensing path | Implemented | Audio keyword spotting, gesture, and face modalities can run through `AdaptiveGate` in `single`, `all_on`, or `adaptive` modes with latency and power budgets. | The runnable demo and self-test use simulated events, not saved real multimodal trials. |
| Compute-cost artifact | Hardware-measured compute cost | `results/latency_power.csv` records latency and board-power columns for modality compute paths across Jetson power modes. | Its benchmark inputs and model states do not support an accuracy, utility, or end-to-end adaptive-performance claim. |
| Model and strategy performance | Pending validation | Training, face evaluation, and strategy-evaluation harnesses are present. | No committed checkpoints, model metrics, calibrated subgroup results, or paired strategy comparisons are available. |

**Scope.** NeuroBot is a research and educational prototype, not a therapist, clinical tool, diagnostic system, crisis service, or replacement for professional care.

## Engineering objective

The engineering objective is to explore adaptive multimodal sensing for companion interaction: preserve useful context while choosing sensing work under latency and power constraints rather than assuming every modality must run on every interaction.

## What I built

- A local companion pipeline that detects a face with MediaPipe, classifies a FER-2013-style emotion label with ResNet18, smooths labels across five frames, derives configured valence/arousal proxies, and can use that context in a conversation turn.
- Voice input and output plumbing: microphone speech recognition, optional OpenAI conversation generation with local fallback responses, platform text-to-speech, and plaintext session logging.
- An adaptive-sensing extension with keyword-spotting, gesture, and face modality adapters; `AdaptiveGate`; late fusion; fault-injection helpers; and a Jetson-oriented latency and power harness.
- Training and evaluation entry points for face, keyword spotting, and gesture models. The data, trained checkpoints, and resulting evaluation artifacts are not committed.

## Architecture and data flow

See [architecture notes](docs/architecture.md) for the intended component-level view.

### Companion path

1. A webcam frame is captured and MediaPipe performs face detection.
2. The face crop is preprocessed and passed to a ResNet18 classifier trained for FER-2013-style labels.
3. A five-frame majority smoother produces an emotion label and configured valence/arousal context.
4. In the voice session, recognized speech and optional emotion context are provided to the conversation service when configured.
5. Text-to-speech returns the response, while emotion and conversation session records are written locally.

The webcam image is processed locally in this path. The implementation uses an inferred facial label and hard-coded valence/arousal mapping; it does not establish a person's internal state.

### Adaptive sensing path

`AdaptiveGate` can invoke audio keyword spotting, gesture recognition, and the face modality under `single`, `all_on`, or cost-aware `adaptive` modes. The gate uses configured modality costs, reliability, confidence, and latency/power budgets.

The targets are deliberately not all the same. In [`src/eval_strategies.py`](src/eval_strategies.py), only audio and gesture labels are mapped into the shared command-intent vote. Face emotion is excluded from that vote: it contributes emotion context and compute-cost profiling, not command intent. Facial smoothing in the companion path is also separate from late fusion in the adaptive path.

## Quickstart and reproduction boundaries

Read [reproduction notes](docs/reproduction.md) before running a workflow. `requirements.txt` specifies lower version bounds, not a fully pinned or fully tested environment. Install dependencies in an isolated environment appropriate to your platform, and expect platform-specific setup for PyAudio, camera access, and Jetson packages.

| Mode | Command | Datasets and checkpoints | Device, service, and credential boundary |
| --- | --- | --- | --- |
| Source-only self-test | `python selftest.py` | No dataset or checkpoint. It uses simulated modalities and synthetic events. | No camera, microphone, cloud service, credential, or Jetson is required. |
| Webcam-only emotion demo | `python -m src.demo_webcam_only` | Requires a locally available FER face checkpoint at `artifacts/best_fer_resnet.pt`; no inference dataset is needed. | Requires a webcam and local camera permission. No cloud service is used by this demo. |
| Voice and optional cloud conversation demo | `python -m src.demo_voice_session` | Requires the same local face checkpoint; no inference dataset is needed. | Requires webcam and microphone access. Google Speech Recognition processes microphone audio. An `OPENAI_API_KEY` and network access are required only for the OpenAI conversation path; otherwise the code uses local fallback replies. |
| Simulated adaptive-gate demo | `python -m src.demo_adaptive` | No dataset or checkpoint. The modality behavior and comparison values are simulated. | No camera, microphone, cloud service, credential, or Jetson is required. |
| Jetson compute benchmark | `python -m src.eval_strategies --face artifacts/best_fer_resnet.pt --kws artifacts/kws.pt --gesture artifacts/gesture.pt --bench-n 200` | The command expects local checkpoints when supplied. The committed CSV instead reflects dummy-input compute measurement with audio and gesture uncheckpointed and face labeled random weights. | Requires a Jetson environment and `tegrastats` for board-power sampling. Install Jetson-compatible `torch`, `torchvision`, and `torchaudio` through the NVIDIA JetPack path rather than assuming PyPI wheels are suitable. |

### Model training and evaluation workflows

These workflows create local data and artifacts that are absent from this checkout. They are not evidence of a result until their inputs, outputs, and evaluation provenance are retained.

| Workflow | Command | Required local inputs |
| --- | --- | --- |
| Train FER face model | `python -m src.train_fer --data-root data/archive --epochs 30` | A folder-based FER-2013-style train/test dataset under `data/archive`; output checkpoint is written under `artifacts/`. |
| Evaluate FER face model | `python -m src.eval_fer --data-root data/archive --checkpoint artifacts/best_fer_resnet.pt` | The same dataset and a trained face checkpoint. |
| Train keyword-spotting model | `python -m src.train_kws --epochs 20 --out artifacts/kws.pt` | torchaudio downloads Speech Commands on first use, approximately 2.3 GB; resulting checkpoint is local. |
| Train gesture model | `python -m src.train_gesture --data-root data_gesture --epochs 15 --out artifacts/gesture.pt` | A Jester-style train/test gesture folder tree and its local checkpoint output. |
| Evaluate strategies on a shared task | Use `src/eval_strategies.py` after defining paired multimodal command events. | Aligned, real audio and gesture inputs with shared command labels; this repository has no such committed evaluation set or saved comparison. |

## Results and evidence

See [results notes](docs/results.md) for provenance and reporting boundaries.

`results/latency_power.csv` is the only committed result artifact. Its 12 rows report p50, p95, and mean latency plus average and peak board-power measurements across four Jetson power modes for audio, gesture, and face compute paths.

The artifact is compute-cost evidence only. The benchmark code uses dummy inputs; audio and gesture use no checkpoint, and the face path is labeled random weights. It therefore does not measure task accuracy or end-to-end adaptive performance.

There are no committed model checkpoints, confusion matrices, raw trials, calibrated subgroup results, or saved real `single`, `all_on`, and `adaptive` comparisons. The previous README's FER range of approximately 60 to 70 percent was a general expectation, not a measured project result, and is not reported here as repository accuracy.

## Privacy, safety, and limitations

### Data handling and privacy boundaries

- Plaintext session logs can contain timestamps, inferred emotion labels, confidence, valence/arousal values, and full user and assistant text.
- Microphone audio is sent to Google Speech Recognition for transcription. When the OpenAI path is configured, recognized text, recent transcript history, and inferred emotional context can be sent to OpenAI. The implementation does not send the raw webcam frame to OpenAI.
- Study survey responses use the external [Qualtrics pre-survey](https://duke.qualtrics.com/jfe/form/SV_emJ6kDeQnsUTdqe) and [Qualtrics post-survey](https://duke.qualtrics.com/jfe/form/SV_bq6jOtY851yhQgK) links. Access requirements for those links may apply.
- The repository does not establish encryption, access control, retention or deletion schedules, export controls, pseudonymization, bystander consent, a minors policy, or provider-specific retention guarantees for any of these paths.

For any participant interaction, use persistent AI disclosure, obtain meaningful consent, minimize collected data, provide local emergency resources appropriate to the participant, and use supervised research procedures.

### Safety and inference limitations

- FER-2013-style facial labels can be affected by label uncertainty, lighting, pose, cultural and demographic bias, neurodiversity, and other factors. Facial affect inference can be overinterpreted and must not be treated as a reliable account of emotion, intent, or wellbeing.
- The crisis guardrail is an English substring keyword check. It can miss context, other languages, misspellings, euphemisms, and escalation, and it can also false-trigger. It is not a crisis assessment.
- The system provides no continuous monitoring, human escalation, location awareness, clinical validation, HIPAA compliance claim, or emergency-response guarantee. It must not be described as diagnosing or treating a condition.
- If someone may be in immediate danger, contact local emergency services. The code's U.S.-centric resource text is not a substitute for local, current, or professional guidance.

### Human-subjects governance

Institutional human-subjects review depends on the activity. Public-dataset work and device-only compute benchmarking are not the same category as participant surveys or logged interactions. `RUNBOOK.md` currently makes a broad statement that public datasets and an owned device need no IRB; that unresolved documentation inconsistency is preserved outside this documentation phase's allowlist and should not be used to decide a study's review requirements.

## Repository layout

```text
assets/                 # existing NeuroBot image
results/                # committed latency/power CSV
src/                    # companion, adaptive sensing, training, and evaluation modules
selftest.py             # source-only simulated self-test
requirements.txt        # dependency lower bounds
RUNBOOK.md              # existing operational notes, including the unresolved IRB wording
```

## License

This repository contains an [MIT License](LICENSE). The MIT grant is not limited to educational or research use. Non-clinical intended-use warnings above describe responsible deployment boundaries; they do not narrow the copyright license.

## Roadmap

All items below are future work suitable for scoped contributor contributions:

- Pin and test reproducible environments, including Jetson and desktop metadata.
- Add model cards, data cards, dataset split provenance, checkpoint hashes, and retained evaluation artifacts.
- Publish confusion matrices, subgroup evaluation, calibration analysis, and documented limitations for each trained model.
- Define a shared command task and retain real paired multimodal trials for `single`, `all_on`, and `adaptive` comparisons.
- Record benchmark device identity, power-mode settings, sample counts, model states, and power-monitor availability with each compute run.
- Add encrypted and minimized logging, retention/deletion controls, export controls, and consent controls.
- Provide localized crisis resources and a reviewed human-subjects protocol before participant-facing research.

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidance.

## Acknowledgments

- FER-2013 dataset
- MediaPipe for face detection
- PyTorch and torchvision for modeling
- OpenAI, when the optional conversation path is configured
- SpeechRecognition, Google Speech Recognition, and pyttsx3 for voice input and output
- Qualtrics for the linked study surveys
