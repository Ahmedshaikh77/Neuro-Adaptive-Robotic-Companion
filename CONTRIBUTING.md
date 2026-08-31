# Contributing to NeuroBot

Thank you for helping improve NeuroBot. This repository is a research and educational prototype for adaptive multimodal companion interaction. It is not a therapist, clinical tool, diagnostic system, crisis service, or replacement for professional care.

## Before you open an issue or pull request

Describe the scope of the proposed change and its intended boundary. For a bug report, feature request, benchmark, model, data, or workflow change, provide the information that applies:

- Operating system, Python version, and relevant dependency versions.
- Execution mode, such as source-only self-test, webcam-only demo, voice session, simulated adaptive gate, training, evaluation, or Jetson benchmark.
- Hardware and permissions used when relevant, such as device model, accelerator, webcam, microphone, Jetson model, JetPack, power mode, and `tegrastats` availability. Do not add hardware details for a source-only or simulated workflow that does not use hardware.
- Dataset and checkpoint provenance, including source, version, license status, split or hash, checkpoint hash, and whether weights are trained, random, absent, or synthetic.
- Whether the change uses Google Speech Recognition, OpenAI, Duke Qualtrics, or another external provider, and what data crosses that boundary.
- Privacy, bias or fairness, mental-health safety, and human-participant implications when the change affects those areas.

Use the reproduction and results guides when reporting measurements. Metrics require retained, reviewable evidence. Configuration values, synthetic output, and unreviewed claims are not measured results.

## Privacy and sensitive material

Do not include credentials, API keys, tokens, passwords, participant records, raw conversations, transcripts, identifiable audio or video, faces, voices, survey data, or other personal data in issues, pull requests, commits, logs, screenshots, or attachments. Redact logs and reproduce problems with synthetic or authorized de-identified inputs where possible.

Do not make clinical, diagnostic, therapeutic, crisis-response, HIPAA, or participant-safety claims unless they have undergone the appropriate review and have supporting evidence. Facial-affect outputs and configured valence or arousal values must not be presented as a person's confirmed internal state, diagnosis, intent, or wellbeing.

## Human participants and external services

Any activity involving participants, surveys, recordings, interactions, or collection of personal data requires an appropriate institutional determination before it begins. Where required, use an approved protocol and meaningful informed-consent process that discloses the system, data handling, risks, external providers, and retention or deletion plan. Public-dataset work and device-only compute benchmarking do not determine the status of a participant-facing activity.

Google Speech Recognition can receive microphone audio. The optional OpenAI path can receive recognized text, recent conversation history, and inferred emotion context. The repository links to Duke Qualtrics surveys. Contributors must disclose changes to these boundaries and must not assume this repository supplies provider retention controls, consent, or institutional approval.

## Pull request expectations

Keep changes focused and include tests or a clear explanation when tests are not applicable. State how the change was exercised, identify evidence for any reported metric, and update documentation, model cards, data cards, or safety language when the change makes that necessary. Review the pull request template before submitting.
