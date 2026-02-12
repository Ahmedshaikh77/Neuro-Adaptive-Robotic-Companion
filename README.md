# NeuroBot — Emotion Aware Neuro Adaptive Robotic Companion

NeuroBot is an **emotion-aware companion system** that combines real-time facial emotion recognition with a voice-based conversational agent to offer **supportive check-ins between therapy sessions**. It's built to be *helpful, transparent, and safety-first*—not a replacement for professional care.

> **Scope note:** This is a research/educational prototype focused on human-centered interaction, logging, and safe supportive dialogue.

---

##  Study Surveys (Pre / Post)

Please use the following Qualtrics links:

- **NeuroBot Pre Survey:** https://duke.qualtrics.com/jfe/form/SV_emJ6kDeQnsUTdqe
- **NeuroBot Post Survey:** https://duke.qualtrics.com/jfe/form/SV_bq6jOtY851yhQgK

*(If these links require Duke access, participants may need to be signed in.)*

---

##  Key Features

- **Emotion Classification:** ResNet18-based classifier trained on FER-2013
- **Real-time Inference:** MediaPipe face detection + webcam emotion recognition
- **Voice Experience:** Speech-to-Text (STT) + Text-to-Speech (TTS) for natural interaction
- **LLM Integration (Optional):** Empathetic, supportive responses with safety guardrails
- **Session Logging:** Emotion + conversation logs, plus structured JSON session reports
- **Safety First:** Crisis keyword detection with appropriate resource guidance

---

##  Safety & Ethics

NeuroBot is designed as a **supportive companion**:

-  Does **not** claim to be a therapist
-  Does **not** provide medical diagnoses or clinical advice
-  Includes **basic crisis detection** and encourages **professional help**
-  Logs sessions for evaluation and improvement (see `logs/`)

If you or someone else is in immediate danger, call local emergency services. In the U.S., you can also call/text **988** for the Suicide & Crisis Lifeline.

---

##  Project Structure

```text
final_project/
├── requirements.txt
├── README.md
├── artifacts/                 # model checkpoints + confusion matrices
├── logs/
│   ├── sessions/              # emotion CSV logs
│   └── reports/               # JSON session reports
├── conv_logs/                 # conversation CSV logs
└── src/
    ├── __init__.py
    ├── config.py              # configuration and paths
    ├── fer_dataset.py         # FER-2013 dataset loader
    ├── train_fer.py           # training script
    ├── eval_fer.py            # evaluation script
    ├── emotion_engine.py      # real-time emotion detection
    ├── voice_io.py            # STT and TTS
    ├── conversation_manager.py# LLM conversation management
    ├── session_logger.py      # logging and reporting
    ├── demo_webcam_only.py    # webcam-only demo
    └── demo_voice_session.py  # full voice session demo
```

---

##  Quick Start

### 1) Create a Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows
```

### 2) Install Dependencies

```bash
pip install -r requirements.txt
```

**PyAudio note (if needed):**

* **macOS:** `brew install portaudio` then `pip install pyaudio`
* **Ubuntu/Debian:** `sudo apt-get install portaudio19-dev` then `pip install pyaudio`
* **Windows:** install a pre-built wheel (common approach)

### 3) Prepare FER-2013 Dataset

The system expects a folder-based dataset structure:

```text
data/
  archive/
    train/
      angry/*.jpg
      disgust/*.jpg
      fear/*.jpg
      happy/*.jpg
      neutral/*.jpg
      sad/*.jpg
      surprise/*.jpg
    test/
      angry/*.jpg
      disgust/*.jpg
      fear/*.jpg
      happy/*.jpg
      neutral/*.jpg
      sad/*.jpg
      surprise/*.jpg
```

You can download FER-2013 from Kaggle and organize it into this structure (or use a similar dataset with the same folder layout).

### 4) (Optional) Set Up OpenAI API Key

For LLM-powered conversations, set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

If no API key is provided, the system will use fallback responses.

---

##  Usage

### 1) Train the Emotion Classifier

```bash
# Quick test (1 epoch)
python -m src.train_fer --data-root data/archive --epochs 1 --batch-size 64

# Full training (30 epochs - recommended)
python -m src.train_fer --data-root data/archive --epochs 30 --batch-size 64 --lr 0.001
```

**Training Output**

* Model saved to: `artifacts/best_fer_resnet.pt`
* Best validation accuracy printed at completion

**Expected Performance**

* FER-2013 is challenging; ~60–70% after ~30 epochs is common.

### 2) Evaluate the Model

```bash
python -m src.eval_fer --data-root data/archive --checkpoint artifacts/best_fer_resnet.pt
```

**Evaluation Output**

* Test accuracy + per-class metrics
* Confusion matrix saved as:

  * `artifacts/confusion_matrix.npy`
  * `artifacts/confusion_matrix.png`

### 3) Run Webcam-Only Demo

```bash
python -m src.demo_webcam_only
```

**Controls**

* Press `q` to quit
* Overlay shows detected face bounding box + emotion label + confidence + valence/arousal

### 4) Run Full Voice Companion Session

```bash
python -m src.demo_voice_session
```

**Session Flow**

1. System greets you via TTS
2. Webcam captures emotion (short burst)
3. Microphone listens for speech
4. LLM generates supportive response (if configured)
5. TTS speaks the response
6. Repeat until exit

**Exit**

* Say: "goodbye", "quit", or "exit"
* Press `q` in webcam window
* Ctrl+C to force quit

**Session Output**

* Emotion log: `logs/sessions/session_<timestamp>_emotion.csv`
* Conversation log: `conv_logs/session_<timestamp>_conversation.csv`
* JSON report: `logs/reports/session_<timestamp>_report.json`

---

##  Architecture

### Emotion Detection Pipeline

1. **Face Detection:** MediaPipe (short-range model)
2. **Preprocessing:** crop → grayscale → resize 224×224 → 3-channel → ImageNet normalize
3. **Classification:** ResNet18 → softmax → emotion label
4. **Valence/Arousal:** mapping from emotion labels
5. **Smoothing:** majority voting over recent frames

### Conversation Flow

1. **Emotion Context:** current emotion state added to the system prompt
2. **Safety Check:** keyword detection for crisis situations
3. **LLM Call:** OpenAI API with conversation history (optional)
4. **Response:** warm, supportive 2–4 sentence reply
5. **Logging:** stored with timestamps + emotion state

### Safety Guardrails

The system monitors for keywords related to:

* self-harm / suicide
* harming others
* feeling unsafe

When detected, it:

* responds with empathy
* provides crisis resources (e.g., U.S. 988 Lifeline)
* encourages reaching out to trusted people or professionals
* never provides harmful instructions

---

## 🔧 Customization

### Switching to AffectNet (or Other Datasets)

1. Organize your dataset similarly:

   ```text
   data/affectnet/
     train/<class>/*.jpg
     test/<class>/*.jpg
   ```
2. Update label list and mapping in `src/config.py` (e.g., `FER_LABELS`, `EMOTION_VA_MAP`)
3. Train with the new dataset:

   ```bash
   python -m src.train_fer --data-root data/affectnet --epochs 30
   ```

### Adjusting LLM Behavior

Edit `src/conversation_manager.py`:

* `SYSTEM_PROMPT`: companion personality + guidelines
* `SAFETY_KEYWORDS`: crisis keyword list
* `model`: choose OpenAI model
* `temperature`: creativity (0.0–1.0)

### Emotion Smoothing

Edit `src/emotion_engine.py`:

* `smoothing_window`: number of frames used for majority vote (default: 5)

---

##  Troubleshooting

### Webcam Not Opening

* check if another app is using the webcam
* try changing camera index in OpenCV (`0` → `1` → `2`)

### Microphone Issues

* test microphone configuration
* tune thresholds inside `src/voice_io.py` if background noise is high

### TTS Not Working

* macOS: typically works out of the box
* Linux: install `espeak` (`sudo apt-get install espeak`)
* Windows: typically works out of the box

### Low Emotion Recognition Accuracy

* improve lighting and keep face frontal
* train longer (30+ epochs)
* consider data augmentation or fine-tuning
* try a higher-quality dataset like AffectNet

### LLM Not Responding

* ensure `OPENAI_API_KEY` is set
* confirm network access + rate limits

---

##  Dataset Details

### FER-2013 (Folder-Based)

* **Classes:** 7 emotions (angry, disgust, fear, happy, neutral, sad, surprise)
* **Splits:** train + test folders
* **Input:** resized to 224×224
* **Challenge:** varied lighting/pose + noisy labels

### Alternative: AffectNet

* larger dataset and typically better real-world performance
* can be used if organized into the same folder layout

---

##  Performance Expectations

### Training Time (rough)

* **CPU:** slow (can be hours per epoch depending on machine)
* **GPU (CUDA):** typically much faster
* **Apple Silicon (MPS):** often in between

### Accuracy

* **FER-2013:** 60–70% is common; strong models can do ~70%+
* **Real-time FPS:** depends on device and camera resolution

---

##  Citation

If you use FER-2013, please cite:

```bibtex
@inproceedings{goodfellow2013challenges,
  title={Challenges in representation learning: A report on three machine learning contests},
  author={Goodfellow, Ian J and Erhan, Dumitru and Carrier, Pierre Luc and Courville, Aaron and Mirza, Mehdi and Hamner, Ben and Cukierski, Will and Tang, Yichuan and Thaler, David and Lee, Dong-Hyun and others},
  booktitle={International conference on neural information processing},
  pages={117--124},
  year={2013},
  organization={Springer}
}
```

---

##  License

Educational and research use. NeuroBot is **not** a substitute for professional mental health care.

---

##  Acknowledgments

* FER-2013 dataset (Kaggle distribution)
* MediaPipe for face detection
* PyTorch + torchvision for modeling
* OpenAI (optional) for LLM capability
* SpeechRecognition + pyttsx3 for voice I/O
