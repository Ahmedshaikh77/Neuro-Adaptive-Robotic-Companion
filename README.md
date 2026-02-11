# Emotion-Aware Companion System

A comprehensive mental health support system that combines real-time emotion detection with voice-based conversation to provide supportive companionship between therapy sessions.

## Features

- **Emotion Classification**: ResNet18-based emotion classifier trained on FER-2013
- **Real-time Detection**: MediaPipe face detection + emotion recognition from webcam
- **Voice Interface**: Speech-to-text (STT) and text-to-speech (TTS) for natural conversation
- **LLM Integration**: Supportive, empathetic responses with safety guardrails
- **Session Logging**: Comprehensive emotion and conversation tracking with JSON reports
- **Safety First**: Built-in crisis detection and appropriate resource recommendations

## Safety & Ethics

This system is designed as a **supportive companion** and:
- Does NOT claim to be a therapist
- Does NOT provide medical diagnoses or clinical advice
- DOES include safety protocols for crisis situations
- DOES encourage seeking professional help when needed
- DOES maintain logs for potential review with healthcare providers

## Project Structure

```
final_project/
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── artifacts/                # Model checkpoints and confusion matrices
├── logs/
│   ├── sessions/            # Emotion CSV logs
│   └── reports/             # JSON session reports
├── conv_logs/               # Conversation CSV logs
└── src/
    ├── __init__.py
    ├── config.py            # Configuration and paths
    ├── fer_dataset.py       # FER-2013 dataset loader
    ├── train_fer.py         # Training script
    ├── eval_fer.py          # Evaluation script
    ├── emotion_engine.py    # Real-time emotion detection
    ├── voice_io.py          # STT and TTS
    ├── conversation_manager.py  # LLM conversation management
    ├── session_logger.py    # Logging and reporting
    ├── demo_webcam_only.py  # Webcam-only demo
    └── demo_voice_session.py  # Full voice session demo
```

## Quick Start

### 1. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note on PyAudio**: If you encounter issues installing PyAudio:
- **macOS**: `brew install portaudio` then `pip install pyaudio`
- **Ubuntu/Debian**: `sudo apt-get install portaudio19-dev` then `pip install pyaudio`
- **Windows**: Download pre-built wheel from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)

### 3. Prepare FER Dataset

The system expects a folder-based dataset structure:

```
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

You can download the FER-2013 dataset from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and organize it into this structure, or use any similar emotion dataset with the same folder layout.

### 4. Set Up OpenAI API Key (Optional)

For LLM-powered conversations, set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

If no API key is provided, the system will use fallback responses.

## Usage

### 1. Train the Emotion Classifier

Train ResNet18 on FER dataset (adjust epochs as needed):

```bash
# Quick test (1 epoch - for testing only)
python -m src.train_fer --data-root data/archive --epochs 1 --batch-size 64

# Full training (30 epochs - recommended)
python -m src.train_fer --data-root data/archive --epochs 30 --batch-size 64 --lr 0.001
```

**Training Output**:
- Model saved to: `artifacts/best_fer_resnet.pt`
- Best validation accuracy displayed at completion

**Expected Performance**:
- Val accuracy: ~60-70% after 30 epochs (FER-2013 is challenging)

### 2. Evaluate the Model

Evaluate on test set and generate metrics:

```bash
python -m src.eval_fer --data-root data/archive --checkpoint artifacts/best_fer_resnet.pt
```

**Evaluation Output**:
- Test accuracy
- Per-class precision, recall, F1-score
- Confusion matrix saved as:
  - `artifacts/confusion_matrix.npy` (NumPy array)
  - `artifacts/confusion_matrix.png` (visualization)

### 3. Run Webcam-Only Demo

Test real-time emotion detection:

```bash
python -m src.demo_webcam_only
```

**Controls**:
- Press `q` to quit
- Green bounding box shows detected face
- Emotion label, confidence, valence, and arousal displayed

### 4. Run Full Voice Companion Session

Experience the complete system:

```bash
python -m src.demo_voice_session
```

**Session Flow**:
1. System greets you via TTS
2. Webcam captures your emotion (3-second burst)
3. Microphone listens for your speech
4. LLM generates supportive response
5. TTS speaks the response
6. Repeat until you say "goodbye" or "quit"

**Exit**:
- Say "goodbye", "quit", or "exit"
- Press `q` in webcam window
- Ctrl+C to force quit

**Session Output**:
- Emotion log: `logs/sessions/session_<timestamp>_emotion.csv`
- Conversation log: `conv_logs/session_<timestamp>_conversation.csv`
- JSON report: `logs/reports/session_<timestamp>_report.json`

## Architecture

### Emotion Detection Pipeline

1. **Face Detection**: MediaPipe face detection (short-range model)
2. **Preprocessing**: Crop face → grayscale → resize to 224x224 → 3-channel → ImageNet normalize
3. **Classification**: ResNet18 → softmax → emotion label
4. **Valence/Arousal**: Hard-coded mapping from emotion labels
5. **Smoothing**: Majority voting over last 5 frames

### Conversation Flow

1. **Emotion Context**: Current emotion state added to LLM system prompt
2. **Safety Check**: Keyword detection for crisis situations
3. **LLM Call**: OpenAI API with conversation history
4. **Response**: Warm, supportive 2-4 sentence reply
5. **Logging**: Turn logged with timestamp, texts, and emotion state

### Safety Guardrails

The system monitors for keywords related to:
- Self-harm
- Suicide
- Harming others
- Feeling unsafe

When detected, it:
- Responds with high empathy
- Provides crisis resources (988 Lifeline, emergency services)
- Encourages reaching out to trusted people
- Never provides harmful instructions

## Customization

### Switching to AffectNet

To use AffectNet or other emotion datasets:

1. Organize your dataset in the same folder structure:
   ```
   data/affectnet/
     train/
       angry/*.jpg
       ...
     test/
       angry/*.jpg
       ...
   ```
2. Update `FER_LABELS` in [config.py](src/config.py) if label set differs
3. Update `EMOTION_VA_MAP` for your dataset's emotions
4. Train with new dataset:
   ```bash
   python -m src.train_fer --data-root data/affectnet --epochs 30
   ```

### Adjusting LLM Behavior

Edit [conversation_manager.py](src/conversation_manager.py):
- `SYSTEM_PROMPT`: Modify companion personality and guidelines
- `SAFETY_KEYWORDS`: Add/remove crisis detection keywords
- `model`: Change OpenAI model (e.g., "gpt-3.5-turbo", "gpt-4")
- `temperature`: Adjust response creativity (0.0-1.0)

### Emotion Smoothing

Edit [emotion_engine.py](src/emotion_engine.py):
- `smoothing_window`: Change number of frames for smoothing (default: 5)

## Troubleshooting

### Webcam Not Opening
- Check if another app is using the webcam
- Try changing camera index in `cv2.VideoCapture(0)` to `1` or `2`

### Microphone Issues
- Run `python -m speech_recognition` to test microphone
- Adjust `energy_threshold` in [voice_io.py](src/voice_io.py) if background noise is high

### TTS Not Working
- macOS: Should work out of the box
- Linux: Install `espeak`: `sudo apt-get install espeak`
- Windows: Should work out of the box

### Low Emotion Recognition Accuracy
- Ensure good lighting for webcam
- Face should be frontal and clearly visible
- Train longer (30+ epochs)
- Consider data augmentation or fine-tuning

### LLM Not Responding
- Check `OPENAI_API_KEY` is set correctly
- Verify API key has credits
- Check internet connection
- Review API rate limits

## Dataset Details

### FER Dataset (Folder Structure)
- **Format**: Images organized in folders by emotion class
- **Classes**: 7 emotions (angry, disgust, fear, happy, neutral, sad, surprise)
- **Splits**:
  - Train: Used for training and validation (15% held out for val)
  - Test: Separate test set for evaluation
- **Input**: RGB images, resized to 224×224
- **Challenge**: Varied quality, lighting, and pose variations

### Alternative: AffectNet
- **Size**: 400,000+ images (larger, higher quality)
- **Classes**: 8 emotions + valence/arousal annotations
- **Advantage**: Better real-world performance
- **Usage**: Organize in same folder structure and use `--data-root data/affectnet`

## Performance Expectations

### Training Time
- **CPU**: ~2-3 hours per epoch
- **GPU (CUDA)**: ~10-15 minutes per epoch
- **MPS (Apple Silicon)**: ~15-20 minutes per epoch

### Accuracy
- **FER-2013**: 60-70% test accuracy (state-of-the-art: ~73%)
- **Real-time FPS**: 15-30 FPS (depends on device)

## Citation

If you use FER-2013, please cite:

```
@inproceedings{goodfellow2013challenges,
  title={Challenges in representation learning: A report on three machine learning contests},
  author={Goodfellow, Ian J and Erhan, Dumitru and Carrier, Pierre Luc and Courville, Aaron and Mirza, Mehdi and Hamner, Ben and Cukierski, Will and Tang, Yichuan and Thaler, David and Lee, Dong-Hyun and others},
  booktitle={International conference on neural information processing},
  pages={117--124},
  year={2013},
  organization={Springer}
}
```

## License

This project is for educational and research purposes. The companion system is not a substitute for professional mental health care.

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review logs in `logs/` and `conv_logs/`
3. Open an issue with error messages and system details

## Acknowledgments

- FER-2013 dataset from Kaggle
- MediaPipe for face detection
- PyTorch and torchvision for deep learning
- OpenAI for LLM capabilities
- SpeechRecognition and pyttsx3 for voice I/O
