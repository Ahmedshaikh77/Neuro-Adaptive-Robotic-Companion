"""
Configuration file for paths and device management.
"""

import os
from pathlib import Path
import torch

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_ROOT = PROJECT_ROOT / "data" / "archive"  # Default to data/archive for folder-based dataset
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"
SESSIONS_DIR = LOGS_DIR / "sessions"
REPORTS_DIR = LOGS_DIR / "reports"
CONV_LOGS_DIR = PROJECT_ROOT / "conv_logs"

# Model checkpoint
BEST_MODEL_PATH = ARTIFACTS_DIR / "best_fer_resnet.pt"
CONFUSION_MATRIX_NPY = ARTIFACTS_DIR / "confusion_matrix.npy"
CONFUSION_MATRIX_PNG = ARTIFACTS_DIR / "confusion_matrix.png"

# FER emotion labels (from folder names)
FER_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Valence-Arousal mapping (hard-coded for FER labels)
# Valence: -1 (negative) to +1 (positive)
# Arousal: -1 (low energy) to +1 (high energy)
EMOTION_VA_MAP = {
    "angry": {"valence": -0.8, "arousal": 0.7},
    "disgust": {"valence": -0.7, "arousal": 0.3},
    "fear": {"valence": -0.9, "arousal": 0.8},
    "happy": {"valence": 0.9, "arousal": 0.6},
    "sad": {"valence": -0.7, "arousal": -0.5},
    "surprise": {"valence": 0.2, "arousal": 0.8},
    "neutral": {"valence": 0.0, "arousal": 0.0},
}


def get_device() -> torch.device:
    """
    Get the best available device (cuda > mps > cpu).

    Returns:
        torch.device: The device to use for computation
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def ensure_directories() -> None:
    """
    Create all necessary directories if they don't exist.
    """
    for directory in [DATA_ROOT, ARTIFACTS_DIR, LOGS_DIR, SESSIONS_DIR, REPORTS_DIR, CONV_LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
