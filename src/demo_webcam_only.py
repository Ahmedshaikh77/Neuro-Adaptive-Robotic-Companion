"""
Demo: Webcam-only emotion detection.
Displays live emotion predictions with valence/arousal.
"""

import sys
from src.emotion_engine import EmotionEngine
from src.config import BEST_MODEL_PATH


def main():
    """
    Run webcam emotion detection demo.
    """
    print("=" * 60)
    print("WEBCAM EMOTION DETECTION DEMO")
    print("=" * 60)

    # Check if model exists
    if not BEST_MODEL_PATH.exists():
        print(f"\nError: Model not found at {BEST_MODEL_PATH}")
        print("Please train the model first using:")
        print("  python -m src.train_fer --csv-path <path-to-fer2013.csv> --epochs 30")
        sys.exit(1)

    # Run emotion engine main (which has webcam demo built-in)
    print("\nStarting webcam...")
    print("Controls:")
    print("  - Press 'q' to quit")
    print("=" * 60 + "\n")

    # Import and run the emotion engine demo
    from src.emotion_engine import main as run_engine
    run_engine()


if __name__ == "__main__":
    main()
