"""
Real-time emotion detection engine using MediaPipe and ResNet18.
"""

from dataclasses import dataclass
from collections import deque
from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import mediapipe as mp
from PIL import Image

from src.config import FER_LABELS, EMOTION_VA_MAP, get_device, BEST_MODEL_PATH
from src.train_fer import create_model


@dataclass
class EmotionState:
    """
    Represents the current emotional state.
    """
    label_idx: int
    label_name: str
    valence: float
    arousal: float
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)


class MediaPipeFaceDetector:
    """
    Face detection using MediaPipe.
    """

    def __init__(self, min_detection_confidence: float = 0.5):
        """
        Initialize MediaPipe face detection.

        Args:
            min_detection_confidence: Minimum confidence threshold
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=0,  # 0 for short-range (< 2 meters)
        )

    def detect(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face in frame.

        Args:
            frame_bgr: BGR image (H, W, 3)

        Returns:
            Bounding box (x, y, w, h) or None if no face detected
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.detector.process(frame_rgb)

        if results.detections:
            # Get the first detection
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box

            # Convert normalized coordinates to pixel coordinates
            h, w = frame_bgr.shape[:2]
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            box_w = int(bbox.width * w)
            box_h = int(bbox.height * h)

            return (x, y, box_w, box_h)

        return None

    def close(self):
        """Release resources."""
        self.detector.close()


class EmotionEngine:
    """
    Real-time emotion detection engine.
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
        smoothing_window: int = 5,
    ):
        """
        Initialize emotion detection engine.

        Args:
            model_path: Path to model checkpoint
            device: Torch device (auto-detected if None)
            smoothing_window: Number of recent frames to smooth over
        """
        self.device = device if device else get_device()
        self.smoothing_window = smoothing_window

        # Load model
        print(f"Loading emotion model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = create_model(num_classes=len(FER_LABELS), pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Face detector
        self.face_detector = MediaPipeFaceDetector()

        # Transform for preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # Smoothing queue
        self.recent_states = deque(maxlen=smoothing_window)

        print(f"Emotion engine initialized on device: {self.device}")

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[EmotionState]:
        """
        Process a single frame and return smoothed emotion state.

        Args:
            frame_bgr: BGR image (H, W, 3)

        Returns:
            EmotionState or None if no face detected
        """
        # Detect face
        bbox = self.face_detector.detect(frame_bgr)
        if bbox is None:
            return None

        # Crop face region
        x, y, w, h = bbox
        # Add some padding
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame_bgr.shape[1], x + w + padding)
        y2 = min(frame_bgr.shape[0], y + h + padding)

        face_crop = frame_bgr[y1:y2, x1:x2]

        if face_crop.size == 0:
            return None

        # Convert to grayscale and preprocess
        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        face_pil = Image.fromarray(face_gray, mode="L")
        face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)

        # Predict emotion
        with torch.no_grad():
            outputs = self.model(face_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)

        label_idx = predicted.item()
        label_name = FER_LABELS[label_idx]
        confidence_val = confidence.item()

        # Get valence and arousal
        va = EMOTION_VA_MAP[label_name]
        valence = va["valence"]
        arousal = va["arousal"]

        # Create current state
        current_state = EmotionState(
            label_idx=label_idx,
            label_name=label_name,
            valence=valence,
            arousal=arousal,
            confidence=confidence_val,
            bbox=bbox,
        )

        # Add to smoothing queue
        self.recent_states.append(current_state)

        # Return smoothed state
        return self._get_smoothed_state()

    def _get_smoothed_state(self) -> EmotionState:
        """
        Compute smoothed emotion state from recent frames.

        Returns:
            Smoothed EmotionState
        """
        if not self.recent_states:
            return None

        # Use majority voting for label
        labels = [s.label_idx for s in self.recent_states]
        label_idx = max(set(labels), key=labels.count)
        label_name = FER_LABELS[label_idx]

        # Average valence, arousal, confidence
        valence = np.mean([s.valence for s in self.recent_states])
        arousal = np.mean([s.arousal for s in self.recent_states])
        confidence = np.mean([s.confidence for s in self.recent_states])

        # Use most recent bbox
        bbox = self.recent_states[-1].bbox

        return EmotionState(
            label_idx=label_idx,
            label_name=label_name,
            valence=float(valence),
            arousal=float(arousal),
            confidence=float(confidence),
            bbox=bbox,
        )

    def reset_smoothing(self):
        """Clear smoothing queue."""
        self.recent_states.clear()

    def close(self):
        """Release resources."""
        self.face_detector.close()


def main():
    """
    Demo: Run webcam emotion detection.
    """
    import sys

    # Check if model exists
    if not BEST_MODEL_PATH.exists():
        print(f"Error: Model not found at {BEST_MODEL_PATH}")
        print("Please train the model first using train_fer.py")
        sys.exit(1)

    # Initialize engine
    engine = EmotionEngine(str(BEST_MODEL_PATH))

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        sys.exit(1)

    print("Starting webcam emotion detection...")
    print("Press 'q' to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            emotion_state = engine.process_frame(frame)

            # Draw results
            if emotion_state:
                # Draw bounding box
                x, y, w, h = emotion_state.bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Draw emotion label
                label_text = f"{emotion_state.label_name} ({emotion_state.confidence:.2f})"
                cv2.putText(
                    frame,
                    label_text,
                    (x, y - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                # Draw valence/arousal
                va_text = f"V:{emotion_state.valence:.2f} A:{emotion_state.arousal:.2f}"
                cv2.putText(
                    frame,
                    va_text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            # Show frame
            cv2.imshow("Emotion Detection", frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        engine.close()
        print("Webcam closed")


if __name__ == "__main__":
    main()
