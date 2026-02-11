"""
Session logging for emotions and conversations.
Generates comprehensive session reports in JSON format.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import pandas as pd

from src.config import SESSIONS_DIR, REPORTS_DIR, CONV_LOGS_DIR, ensure_directories
from src.emotion_engine import EmotionState
from src.conversation_manager import ConversationTurn


class EmotionLogger:
    """
    Logs emotion states to CSV.
    """

    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize emotion logger.

        Args:
            session_id: Unique session identifier (auto-generated if None)
        """
        ensure_directories()

        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.session_id = session_id
        self.log_path = SESSIONS_DIR / f"session_{session_id}_emotion.csv"

        # Create CSV file with headers
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'label_name', 'valence', 'arousal', 'confidence'])

        print(f"Emotion logger initialized: {self.log_path}")

    def log(self, emotion_state: EmotionState):
        """
        Log a single emotion state.

        Args:
            emotion_state: EmotionState to log
        """
        timestamp = datetime.now().isoformat()
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                emotion_state.label_name,
                f"{emotion_state.valence:.4f}",
                f"{emotion_state.arousal:.4f}",
                f"{emotion_state.confidence:.4f}",
            ])

    def get_log_path(self) -> Path:
        """Get path to emotion log CSV."""
        return self.log_path


class ConversationLogger:
    """
    Logs conversation turns to CSV.
    """

    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize conversation logger.

        Args:
            session_id: Unique session identifier (auto-generated if None)
        """
        ensure_directories()

        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.session_id = session_id
        self.log_path = CONV_LOGS_DIR / f"session_{session_id}_conversation.csv"

        # Create CSV file with headers
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'user_text',
                'assistant_text',
                'emotion_label',
                'valence',
                'arousal'
            ])

        print(f"Conversation logger initialized: {self.log_path}")

    def log(self, turn: ConversationTurn):
        """
        Log a conversation turn.

        Args:
            turn: ConversationTurn to log
        """
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                turn.timestamp.isoformat(),
                turn.user_text,
                turn.assistant_text,
                turn.label_name,
                f"{turn.valence:.4f}",
                f"{turn.arousal:.4f}",
            ])

    def get_log_path(self) -> Path:
        """Get path to conversation log CSV."""
        return self.log_path


def generate_session_report(
    emotion_csv: Path,
    convo_csv: Path,
    out_path: Optional[Path] = None,
) -> dict:
    """
    Generate comprehensive session report from logs.

    Args:
        emotion_csv: Path to emotion log CSV
        convo_csv: Path to conversation log CSV
        out_path: Path to save JSON report (auto-generated if None)

    Returns:
        Report dictionary
    """
    ensure_directories()

    # Load data
    emotion_df = pd.read_csv(emotion_csv)
    convo_df = pd.read_csv(convo_csv)

    # Parse timestamps
    emotion_df['timestamp'] = pd.to_datetime(emotion_df['timestamp'])
    convo_df['timestamp'] = pd.to_datetime(convo_df['timestamp'])

    # Session duration
    if len(emotion_df) > 0:
        session_start = emotion_df['timestamp'].min()
        session_end = emotion_df['timestamp'].max()
        duration_seconds = (session_end - session_start).total_seconds()
    else:
        session_start = datetime.now()
        session_end = datetime.now()
        duration_seconds = 0

    # Emotion statistics
    emotion_counts = emotion_df['label_name'].value_counts().to_dict()
    avg_valence = float(emotion_df['valence'].mean()) if len(emotion_df) > 0 else 0.0
    avg_arousal = float(emotion_df['arousal'].mean()) if len(emotion_df) > 0 else 0.0
    avg_confidence = float(emotion_df['confidence'].mean()) if len(emotion_df) > 0 else 0.0

    # Conversation statistics
    num_turns = len(convo_df)
    avg_user_words = convo_df['user_text'].str.split().str.len().mean() if num_turns > 0 else 0
    avg_assistant_words = convo_df['assistant_text'].str.split().str.len().mean() if num_turns > 0 else 0

    # Build report
    report = {
        "session_id": emotion_csv.stem.replace("session_", "").replace("_emotion", ""),
        "session_start": session_start.isoformat(),
        "session_end": session_end.isoformat(),
        "duration_seconds": duration_seconds,
        "duration_minutes": duration_seconds / 60,
        "emotion_statistics": {
            "total_frames": len(emotion_df),
            "emotion_distribution": emotion_counts,
            "avg_valence": avg_valence,
            "avg_arousal": avg_arousal,
            "avg_confidence": avg_confidence,
        },
        "conversation_statistics": {
            "num_turns": num_turns,
            "avg_user_words": float(avg_user_words) if not pd.isna(avg_user_words) else 0,
            "avg_assistant_words": float(avg_assistant_words) if not pd.isna(avg_assistant_words) else 0,
        },
        "files": {
            "emotion_log": str(emotion_csv),
            "conversation_log": str(convo_csv),
        }
    }

    # Save report
    if out_path is None:
        session_id = report["session_id"]
        out_path = REPORTS_DIR / f"session_{session_id}_report.json"

    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nSession report saved to: {out_path}")
    return report


def print_session_summary(report: dict):
    """
    Print a human-readable session summary.

    Args:
        report: Session report dictionary
    """
    print("\n" + "=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    print(f"Session ID: {report['session_id']}")
    print(f"Duration: {report['duration_minutes']:.1f} minutes")
    print(f"\nEmotion Statistics:")
    print(f"  Total frames: {report['emotion_statistics']['total_frames']}")
    print(f"  Avg Valence: {report['emotion_statistics']['avg_valence']:.2f}")
    print(f"  Avg Arousal: {report['emotion_statistics']['avg_arousal']:.2f}")
    print(f"  Emotion Distribution:")
    for emotion, count in report['emotion_statistics']['emotion_distribution'].items():
        percentage = 100 * count / report['emotion_statistics']['total_frames']
        print(f"    {emotion}: {count} ({percentage:.1f}%)")
    print(f"\nConversation Statistics:")
    print(f"  Turns: {report['conversation_statistics']['num_turns']}")
    print(f"  Avg User Words: {report['conversation_statistics']['avg_user_words']:.1f}")
    print(f"  Avg Assistant Words: {report['conversation_statistics']['avg_assistant_words']:.1f}")
    print("=" * 60 + "\n")


def demo():
    """
    Demo logger usage.
    """
    from src.emotion_engine import EmotionState
    from src.conversation_manager import ConversationTurn

    print("Session Logger Demo")
    print("=" * 50)

    # Create loggers with same session ID
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_demo")
    emotion_logger = EmotionLogger(session_id)
    convo_logger = ConversationLogger(session_id)

    # Log some dummy emotion states
    for i in range(5):
        state = EmotionState(
            label_idx=i % 7,
            label_name=["happy", "sad", "neutral", "angry", "surprise", "fear", "disgust"][i % 7],
            valence=0.5,
            arousal=0.3,
            confidence=0.8,
        )
        emotion_logger.log(state)

    # Log some dummy conversation turns
    for i in range(2):
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_text=f"User message {i+1}",
            assistant_text=f"Assistant response {i+1}",
            label_name="happy",
            valence=0.5,
            arousal=0.3,
        )
        convo_logger.log(turn)

    # Generate report
    report = generate_session_report(
        emotion_logger.get_log_path(),
        convo_logger.get_log_path(),
    )

    print_session_summary(report)
    print("Demo complete!")


if __name__ == "__main__":
    demo()
