"""
Demo: Full voice companion session.
Integrates webcam emotion detection with voice conversation.
"""

import sys
import cv2
import time
from datetime import datetime
from collections import deque

from src.emotion_engine import EmotionEngine, EmotionState
from src.voice_io import MicListener, VoiceAgent
from src.conversation_manager import ConversationManager
from src.session_logger import EmotionLogger, ConversationLogger, generate_session_report, print_session_summary
from src.config import BEST_MODEL_PATH


def aggregate_emotion_states(states: list[EmotionState]) -> EmotionState:
    """
    Aggregate multiple emotion states into a single representative state.

    Args:
        states: List of EmotionState objects

    Returns:
        Aggregated EmotionState
    """
    if not states:
        return None

    # Use majority voting for label
    labels = [s.label_idx for s in states]
    label_idx = max(set(labels), key=labels.count)

    # Get the corresponding state to extract label_name
    representative_state = next(s for s in states if s.label_idx == label_idx)

    # Average valence, arousal, confidence
    import numpy as np
    valence = float(np.mean([s.valence for s in states]))
    arousal = float(np.mean([s.arousal for s in states]))
    confidence = float(np.mean([s.confidence for s in states]))

    return EmotionState(
        label_idx=label_idx,
        label_name=representative_state.label_name,
        valence=valence,
        arousal=arousal,
        confidence=confidence,
        bbox=states[-1].bbox if states else None,
    )


def main():
    """
    Run full voice companion session.
    """
    print("=" * 60)
    print("EMOTION-AWARE COMPANION SESSION")
    print("=" * 60)

    # Check if model exists
    if not BEST_MODEL_PATH.exists():
        print(f"\nError: Model not found at {BEST_MODEL_PATH}")
        print("Please train the model first using:")
        print("  python -m src.train_fer --csv-path <path-to-fer2013.csv> --epochs 30")
        sys.exit(1)

    # Initialize session ID
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nSession ID: {session_id}")

    # Initialize components
    print("\nInitializing components...")
    try:
        engine = EmotionEngine(str(BEST_MODEL_PATH))
        mic_listener = MicListener()
        voice_agent = VoiceAgent()
        conversation_manager = ConversationManager()
        emotion_logger = EmotionLogger(session_id)
        convo_logger = ConversationLogger(session_id)
    except Exception as e:
        print(f"Error initializing components: {e}")
        sys.exit(1)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Session started!")
    print("Controls:")
    print("  - Speak naturally when prompted")
    print("  - Say 'goodbye' or 'quit' to end session")
    print("  - Close the webcam window to force quit")
    print("=" * 60 + "\n")

    # Greeting
    greeting = "Hi, I'm your companion between sessions. How are you feeling today?"
    print(f"Companion: {greeting}")
    voice_agent.speak(greeting)
    voice_agent.wait_until_done()

    # Main conversation loop
    turn_count = 0
    max_turns = 20  # Safety limit

    try:
        while turn_count < max_turns:
            print(f"\n{'─' * 60}")
            print(f"Turn {turn_count + 1}")
            print(f"{'─' * 60}")

            # Capture emotion over a short burst (3 seconds)
            print("Capturing emotion from webcam...")
            frame_states = []
            start_time = time.time()
            capture_duration = 3.0  # seconds

            while time.time() - start_time < capture_duration:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                emotion_state = engine.process_frame(frame)

                # Draw results on frame
                if emotion_state:
                    frame_states.append(emotion_state)
                    emotion_logger.log(emotion_state)

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
                        0.7,
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
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                # Show frame
                cv2.imshow("Companion Session", frame)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nSession ended by user")
                    raise KeyboardInterrupt

            # Aggregate emotion for this turn
            if frame_states:
                turn_emotion = aggregate_emotion_states(frame_states)
                print(f"Detected emotion: {turn_emotion.label_name} "
                      f"(V: {turn_emotion.valence:.2f}, A: {turn_emotion.arousal:.2f})")
            else:
                turn_emotion = None
                print("No emotion detected this turn")

            # Listen for user input
            user_text = mic_listener.listen_once("\nYour turn to speak:")

            if not user_text:
                voice_agent.speak("I didn't catch that. Could you please repeat?")
                voice_agent.wait_until_done()
                continue

            # Check for exit keywords
            if any(keyword in user_text.lower() for keyword in ['goodbye', 'bye', 'quit', 'exit', 'end session']):
                farewell = "Take care of yourself. Remember, you can always reach out to someone you trust if you need support. Goodbye!"
                print(f"\nCompanion: {farewell}")
                voice_agent.speak(farewell)
                voice_agent.wait_until_done()
                break

            # Generate response
            print("Companion is thinking...")
            assistant_reply = conversation_manager.process_turn(user_text, turn_emotion)
            print(f"Companion: {assistant_reply}")

            # Log conversation turn
            turn = conversation_manager.get_history()[-1]
            convo_logger.log(turn)

            # Speak response
            voice_agent.speak(assistant_reply)
            voice_agent.wait_until_done()

            turn_count += 1

    except KeyboardInterrupt:
        print("\n\nSession interrupted by user")

    finally:
        # Cleanup
        print("\nCleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        engine.close()
        voice_agent.close()

        # Generate session report
        print("\nGenerating session report...")
        try:
            report = generate_session_report(
                emotion_logger.get_log_path(),
                convo_logger.get_log_path(),
            )
            print_session_summary(report)
        except Exception as e:
            print(f"Error generating report: {e}")

        print("\n" + "=" * 60)
        print("Session complete. Thank you!")
        print("=" * 60)


if __name__ == "__main__":
    main()
