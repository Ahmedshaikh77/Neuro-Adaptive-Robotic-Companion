"""
Voice I/O: Speech-to-Text (STT) and Text-to-Speech (TTS).
"""

from typing import Optional
import subprocess
import platform
import speech_recognition as sr


class MicListener:
    """
    Microphone listener using Google Speech Recognition.
    """

    def __init__(self, energy_threshold: int = 300, pause_threshold: float = 0.8):
        """
        Initialize microphone listener.

        Args:
            energy_threshold: Minimum audio energy to consider for recording
            pause_threshold: Seconds of silence to mark end of phrase
        """
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True  # Auto-adjust to ambient noise
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.pause_threshold = pause_threshold
        self.microphone = sr.Microphone()

        # Adjust for ambient noise
        print("Adjusting for ambient noise... Please wait.")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            print(f"Energy threshold set to: {self.recognizer.energy_threshold}")
        print("Microphone ready!")

    def listen_once(self, prompt: Optional[str] = None, timeout: int = 15, phrase_time_limit: int = 20) -> str:
        """
        Listen for one utterance and convert to text.

        Args:
            prompt: Optional prompt to display before listening
            timeout: Maximum seconds to wait for speech to start
            phrase_time_limit: Maximum seconds for the phrase

        Returns:
            Transcribed text (empty string if failed)
        """
        if prompt:
            print(prompt)

        max_retries = 2
        for attempt in range(max_retries):
            with self.microphone as source:
                if attempt > 0:
                    print(f"Retry {attempt}/{max_retries - 1}...")
                print("🎤 Listening... (speak now)")
                try:
                    # Adjust for ambient noise briefly before each listen
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                except sr.WaitTimeoutError:
                    if attempt < max_retries - 1:
                        print("No speech detected. Please try speaking louder...")
                        continue
                    else:
                        print("⏱️  Listening timed out.")
                        return ""

            print("🔄 Processing speech...")
            try:
                text = self.recognizer.recognize_google(audio)
                print(f"✅ You said: '{text}'")
                return text
            except sr.UnknownValueError:
                if attempt < max_retries - 1:
                    print("❌ Could not understand audio. Please speak more clearly...")
                    continue
                else:
                    print("❌ Could not understand audio after retries")
                    return ""
            except sr.RequestError as e:
                print(f"⚠️  Speech recognition service error: {e}")
                return ""

        return ""


class VoiceAgent:
    """
    Text-to-Speech agent.
    Uses macOS native 'say' command (reliable) with pyttsx3 fallback for other platforms.
    """

    def __init__(self, rate: int = 150, voice: str = "Samantha"):
        """
        Initialize TTS engine.

        Args:
            rate: Speech rate (words per minute)
            voice: Voice name for macOS 'say' command
        """
        self.rate = rate
        self.voice = voice
        self._use_say = platform.system() == "Darwin"

        if self._use_say:
            print(f"Using macOS 'say' command with voice: {self.voice}")
        else:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', 0.9)
            voices = self.engine.getProperty('voices')
            for v in voices:
                if any(kw in v.name.lower() for kw in ['female', 'samantha', 'victoria', 'zira', 'fiona']):
                    self.engine.setProperty('voice', v.id)
                    print(f"Using voice: {v.name}")
                    break

    def speak(self, text: str):
        """
        Speak text (blocking).

        Args:
            text: Text to speak
        """
        try:
            print(f"🔊 Speaking: {text[:50]}{'...' if len(text) > 50 else ''}")
            if self._use_say:
                subprocess.run(
                    ["say", "-v", self.voice, "-r", str(self.rate), text],
                    check=True,
                )
            else:
                self.engine.say(text)
                self.engine.runAndWait()
        except Exception as e:
            print(f"⚠️  TTS Error: {e}")

    def wait_until_done(self):
        """
        Wait until all queued speech is finished.
        (No-op — speak() is already blocking)
        """
        pass

    def close(self):
        """
        Shutdown TTS engine.
        """
        if not self._use_say:
            try:
                self.engine.stop()
            except:
                pass


def demo():
    """
    Demo: Listen and repeat.
    """
    print("Voice I/O Demo")
    print("=" * 50)

    listener = MicListener()
    agent = VoiceAgent()

    agent.speak("Hi! I'm your voice assistant. Say something and I'll repeat it back to you.")
    agent.wait_until_done()

    for i in range(3):
        text = listener.listen_once(f"\n[Turn {i+1}/3] Speak now:")
        if text:
            agent.speak(f"You said: {text}")
            agent.wait_until_done()
        else:
            agent.speak("I didn't catch that. Let's try again.")
            agent.wait_until_done()

    agent.speak("Demo complete. Goodbye!")
    agent.wait_until_done()
    agent.close()


if __name__ == "__main__":
    demo()
