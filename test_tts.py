"""
Standalone TTS test script to verify pyttsx3 works.
"""

import pyttsx3
import time

print("Initializing pyttsx3 engine...")
engine = pyttsx3.init()

# Get available voices
voices = engine.getProperty('voices')
print(f"\nAvailable voices ({len(voices)}):")
for i, voice in enumerate(voices):
    print(f"  {i}: {voice.name} ({voice.id})")

# Set properties
rate = engine.getProperty('rate')
volume = engine.getProperty('volume')
print(f"\nCurrent settings:")
print(f"  Rate: {rate}")
print(f"  Volume: {volume}")

engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

# Try to use Samantha voice
for voice in voices:
    if 'samantha' in voice.name.lower():
        engine.setProperty('voice', voice.id)
        print(f"\nUsing voice: {voice.name}")
        break

# Test 1: Simple blocking speech
print("\n" + "="*50)
print("Test 1: Simple blocking speech")
print("="*50)
print("Speaking: 'Hello, this is a test.'")
engine.say("Hello, this is a test.")
print("Calling runAndWait()...")
engine.runAndWait()
print("runAndWait() returned")

# Give a moment between tests
time.sleep(1)

# Test 2: Multiple utterances
print("\n" + "="*50)
print("Test 2: Multiple utterances")
print("="*50)
print("Speaking: 'First sentence.' and 'Second sentence.'")
engine.say("First sentence.")
engine.say("Second sentence.")
print("Calling runAndWait()...")
engine.runAndWait()
print("runAndWait() returned")

print("\n" + "="*50)
print("TTS test complete!")
print("If you heard audio, pyttsx3 is working correctly.")
print("If you didn't hear anything, there may be a system audio issue.")
print("="*50)
