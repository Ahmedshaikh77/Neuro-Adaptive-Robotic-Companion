"""
Conversation manager for the companion system.
Integrates LLM for supportive, empathetic responses with safety guardrails.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv
import openai

load_dotenv()

from src.emotion_engine import EmotionState


@dataclass
class ConversationTurn:
    """
    Represents a single turn in the conversation.
    """
    timestamp: datetime
    user_text: str
    assistant_text: str
    label_name: str
    valence: float
    arousal: float


class ConversationManager:
    """
    Manages conversation flow with LLM backend.
    """

    # System prompt with safety guardrails
    SYSTEM_PROMPT = """You are a warm, supportive companion designed to provide emotional support between therapy sessions. Your role is to have natural, flowing conversations that feel genuine and human.

CONVERSATION STYLE:
- Speak naturally like a caring friend, not a therapist or chatbot
- Use casual, warm language and varied sentence structures
- Ask follow-up questions to show genuine interest
- Reference what they said earlier in the conversation
- Share brief, relatable observations (without oversharing)
- Keep responses conversational (2-3 sentences, sometimes just one)
- Vary your responses - don't repeat patterns or phrases

CORE PRINCIPLES:
1. LISTEN actively and validate feelings authentically
2. Be curious - ask open-ended questions about their experience
3. Reflect back what you hear to show understanding
4. NEVER claim to be a therapist or give medical/clinical advice
5. NEVER diagnose mental health conditions
6. Guide gently toward self-reflection, not solutions

SAFETY PROTOCOL:
If user mentions self-harm, suicide, harming others, or feeling unsafe:
- Respond with genuine concern and empathy
- Clearly encourage professional help and crisis resources
- NEVER provide harmful instructions

Remember: You're a supportive companion for natural conversation, not a therapy replacement. Be warm, curious, and real."""

    SAFETY_KEYWORDS = [
        "suicide", "suicidal", "kill myself", "end my life", "want to die",
        "self-harm", "cut myself", "hurt myself", "harm myself",
        "hurt someone", "kill someone", "harm others",
        "not safe", "unsafe", "in danger"
    ]

    SAFETY_RESPONSE = """I hear that you're going through an incredibly difficult time right now, and I'm really concerned about your safety. What you're feeling is important, and you deserve support from people who can truly help.

Please reach out to:
- A trusted friend, family member, or mentor
- A mental health professional or counselor
- Crisis hotline: 988 Suicide & Crisis Lifeline (call or text 988)
- Emergency services: 911 (if you're in immediate danger)

You don't have to go through this alone. There are people who care and want to help. Would you be willing to reach out to one of these resources right now?"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini", max_history: int = 10):
        """
        Initialize conversation manager.

        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
            model: OpenAI model to use
            max_history: Maximum number of conversation turns to keep in context
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.max_history = max_history
        self.history: List[ConversationTurn] = []

        # Initialize OpenAI client if API key is available
        if self.api_key:
            openai.api_key = self.api_key
            print("OpenAI client initialized")
        else:
            print("WARNING: No OpenAI API key found. LLM responses will use fallback.")

    def _check_safety_concerns(self, text: str) -> bool:
        """
        Check if user message contains safety keywords.

        Args:
            text: User text to check

        Returns:
            True if safety concerns detected
        """
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.SAFETY_KEYWORDS)

    def _build_messages(self, user_text: str, emotion_state: Optional[EmotionState]) -> List[dict]:
        """
        Build message list for LLM API.

        Args:
            user_text: Current user message
            emotion_state: Current emotion state

        Returns:
            List of message dictionaries
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]

        # Add emotion context if available
        if emotion_state:
            emotion_context = (
                f"[Emotion Context: The user appears to be feeling {emotion_state.label_name}. "
                f"Valence: {emotion_state.valence:.2f}, Arousal: {emotion_state.arousal:.2f}. "
                f"Use this to inform your tone, but do not explicitly mention these metrics.]"
            )
            messages.append({"role": "system", "content": emotion_context})

        # Add recent conversation history
        recent_history = self.history[-self.max_history:]
        for turn in recent_history:
            messages.append({"role": "user", "content": turn.user_text})
            messages.append({"role": "assistant", "content": turn.assistant_text})

        # Add current user message
        messages.append({"role": "user", "content": user_text})

        return messages

    def _call_llm(self, messages: List[dict]) -> str:
        """
        Call LLM API to generate response.

        Args:
            messages: List of message dictionaries

        Returns:
            Assistant response text
        """
        if not self.api_key:
            # Fallback response if no API key
            return "I'm here to listen. Can you tell me more about how you're feeling?"

        try:
            # Use new OpenAI API (v1.0+)
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return "I'm having trouble responding right now, but I'm here to listen. Can you tell me more?"

    def process_turn(
        self,
        user_text: str,
        emotion_state: Optional[EmotionState] = None
    ) -> str:
        """
        Process a conversation turn.

        Args:
            user_text: User's message
            emotion_state: Current emotional state

        Returns:
            Assistant's response
        """
        # Check for safety concerns
        if self._check_safety_concerns(user_text):
            assistant_reply = self.SAFETY_RESPONSE
        else:
            # Build messages and call LLM
            messages = self._build_messages(user_text, emotion_state)
            assistant_reply = self._call_llm(messages)

        # Create turn record
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_text=user_text,
            assistant_text=assistant_reply,
            label_name=emotion_state.label_name if emotion_state else "unknown",
            valence=emotion_state.valence if emotion_state else 0.0,
            arousal=emotion_state.arousal if emotion_state else 0.0,
        )

        # Add to history
        self.history.append(turn)

        return assistant_reply

    def get_history(self) -> List[ConversationTurn]:
        """Get conversation history."""
        return self.history

    def clear_history(self):
        """Clear conversation history."""
        self.history.clear()


def demo():
    """
    Demo conversation manager (without emotion state).
    """
    print("Conversation Manager Demo")
    print("=" * 50)
    print("Type 'quit' to exit\n")

    manager = ConversationManager()

    # Simulate conversation
    test_inputs = [
        "I've been feeling really stressed lately",
        "I'm worried about my exams",
        "quit"
    ]

    for user_input in test_inputs:
        if user_input == "quit":
            break

        print(f"\nUser: {user_input}")
        response = manager.process_turn(user_input)
        print(f"Assistant: {response}")

    print("\n" + "=" * 50)
    print("Demo complete")


if __name__ == "__main__":
    demo()
