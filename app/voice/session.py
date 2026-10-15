"""Voice session - orchestrates listener, agent, TTS, and voice confirmations."""
from __future__ import annotations

import logging
import re
import threading
import time

from app.assistant.agent import JarvusAgent
from app.voice.listener import VoiceListener
from app.voice.tts import TextToSpeech

log = logging.getLogger("jarvus.voice.session")


def clean_for_speech(text: str) -> str:
    text = re.sub(r'\[tool.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[SECURITY.*?\].*?(?=\n\n|\n$|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`.*?`', '', text)
    text = re.sub(r'Jarvus> ?', '', text)
    text = re.sub(r'\n+', '. ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()[:600]


class JarvusVoice:
    def __init__(
        self,
        agent: JarvusAgent,
        *,
        listener: VoiceListener,
        tts: TextToSpeech,
        always_speak: bool = True,
        auto_brief: bool = False,
    ):
        self.agent = agent
        self.listener = listener
        self.tts = tts
        self.always_speak = always_speak
        self.auto_brief = auto_brief
        self.processing = threading.Event()
        self.listener.callback = self.handle_voice

    def speak(self, text: str) -> None:
        clean = clean_for_speech(text)
        if not clean:
            return
        self.listener.paused.set()
        try:
            self.tts.speak(clean)
            time.sleep(0.35)  # let room echo of the tail decay before re-arming the mic
        finally:
            self.listener.paused.clear()

    def handle_voice(self, text: str) -> None:
        if self.processing.is_set():
            return
        self.processing.set()
        try:
            if not text:
                self.speak("Yes?")
                self.listener.expect_command.set()
                return
            print(f"\nYou said: {text}")
            res = self.agent.process_user_text(text)
            print(f"Jarvus: {res}\n")
            if self.always_speak:
                self.speak(res)
            # Jarvus asked a question - take the next utterance as the answer
            # even without the wake word, so follow-ups aren't dropped.
            if res.strip().endswith("?"):
                self.listener.expect_command.set()
        except Exception as e:
            log.exception("voice turn failed")
            print(f"Error: {e}")
            if self.always_speak:
                self.speak("Sorry, an error occurred")
        finally:
            self.processing.clear()

    def voice_confirmer(self, tool_name: str, summary: str) -> bool:
        """Ask the user by voice to confirm a HIGH-risk tool. Runs on the agent worker thread."""
        self.listener.paused.set()
        try:
            self.tts.speak(f"Jarvus requests {tool_name.replace('_', ' ')}. Say yes to confirm, or anything else to cancel.")
            time.sleep(0.3)
            answer = self.listener.listen_once(timeout=10.0).lower()
            if not answer.strip():
                # Right after TTS there is a short mic re-arm window - a fast
                # 'yes' can be eaten. Ask once more before denying.
                self.tts.speak("I didn't hear you. Say yes to confirm, or no to cancel.")
                time.sleep(0.3)
                answer = self.listener.listen_once(timeout=8.0).lower()
            # Whisper adds punctuation ('yes.') and the hotword bias can prepend
            # the wake word ('jarvus, yes.') - normalize both away. Still
            # exact-match so vague answers like 'yes i think' deny.
            normalized = re.sub(r"[^a-z ]", "", answer)
            normalized = re.sub(r"\b(?:hey\s+)?(?:jarvus|jarvis|please)\b", "", normalized)
            normalized = re.sub(r"\s+", " ", normalized).strip()
            approved = normalized in ("yes", "yeah", "yep", "yup", "yes please", "confirm", "sure",
                                      "ok", "okay", "go ahead", "do it", "absolutely", "affirmative")
            self.speak("Confirmed." if approved else f"Cancelled, I heard '{answer or 'nothing'}'.")
            log.info("Voice confirm %s -> %s (%r)", tool_name, "APPROVED" if approved else "denied", answer)
            return approved
        finally:
            self.listener.paused.clear()

    def start(self) -> None:
        self.speak("Jarvus is online.")
        if self.auto_brief:
            from app.tools.briefing import maybe_auto_brief
            digest = maybe_auto_brief(getattr(self.agent, "_memory", None), enabled=True)
            if digest:
                self.speak(digest)
        self.listener.paused.clear()
        self.listener.start()
