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

_SENTENCE_END_RE = re.compile(r"[.!?]+[\"')\]]*\s+|\n+")


class _SentenceStreamer:
    """Feeds streamed LLM deltas to TTS one complete sentence at a time, so
    speech starts while the model is still generating. Buffer is dropped if
    the turn ends in tool calls instead of a spoken reply."""

    def __init__(self, speak):
        self._speak = speak
        self._buf = ""
        self.spoken = 0

    def feed(self, delta: str) -> None:
        self._buf += delta
        while True:
            m = _SENTENCE_END_RE.search(self._buf)
            if not m:
                break
            sent, self._buf = self._buf[:m.end()], self._buf[m.end():]
            sent = clean_for_speech(sent)
            if sent:
                self.spoken += 1
                self._speak(sent)

    def flush(self) -> None:
        rest = clean_for_speech(self._buf)
        self._buf = ""
        if rest:
            self.spoken += 1
            self._speak(rest)


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
            stt_ms = getattr(self.listener, "last_stt_ms", 0)
            streamer = _SentenceStreamer(self.speak) if self.always_speak else None
            t0 = time.monotonic()
            res = self.agent.process_user_text(
                text, on_delta=streamer.feed if streamer else None)
            agent_ms = int((time.monotonic() - t0) * 1000)
            print(f"Jarvus: {res}\n")
            t1 = time.monotonic()
            if self.always_speak:
                if streamer.spoken:
                    streamer.flush()
                else:
                    self.speak(res)
            log.info("turn_timing stt_ms=%d agent_ms=%d tts_ms=%d",
                     stt_ms, agent_ms, int((time.monotonic() - t1) * 1000))
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
