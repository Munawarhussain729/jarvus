"""Voice session - orchestrates listener, agent, TTS, and voice confirmations."""
from __future__ import annotations

import logging
import queue
import re
import threading
import time

from app.assistant.agent import JarvusAgent
from app.voice.listener import VoiceListener
from app.voice.tts import TextToSpeech

log = logging.getLogger("jarvus.voice.session")

_SENTENCE_END_RE = re.compile(r"[.!?]+[\"')\]]*\s+|\n+")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
# Prefixes of markup that must never reach TTS. A trailing partial match is
# held back until the next delta proves or disproves it.
_TAG_HEADS = ("<think>", "```", "<|", "[tool", "[security")


class _StreamSanitizer:
    """Removes markup (<think> blocks, code fences, [tool]/[SECURITY] lines)
    from a token stream *before* sentences are formed for TTS. Streaming-safe:
    holds back a short trailing prefix that could complete into a tag opener."""

    def __init__(self) -> None:
        self._raw = ""
        self._emitted = 0

    def _sanitized(self, final: bool) -> str:
        t = _THINK_BLOCK_RE.sub("", self._raw)
        t = _CODE_FENCE_RE.sub("", t)
        if final:  # drop unclosed tail tags at end of round
            t = re.sub(r"<think>.*", "", t, flags=re.DOTALL | re.IGNORECASE)
            t = re.sub(r"```.*", "", t, flags=re.DOTALL)
            t = re.sub(r"<\|.*", "", t, flags=re.DOTALL)
        return t

    def feed(self, delta: str) -> str:
        self._raw += delta
        t = self._sanitized(final=False)
        safe = len(t)
        i = self._emitted
        while i < len(t):
            # next markup-looking character after what we already emitted
            idx = min((j for j in (t.find(m, i) for m in "<`[") if j != -1), default=len(t))
            if idx >= len(t):
                break
            suffix = t[idx:]
            hold = False
            if suffix[:7].lower() == "<think>":
                # unclosed think block: everything after is monologue, hold it all
                if not re.search(r"</think\s*>", suffix, re.IGNORECASE):
                    hold = True
            elif suffix.startswith("```"):
                if "```" not in suffix[3:]:
                    hold = True  # unclosed fence
            elif suffix.startswith("<|"):
                hold = True  # <|im_start|>-style control token
            elif len(suffix) < 10 and any(h.startswith(suffix) for h in _TAG_HEADS):
                hold = True  # possible partial opener - wait for the rest
            if hold:
                safe = idx
                break
            i = idx + 1
        out = t[self._emitted:safe]
        self._emitted = safe
        return out

    def finish(self) -> str:
        t = self._sanitized(final=True)
        out = t[self._emitted:]
        self._emitted = len(t)
        return out


class _SpeechPipeline:
    """Bounded queue between LLM deltas and the speaker worker.

    Solves three real bugs at once:
    - the stream reader never blocks on TTS playback (a slow 'say' call no
      longer stalls generation or drops the HTTP read timeout);
    - markup is stripped before sentences form, so reasoning/code never leak
      into speech, including tags split across streamed chunks;
    - each sentence is deduplicated, so a nudged retry can't re-speak text the
      user already heard.
    """

    MAX_QUEUE = 8

    def __init__(self, session: "JarvusVoice") -> None:
        self._session = session
        self._san = _StreamSanitizer()
        self._buf = ""
        self.spoken = 0
        self._seen: set[str] = set()

    def feed(self, delta: str) -> None:
        self._buf += self._san.feed(delta)
        while True:
            m = _SENTENCE_END_RE.search(self._buf)
            if not m:
                break
            sent, self._buf = self._buf[:m.end()], self._buf[m.end():]
            self._enqueue(sent)

    def discard_pending(self) -> None:
        """Drop unsentenced text from an aborted round (tool calls coming)."""
        self._buf = ""
        self._san = _StreamSanitizer()

    def _enqueue(self, text: str) -> None:
        if self._session._interrupt.is_set():
            return  # user barged in - drop speech still streaming from the LLM
        sent = clean_for_speech(text)
        if not sent or sent in self._seen:
            return
        self._seen.add(sent)
        self.spoken += 1
        try:
            self._session._speech_q.put_nowait(sent)
        except queue.Full:
            log.warning("speech queue full - dropping sentence")

    def finish(self) -> None:
        self._buf += self._san.finish()
        if self._buf:
            rest, self._buf = self._buf, ""
            self._enqueue(rest)


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
        self.listener.on_interrupt = self.interrupt_speech
        # Set by interrupt_speech (barge-in / Enter key): kills current
        # playback and suppresses queued + still-streaming speech this turn.
        self._interrupt = threading.Event()
        self._speech_q: queue.Queue[str] = queue.Queue(maxsize=_SpeechPipeline.MAX_QUEUE)
        self._speaker_stop = threading.Event()
        self._speaker_thread: threading.Thread | None = None

    def _speaker_loop(self) -> None:
        """Serial TTS playback. Holds the mic drain (paused) across the whole
        queue so echo between consecutive sentences can't reach the mic."""
        while not self._speaker_stop.is_set():
            try:
                text = self._speech_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if self._interrupt.is_set():
                self._speech_q.task_done()
                continue
            self.listener.paused.set()
            self.listener.state = "speaking"
            try:
                self.tts.speak(text)
                time.sleep(0.35)  # let room echo of the tail decay
            finally:
                self._speech_q.task_done()
                if self._speech_q.empty():
                    self.listener.paused.clear()
                    self.listener.state = "idle"

    def _ensure_speaker(self) -> None:
        if self._speaker_thread is None or not self._speaker_thread.is_alive():
            self._speaker_thread = threading.Thread(target=self._speaker_loop, daemon=True)
            self._speaker_thread.start()

    def speak(self, text: str) -> None:
        clean = clean_for_speech(text)
        if not clean:
            return
        self.listener.paused.set()
        prev_state = self.listener.state
        self.listener.state = "speaking"
        try:
            self.tts.speak(clean)
            time.sleep(0.35)  # let room echo of the tail decay before re-arming the mic
        finally:
            self.listener.state = prev_state
            self.listener.paused.clear()

    def interrupt_speech(self) -> None:
        """Barge-in (voice 'stop' or Enter key): kill current playback, drop
        queued sentences, and suppress speech still streaming from the LLM."""
        self._interrupt.set()
        self.tts.stop()
        while True:
            try:
                self._speech_q.get_nowait()
            except queue.Empty:
                break
            self._speech_q.task_done()

    def handle_voice(self, text: str) -> None:
        if self.processing.is_set():
            return
        self.processing.set()
        self._interrupt.clear()  # new turn: any prior barge-in is spent
        self.listener.state = "processing"
        try:
            if not text:
                self.speak("Yes?")
                self.listener.arm_followup()
                return
            print(f"\nYou said: {text}")
            stt_ms = getattr(self.listener, "last_stt_ms", 0)
            pipe = _SpeechPipeline(self) if self.always_speak else None
            if pipe:
                self._ensure_speaker()
            t0 = time.monotonic()
            res = self.agent.process_user_text(
                text, on_delta=pipe.feed if pipe else None, speech=pipe)
            agent_ms = int((time.monotonic() - t0) * 1000)
            print(f"Jarvus: {res}\n")
            t1 = time.monotonic()
            if self.always_speak:
                if pipe.spoken:
                    pipe.finish()
                    self._speech_q.join()  # wait for serial playback to drain
                else:
                    self.speak(res)
            tts_ms = int((time.monotonic() - t1) * 1000)
            log.info("turn_timing stt_ms=%d agent_ms=%d tts_ms=%d", stt_ms, agent_ms, tts_ms)
            try:
                from app.telemetry import record
                record("turn", stt_ms=stt_ms, agent_ms=agent_ms, tts_ms=tts_ms)
            except Exception:
                pass
            # Jarvus asked a question - take the next utterance as the answer
            # even without the wake word, so follow-ups aren't dropped (expiry
            # prevents a stale arm from swallowing a later utterance).
            if res.strip().endswith("?"):
                self.listener.arm_followup()
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
        self.listener.state = "confirming"
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
            self.listener.state = "idle"

    def stop(self) -> None:
        self._speaker_stop.set()
        self.interrupt_speech()  # also kill any in-flight TTS playback
        if self._speaker_thread:
            self._speaker_thread.join(timeout=2)

    def start(self) -> None:
        self._interrupt.clear()
        self._ensure_speaker()
        self.speak("Jarvus is online.")
        if self.auto_brief:
            from app.tools.briefing import maybe_auto_brief
            digest = maybe_auto_brief(getattr(self.agent, "_memory", None), enabled=True)
            if digest:
                self.speak(digest)
        self.listener.paused.clear()
        self.listener.state = "listening"
        self.listener.start()
