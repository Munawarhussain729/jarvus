"""Voice input listener - continuous capture on one persistent mic stream.

Runs 100% local STT. The capture loop never stops draining the audio buffer;
while `paused` is set (i.e. during TTS playback) audio is discarded so Jarvus
cannot hear and re-trigger itself - except while `state == "speaking"`, where
chunks are still transcribed and matched against `interrupt_phrases` so the
user can barge in ("stop", "jarvis stop", ...). Commands are dispatched on a
worker thread so the stream keeps draining during processing and playback.
"""
from __future__ import annotations

import logging
import queue
import re
import threading
import time
from typing import Callable, Optional

import numpy as np

from app.voice.stt import SAMPLE_RATE, SpeechToText, get_stt, has_speech

log = logging.getLogger("jarvus.voice.listener")

DEFAULT_WAKE_WORDS = ["jarvis", "jarvus", "hey jarvis", "hey jarvus"]
DEFAULT_STOP_PHRASES = ["go to sleep", "stop listening", "ok go to sleep"]
# Exact-match barge-in phrases heard during TTS playback. Exact match (after
# normalization + wake-word strip) so echo of Jarvus's own sentences - which
# transcribe to longer strings - can't self-interrupt.
DEFAULT_INTERRUPT_PHRASES = [
    "stop", "stop it", "cancel", "be quiet", "quiet", "enough",
    "shut up", "never mind", "nevermind", "shush",
]
# Below this RMS (float32 audio) a chunk is effectively silence - skip the
# whisper inference entirely instead of paying GPU time for a hallucination.
DEFAULT_SILENCE_RMS = 0.003

try:
    import speech_recognition as sr
except ImportError:
    sr = None


class VoiceListener:
    def __init__(
        self,
        *,
        stt: SpeechToText | None = None,
        wake_words: Optional[list[str]] = None,
        stop_phrases: Optional[list[str]] = None,
        interrupt_phrases: Optional[list[str]] = None,
        callback: Optional[Callable[[str], None]] = None,
        on_interrupt: Optional[Callable[[], None]] = None,
        use_wake_word: bool = False,
        stt_engine: str = "auto",
        stt_model: str = "turbo",
        stt_no_speech_threshold: float = 0.6,
        stt_min_logprob: float = -1.0,
        silence_rms: float = DEFAULT_SILENCE_RMS,
    ):
        self.wake_words = [w.lower() for w in (wake_words or DEFAULT_WAKE_WORDS)]
        self.stop_phrases = [p.lower() for p in (stop_phrases or DEFAULT_STOP_PHRASES)]
        self.interrupt_phrases = [p.lower() for p in (interrupt_phrases or DEFAULT_INTERRUPT_PHRASES)]
        self.callback = callback
        # Called (on the capture thread) when an interrupt phrase is heard
        # during TTS playback - the session hooks this to kill speech output.
        self.on_interrupt = on_interrupt
        self.use_wake_word = use_wake_word
        self.listening = False
        self.should_stop = False
        # When set, captured audio is drained and discarded (echo suppression).
        self.paused = threading.Event()
        # When set, the next transcribed phrase is a command even without a wake word
        # (used after Jarvus answers a lone "Jarvis?" with "Yes?"). Auto-expires so
        # an unanswered follow-up can't arm a later utterance forever.
        self.expect_command = threading.Event()
        self._expect_deadline = 0.0
        # Observable pipeline state for logs/diagnostics.
        self.state = "idle"  # idle | listening | processing | speaking | confirming

        self._stt = stt  # lazily resolved via get_stt(engine, model)
        self._stt_engine = stt_engine
        self._stt_model = stt_model
        self._stt_no_speech = stt_no_speech_threshold
        self._stt_min_logprob = stt_min_logprob
        self._silence_rms = silence_rms
        self._one_shot: queue.Queue[str] | None = None
        self._one_shot_lock = threading.Lock()
        self.last_stt_ms = 0  # last transcribe() wall time, for turn_timing logs
        self._thread: Optional[threading.Thread] = None
        self._worker: Optional[threading.Thread] = None

        self.recognizer = sr.Recognizer() if sr else None
        self.microphone = sr.Microphone(sample_rate=SAMPLE_RATE) if sr else None
        if self.recognizer:
            self.recognizer.energy_threshold = 300
            self.recognizer.pause_threshold = 0.7
            self.recognizer.non_speaking_duration = 0.4

    def _engine(self) -> SpeechToText:
        if self._stt is None:
            self._stt = get_stt(
                self._stt_engine, self._stt_model,
                no_speech_threshold=self._stt_no_speech,
                min_logprob=self._stt_min_logprob,
            )
        return self._stt

    def _transcribe(self, audio) -> str:
        raw = audio.get_raw_data(convert_rate=SAMPLE_RATE, convert_width=2)
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if not has_speech(arr, rms_gate=self._silence_rms):
            return ""
        t0 = time.monotonic()
        text = self._engine().transcribe(arr)
        self.last_stt_ms = int((time.monotonic() - t0) * 1000)
        return text

    def _contains_wake_word(self, text: str) -> bool:
        return any(w in text for w in self.wake_words)

    def _extract_command(self, text: str) -> str:
        text_low = text.lower()
        for w in sorted(self.wake_words, key=len, reverse=True):
            idx = text_low.find(w)
            if idx != -1:
                return text[idx + len(w):].strip(" ,.!?")
        return text

    def _check_barge_in(self, audio) -> None:
        """While TTS plays, transcribe drained chunks for an interrupt phrase.

        Only exact matches (after stripping punctuation and any wake word)
        trigger, so a transcribed echo of Jarvus's own playback - normally a
        longer sentence - won't cut its own speech short.
        """
        try:
            text = self._transcribe(audio)
        except Exception as e:
            log.debug("barge-in transcribe error: %s", e)
            return
        if not text:
            return
        log.debug("Heard during playback: %s", text)
        norm = re.sub(r"[^a-z ]", " ", text.lower())
        norm = re.sub(r"\s+", " ", norm).strip()
        if any(sp in norm for sp in self.stop_phrases):
            log.info("Stop phrase detected during playback")
            self.stop()
            return
        cmd = norm
        for w in sorted(self.wake_words, key=len, reverse=True):
            cmd = re.sub(rf"\b{re.escape(w)}\b", " ", cmd)
        cmd = re.sub(r"\s+", " ", cmd).strip()
        if cmd and cmd in self.interrupt_phrases and self.on_interrupt is not None:
            log.info("Barge-in interrupt: %r", text)
            self.on_interrupt()

    def _dispatch(self, text: str) -> None:
        if self.callback is None:
            return
        if self._worker and self._worker.is_alive():
            log.debug("Busy - ignoring overlapping command")
            return
        self._worker = threading.Thread(target=self.callback, args=(text,), daemon=True)
        self._worker.start()

    def _listen_loop(self) -> None:
        log.info("Voice listener started (wake_word=%s)", self.use_wake_word)
        try:
            self._engine()  # warm up STT before opening mic
        except Exception as e:
            log.error("STT unavailable: %s", e)
            return
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            while not self.should_stop:
                # Short phrase chunks while paused (TTS drain) or awaiting a
                # one-shot answer: shrinks the dead window so a quick 'yes'
                # said right after the prompt isn't eaten by a stale capture.
                with self._one_shot_lock:
                    one_shot_pending = self._one_shot is not None
                ptl = 4 if one_shot_pending else (1.5 if self.paused.is_set() else 10)
                try:
                    audio = self.recognizer.listen(source, timeout=0.5, phrase_time_limit=ptl)
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    log.debug("capture error: %s", e)
                    time.sleep(0.1)
                    continue

                if self.paused.is_set():
                    # While speaking, still scan the drained audio for an
                    # interrupt phrase so the user can cut playback short.
                    if self.state == "speaking":
                        self._check_barge_in(audio)
                    continue  # drain + discard: prevents TTS echo / buffer buildup

                with self._one_shot_lock:
                    one_shot = self._one_shot
                if one_shot is not None:
                    try:
                        one_shot.put_nowait(self._transcribe(audio))
                    except queue.Full:
                        log.debug("one-shot answer dropped (queue full)")
                    except Exception as e:
                        log.debug("one-shot transcribe error: %s", e)
                        try:
                            one_shot.put_nowait("")
                        except queue.Full:
                            pass
                    continue

                try:
                    text = self._transcribe(audio)
                except Exception as e:
                    log.debug("Transcription error: %s", e)
                    continue
                if not text:
                    continue

                text_low = text.lower()
                log.info("Heard: %s", text)

                if any(sp in text_low for sp in self.stop_phrases):
                    log.info("Stop phrase detected")
                    self.stop()
                    break

                if self.use_wake_word:
                    if self.expect_command.is_set():
                        if time.monotonic() > self._expect_deadline:
                            self.expect_command.clear()
                            continue  # stale follow-up arm: require the wake word again
                        self.expect_command.clear()
                        self._dispatch(text)
                    elif self._contains_wake_word(text_low):
                        self._dispatch(self._extract_command(text))
                else:
                    self._dispatch(text)
        self.listening = False
        log.info("Voice listener stopped")

    def arm_followup(self, seconds: float = 15.0) -> None:
        """Arm expect_command with an expiry - a next utterance is a command."""
        self._expect_deadline = time.monotonic() + seconds
        self.expect_command.set()

    def listen_once(self, timeout: float = 8.0) -> str:
        """Capture one phrase and return its transcript (used for voice confirmations)."""
        q: queue.Queue[str] = queue.Queue(maxsize=1)
        with self._one_shot_lock:
            self._one_shot = q
        # Loud TTS playback inflates the adaptive energy threshold - cap it so
        # a normal-volume 'yes' right after the prompt isn't filtered out.
        old_threshold = self.recognizer.energy_threshold
        self.recognizer.energy_threshold = min(old_threshold, 200)
        # Register before unpausing so the answer can't be dispatched as a command.
        self.paused.clear()
        try:
            return q.get(timeout=timeout).strip()
        except queue.Empty:
            return ""
        finally:
            with self._one_shot_lock:
                self._one_shot = None
            self.recognizer.energy_threshold = old_threshold
            self.paused.set()  # back to discarding until session resumes listening

    def start(self) -> None:
        if self.listening:
            return
        if not self.recognizer or not self.microphone:
            log.error("Cannot start listener: speech_recognition/pyaudio unavailable")
            return
        self.should_stop = False
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        self.listening = True

    def stop(self) -> None:
        self.should_stop = True
        self.listening = False
