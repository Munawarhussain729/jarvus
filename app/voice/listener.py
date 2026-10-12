"""Voice input listener with wake word and continuous mode - 100% local STT."""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

log = logging.getLogger("jarvus.voice.listener")

try:
    import speech_recognition as sr
except ImportError:
    sr = None


class VoiceListener:
    def __init__(
        self,
        wake_words: Optional[list[str]] = None,
        callback: Optional[Callable[[str], None]] = None,
        use_wake_word: bool = False,
    ):
        self.wake_words = [w.lower() for w in (wake_words or ["jarvis", "jarvus", "hey jarvis", "hey jarvus"])]
        self.callback = callback
        self.listening = False
        self.should_stop = False
        self._thread: Optional[threading.Thread] = None
        self.recognizer = sr.Recognizer() if sr else None
        self.microphone = sr.Microphone() if sr else None
        self.use_wake_word = use_wake_word
        self._model = None

    def _get_model(self):
        if self._model is not None:
            return self._model
        from faster_whisper import WhisperModel
        log.info("Loading whisper base model...")
        self._model = WhisperModel("base", device="cpu", compute_type="int8")
        log.info("Whisper model ready")
        return self._model

    def _adjust_for_noise(self):
        if self.microphone and self.recognizer:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.3)

    def _contains_wake_word(self, text: str) -> bool:
        text_low = text.lower()
        return any(w in text_low for w in self.wake_words)

    def _extract_command(self, text: str) -> str:
        text_low = text.lower()
        for w in self.wake_words:
            if w in text_low:
                return text_low.replace(w, "").strip()
        return text

    def _transcribe(self, audio) -> str:
        import numpy as np
        raw_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
        audio_array = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        model = self._get_model()
        segments, _ = model.transcribe(audio_array, beam_size=1, vad_filter=True)
        return " ".join(seg.text.strip() for seg in segments).strip()

    def _listen_continuous(self):
        log.info("Voice listener started (wake_word=%s, local STT)", self.use_wake_word)
        if not self.recognizer or not self.microphone:
            log.error("speech_recognition not available")
            return

        self._adjust_for_noise()

        while not self.should_stop:
            try:
                with self.microphone as source:
                    try:
                        audio = self.recognizer.listen(source, timeout=0.5, phrase_time_limit=6)
                    except sr.WaitTimeoutError:
                        continue

                try:
                    text = self._transcribe(audio)
                except Exception as e:
                    log.debug("Transcription error: %s", e)
                    continue

                if not text:
                    continue

                text_low = text.lower()
                log.debug("Heard: %s", text)

                if "ok go to sleep" in text_low or "go to sleep" in text_low or "stop listening" in text_low:
                    log.info("Stop phrase detected")
                    self.stop()
                    break

                if self.use_wake_word:
                    if self._contains_wake_word(text_low):
                        cmd = self._extract_command(text)
                        if cmd and self.callback:
                            self.callback(cmd)
                else:
                    if self.callback:
                        self.callback(text)

            except Exception as e:
                log.debug("Listen loop error: %s", e)
                time.sleep(0.1)

        self.listening = False
        log.info("Voice listener stopped")

    def start(self):
        if self.listening:
            return
        if not self.recognizer or not self.microphone:
            log.error("Cannot start listener: speech_recognition not available")
            return
        self.should_stop = False
        self._thread = threading.Thread(target=self._listen_continuous, daemon=True)
        self._thread.start()
        self.listening = True

    def stop(self):
        self.should_stop = True
        self.listening = False
