"""Voice input: local STT using faster-whisper (100% offline)."""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

log = logging.getLogger("jarvus.voice.input")

_whisper_model = None
_whisper_lock = threading.Lock()


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    with _whisper_lock:
        if _whisper_model is not None:
            return _whisper_model
        from faster_whisper import WhisperModel
        log.info("Loading whisper base model (local STT)...")
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        log.info("Whisper model loaded")
        return _whisper_model


def is_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def transcribe_audio_data(audio_data, sample_rate: int = 16000) -> str:
    """Transcribe numpy audio data to text using local whisper."""
    model = _get_whisper_model()
    segments, _ = model.transcribe(audio_data, beam_size=1, vad_filter=True)
    text = " ".join(seg.text.strip() for seg in segments)
    return text.strip()


def listen_and_transcribe_local(
    *,
    timeout: float = 5.0,
    phrase_time_limit: float = 10.0,
    adjust_for_noise: bool = True,
) -> str:
    """Listen to microphone and transcribe using local whisper. Returns empty string on failure."""
    try:
        import speech_recognition as sr
    except ImportError:
        raise RuntimeError("speech_recognition not installed")

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        if adjust_for_noise:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            return ""

    raw_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
    import numpy as np
    audio_array = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0

    try:
        return transcribe_audio_data(audio_array)
    except Exception as e:
        log.warning("Local transcription failed: %s", e)
        return ""


def listen_continuous_local(
    callback,
    *,
    wake_words: Optional[list[str]] = None,
    stop_phrases: Optional[list[str]] = None,
    timeout: float = 1.0,
    phrase_time_limit: float = 8.0,
) -> None:
    """Continuously listen using local whisper STT. Calls callback with transcribed text."""
    try:
        import speech_recognition as sr
    except ImportError:
        raise RuntimeError("speech_recognition not installed")

    wake_words = [w.lower() for w in (wake_words or [])]
    stop_phrases = [p.lower() for p in (stop_phrases or ["ok go to sleep", "go to sleep", "stop listening"])]

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    use_wake_word = len(wake_words) > 0

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.3)
        while True:
            try:
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            except sr.WaitTimeoutError:
                continue

            raw_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
            import numpy as np
            audio_array = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0

            try:
                text = transcribe_audio_data(audio_array)
            except Exception as e:
                log.debug("Transcription error: %s", e)
                continue

            if not text:
                continue

            text_lower = text.lower()
            log.debug("Heard: %s", text)

            if any(sp in text_lower for sp in stop_phrases):
                return

            if use_wake_word:
                for w in wake_words:
                    if w in text_lower:
                        cmd = text_lower.replace(w, "").strip()
                        if cmd:
                            callback(cmd)
                        break
            else:
                callback(text)
