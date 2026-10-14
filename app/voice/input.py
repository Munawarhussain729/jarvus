"""Voice input helpers - thin wrappers over the STT engine registry (100% local)."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from app.voice.stt import SAMPLE_RATE, get_stt

log = logging.getLogger("jarvus.voice.input")


def is_available() -> bool:
    try:
        import speech_recognition  # noqa: F401
        return True
    except ImportError:
        return False


def transcribe_audio_data(audio_data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    return get_stt().transcribe(audio_data)


def listen_and_transcribe_local(
    *,
    timeout: float = 5.0,
    phrase_time_limit: float = 10.0,
    adjust_for_noise: bool = True,
    stt_engine: str = "auto",
    stt_model: str = "turbo",
) -> str:
    """One-shot: listen to the mic and transcribe locally. Returns '' on failure."""
    try:
        import speech_recognition as sr
    except ImportError:
        raise RuntimeError("speech_recognition not installed")

    recognizer = sr.Recognizer()
    microphone = sr.Microphone(sample_rate=SAMPLE_RATE)
    with microphone as source:
        if adjust_for_noise:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            return ""
    raw = audio.get_raw_data(convert_rate=SAMPLE_RATE, convert_width=2)
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    try:
        return get_stt(stt_engine, stt_model).transcribe(arr)
    except Exception as e:
        log.warning("Local transcription failed: %s", e)
        return ""


def listen_continuous_local(
    callback,
    *,
    wake_words: Optional[list[str]] = None,
    stop_phrases: Optional[list[str]] = None,
    **kwargs,
) -> None:
    """Blocking continuous listen; delegates to VoiceListener."""
    from app.voice.listener import VoiceListener
    listener = VoiceListener(callback=callback, wake_words=wake_words, stop_phrases=stop_phrases,
                             use_wake_word=bool(wake_words))
    listener.start()
    try:
        while listener.listening:
            import time
            time.sleep(0.5)
    except KeyboardInterrupt:
        listener.stop()
