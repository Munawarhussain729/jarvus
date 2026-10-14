"""Speech-to-text engines - 100% local. MLX-whisper on Apple Silicon, faster-whisper elsewhere."""
from __future__ import annotations

import logging
import platform
import threading
from typing import Protocol

import numpy as np

log = logging.getLogger("jarvus.voice.stt")

SAMPLE_RATE = 16000

# Bias whisper toward the assistant's vocabulary so commands land correctly.
HOTWORDS_PROMPT = (
    "Jarvus. Hey Jarvus. Open the app, read my notifications, set a timer, "
    "weather, reminder, calendar, alarm, volume, brightness, mute, screenshot, "
    "clipboard, play music, pause, next track, send a message, email, notes, "
    "web search, list files, read file, screenshot, run command, system info."
)

# Friendly names -> mlx-community HF repo
MLX_MODELS = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


class SpeechToText(Protocol):
    name: str

    def transcribe(self, audio_f32: np.ndarray) -> str:
        """Transcribe 16kHz float32 mono audio to text."""


class MLXWhisperSTT:
    """mlx-whisper: GPU-accelerated Whisper on Apple Silicon."""

    name = "mlx-whisper"

    def __init__(self, model: str = "turbo") -> None:
        import mlx_whisper  # noqa: F401
        self._model = model if "/" in model else MLX_MODELS.get(model, f"mlx-community/whisper-{model}-mlx")
        log.info("MLX-whisper STT ready (%s)", self._model)

    def transcribe(self, audio_f32: np.ndarray) -> str:
        import mlx_whisper
        result = mlx_whisper.transcribe(
            audio_f32,
            path_or_hf_repo=self._model,
            condition_on_previous_text=False,
            initial_prompt=HOTWORDS_PROMPT,
            no_speech_threshold=0.6,
        )
        return (result.get("text") or "").strip()


class FasterWhisperSTT:
    """faster-whisper (CTranslate2): portable CPU/GPU fallback."""

    name = "faster-whisper"

    def __init__(self, model: str = "small") -> None:
        from faster_whisper import WhisperModel
        size = model if "/" not in model else model.rsplit("/", 1)[-1]
        if size in ("turbo", "large-v3-turbo"):
            size = "large-v3"
        self._model = WhisperModel(size, device="cpu", compute_type="int8")

    def transcribe(self, audio_f32: np.ndarray) -> str:
        segments, _ = self._model.transcribe(
            audio_f32, beam_size=1, vad_filter=True, initial_prompt=HOTWORDS_PROMPT
        )
        return " ".join(seg.text.strip() for seg in segments).strip()


_lock = threading.Lock()
_engine: SpeechToText | None = None


def get_stt(engine: str = "auto", model: str = "turbo") -> SpeechToText:
    """Pick the best available local STT engine."""
    global _engine
    with _lock:
        if _engine is not None:
            return _engine
        candidates: list[type] = []
        if engine in ("auto", "mlx") and platform.machine() == "arm64" and platform.system() == "Darwin":
            candidates.append(MLXWhisperSTT)
        if engine in ("auto", "faster-whisper", "whisper"):
            candidates.append(FasterWhisperSTT)
        if engine == "mlx":  # explicit request on non-ARM still tries (will fail clearly)
            candidates.append(MLXWhisperSTT)
        for cls in candidates:
            try:
                _engine = cls(model)
                return _engine
            except Exception as e:
                log.warning("STT engine %s unavailable: %s", cls.__name__, e)
        raise RuntimeError("No local STT engine available (install mlx-whisper or faster-whisper)")
