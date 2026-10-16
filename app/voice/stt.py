"""Speech-to-text engines - 100% local. MLX-whisper on Apple Silicon, faster-whisper elsewhere."""
from __future__ import annotations

import logging
import platform
import re
import threading
import zlib
from typing import Protocol

import numpy as np

log = logging.getLogger("jarvus.voice.stt")

SAMPLE_RATE = 16000

# Whisper's classic hallucinations on silence/ambient noise - phrases that are
# never real commands. NOTE: yes/no/okay/yeah are excluded on purpose - they
# are valid answers to voice confirmations.
_HALLUCINATIONS = {
    "thank you", "thanks", "thanks for watching", "thank you for watching",
    "thanks for listening", "thank you for listening", "bye", "goodbye",
    "see you", "see you next time", "all right", "alright", "hmm", "mhm",
    "um", "uh", "so", "you", "like and subscribe", "subscribe",
    "subtitles by", "amaraorg", "please subscribe", "copyright",
    "here we go", "like that", "there you go", "youre welcome",
    "take care", "stay safe", "see ya", "see you later", "have a good day",
    "have a nice day", "lets go", "come on", "hold on",
}

# Whisper's own decode-failure bounds: below _MIN_AVG_LOGPROB the model is
# guessing at noise; above _MAX_COMPRESSION_RATIO the decoder looped on a
# repeated phrase (the classic "this link to find all of this link ..." spam).
_MIN_AVG_LOGPROB = -1.0
_MAX_COMPRESSION_RATIO = 2.4


def _compression_ratio(text: str) -> float:
    raw = text.encode("utf-8")
    if not raw:
        return 0.0
    return len(raw) / max(len(zlib.compress(raw)), 1)


def _is_repetitive(text: str) -> bool:
    """Decoder-loop fingerprint: a long transcript built from a tiny vocabulary."""
    words = re.sub(r"[^a-z ]", " ", text.lower()).split()
    return len(words) >= 10 and len(set(words)) / len(words) < 0.4


def _is_hallucination(text: str) -> bool:
    t = re.sub(r"[^a-z ]", "", text.lower()).strip()
    if not t or t in _HALLUCINATIONS:
        return True
    return _is_repetitive(text) or _compression_ratio(t) > _MAX_COMPRESSION_RATIO


def has_speech(audio_f32: np.ndarray, *, rms_gate: float = 0.003,
               min_voiced_frac: float = 0.12, frame_ms: int = 50) -> bool:
    """Cheap VAD before paying whisper inference.

    Overall-RMS alone passes on one loud click; requiring a minimum fraction of
    50ms frames above the gate rejects clicks/thumps while keeping real speech
    (which has pauses but sustained voiced regions)."""
    if not audio_f32.size:
        return False
    frame = max(1, int(SAMPLE_RATE * frame_ms / 1000))
    n_frames = audio_f32.size // frame
    if n_frames < 2:
        # Very short chunk: fall back to plain RMS.
        return float(np.sqrt(np.mean(audio_f32 * audio_f32))) >= rms_gate
    framed = audio_f32[: n_frames * frame].reshape(n_frames, frame)
    rms = np.sqrt(np.mean(framed * framed, axis=1))
    return float(np.mean(rms >= rms_gate)) >= min_voiced_frac


def _keep_segment(seg: dict, *, no_speech_threshold: float = 0.6,
                  min_logprob: float = _MIN_AVG_LOGPROB) -> bool:
    """Per-segment filter using whisper's decoder confidence signals."""
    if seg.get("no_speech_prob", 0.0) > no_speech_threshold:
        return False
    if seg.get("avg_logprob", 0.0) < min_logprob:
        return False
    ratio = seg.get("compression_ratio")
    if ratio is None:
        ratio = _compression_ratio(seg.get("text") or "")
    return ratio <= _MAX_COMPRESSION_RATIO


# Bias whisper toward the assistant's vocabulary so commands land correctly.
HOTWORDS_PROMPT = (
    "Jarvus. Hey Jarvus. Open WhatsApp, Safari, Finder, Messages, Mail, Notes, "
    "Calendar, Music, Spotify, Slack, Terminal, Microsoft Word, Excel, Telegram, "
    "Chrome. Read my notifications, set a timer, weather forecast, reminder, alarm, "
    "volume, brightness, mute, screenshot, clipboard, play music, pause, "
    "next track, send a message, email, notes, web search, Wikipedia, find files, "
    "read file, run command, system info, battery, dark mode, lock screen, "
    "Bitcoin price, stock price, currency, convert miles to kilometers, "
    "calculate, news, headlines, password, coin flip, what's my IP address."
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

    def __init__(self, model: str = "turbo", *, no_speech_threshold: float = 0.6,
                 min_logprob: float = _MIN_AVG_LOGPROB) -> None:
        import mlx_whisper  # noqa: F401
        self._model = model if "/" in model else MLX_MODELS.get(model, f"mlx-community/whisper-{model}-mlx")
        self._no_speech = no_speech_threshold
        self._min_logprob = min_logprob
        log.info("MLX-whisper STT ready (%s)", self._model)

    def transcribe(self, audio_f32: np.ndarray) -> str:
        import mlx_whisper
        result = mlx_whisper.transcribe(
            audio_f32,
            path_or_hf_repo=self._model,
            condition_on_previous_text=False,
            initial_prompt=HOTWORDS_PROMPT,
            no_speech_threshold=self._no_speech,
        )
        # Drop segments the decoder itself flags as non-speech, low-confidence,
        # or looped - this is where silence/noise hallucinations come from.
        segments = result.get("segments") or []
        if segments:
            text = " ".join(
                s.get("text", "").strip() for s in segments
                if _keep_segment(s, no_speech_threshold=self._no_speech, min_logprob=self._min_logprob)
            ).strip()
        else:
            text = (result.get("text") or "").strip()
        return "" if _is_hallucination(text) else text


class FasterWhisperSTT:
    """faster-whisper (CTranslate2): portable CPU/GPU fallback."""

    name = "faster-whisper"

    def __init__(self, model: str = "small", *, no_speech_threshold: float = 0.6,
                 min_logprob: float = _MIN_AVG_LOGPROB) -> None:
        from faster_whisper import WhisperModel
        size = model if "/" not in model else model.rsplit("/", 1)[-1]
        if size in ("turbo", "large-v3-turbo"):
            size = "large-v3"
        self._model = WhisperModel(size, device="cpu", compute_type="int8")
        self._no_speech = no_speech_threshold
        self._min_logprob = min_logprob

    def transcribe(self, audio_f32: np.ndarray) -> str:
        segments, _ = self._model.transcribe(
            audio_f32, beam_size=1, vad_filter=True, initial_prompt=HOTWORDS_PROMPT
        )
        segs = list(segments)
        kept = []
        for s in segs:
            d = {
                "no_speech_prob": getattr(s, "no_speech_prob", 0.0),
                "avg_logprob": getattr(s, "avg_logprob", 0.0),
                "compression_ratio": _compression_ratio(getattr(s, "text", "") or ""),
                "text": getattr(s, "text", ""),
            }
            if _keep_segment(d, no_speech_threshold=self._no_speech, min_logprob=self._min_logprob):
                kept.append(d["text"])
        text = " ".join(t.strip() for t in kept).strip()
        return "" if _is_hallucination(text) else text


_lock = threading.Lock()
_engine: SpeechToText | None = None


def get_stt(engine: str = "auto", model: str = "turbo", *,
            no_speech_threshold: float = 0.6,
            min_logprob: float = _MIN_AVG_LOGPROB) -> SpeechToText:
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
                _engine = cls(model, no_speech_threshold=no_speech_threshold, min_logprob=min_logprob)
                return _engine
            except Exception as e:
                log.warning("STT engine %s unavailable: %s", cls.__name__, e)
        raise RuntimeError("No local STT engine available (install mlx-whisper or faster-whisper)")
