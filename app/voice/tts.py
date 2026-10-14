"""Text-to-speech engines - 100% local. Kokoro (natural voice) with macOS 'say' fallback."""
from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from typing import Protocol

log = logging.getLogger("jarvus.voice.tts")


class TextToSpeech(Protocol):
    name: str

    def speak(self, text: str) -> None:
        """Speak text, blocking until finished or stop() is called."""

    def stop(self) -> None:
        """Interrupt current speech as soon as possible."""


class KokoroTTS:
    """Kokoro-82M via mlx-audio: natural neural voice on Apple Silicon.

    Generates a wav to a temp dir, then plays it via afplay so stop() can
    interrupt playback instantly.
    """

    name = "kokoro"
    MODEL_REPO = "mlx-community/Kokoro-82M-bf16"

    def __init__(self, voice: str = "af_heart", speed: float = 1.0) -> None:
        from mlx_audio.tts.utils import load_model
        self._voice = voice
        self._speed = speed
        self._stop = threading.Event()
        self._model = load_model(self.MODEL_REPO)
        self._player: subprocess.Popen | None = None
        self._plock = threading.Lock()
        log.info("Kokoro TTS ready (voice=%s)", voice)

    def speak(self, text: str) -> None:
        import shutil
        import tempfile
        from mlx_audio.tts.generate import generate_audio
        self._stop.clear()
        out_dir = tempfile.mkdtemp(prefix="jarvus_tts_")
        try:
            generate_audio(
                text=text,
                model=self._model,
                voice=self._voice,
                speed=self._speed,
                output_path=out_dir,
                join_audio=True,
                play=False,
                verbose=False,
            )
            wav = os.path.join(out_dir, "audio.wav")
            if self._stop.is_set() or not os.path.exists(wav):
                return
            with self._plock:
                self._player = subprocess.Popen(
                    ["afplay", wav], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            while self._player.poll() is None:
                if self._stop.is_set():
                    self._player.kill()
                    break
                time.sleep(0.04)
        except Exception as e:
            log.warning("Kokoro TTS failed: %s", e)
            SayTTS().speak(text)
        finally:
            with self._plock:
                self._player = None
            shutil.rmtree(out_dir, ignore_errors=True)

    def stop(self) -> None:
        self._stop.set()
        with self._plock:
            if self._player and self._player.poll() is None:
                self._player.kill()


class SayTTS:
    """macOS built-in 'say' - always available, interruptible via Popen."""

    name = "say"

    def __init__(self, voice: str | None = None, rate: int = 200) -> None:
        self._voice = voice
        self._rate = rate
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def speak(self, text: str) -> None:
        cmd = ["say"]
        if self._voice:
            cmd += ["-v", self._voice]
        if self._rate:
            cmd += ["-r", str(self._rate)]
        cmd.append(text)
        try:
            with self._lock:
                self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            while self._proc.poll() is None:
                time.sleep(0.05)
        except Exception as e:
            log.warning("say TTS failed: %s", e)
        finally:
            with self._lock:
                self._proc = None

    def stop(self) -> None:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                self._proc.kill()


_tts: TextToSpeech | None = None
_tts_lock = threading.Lock()


def get_tts(engine: str = "auto", voice: str = "af_heart", rate: int = 200) -> TextToSpeech:
    global _tts
    with _tts_lock:
        if _tts is not None:
            return _tts
        if engine in ("auto", "kokoro"):
            try:
                _tts = KokoroTTS(voice=voice)
                return _tts
            except Exception as e:
                log.warning("Kokoro TTS unavailable (%s); falling back to 'say'", e)
                if engine == "kokoro":
                    raise
        mac_voice = voice if voice and not voice.startswith(("af_", "am_", "bf_", "bm_")) else None
        _tts = SayTTS(voice=mac_voice, rate=rate)
        return _tts
