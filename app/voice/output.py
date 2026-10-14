"""Voice output: TTS. Prefers the configured engine (Kokoro), falls back to macOS 'say'."""
from __future__ import annotations

import logging
import subprocess
from typing import Optional

log = logging.getLogger("jarvus.voice.output")

DEFAULT_VOICE = "Samantha"


def is_available() -> bool:
    try:
        subprocess.run(["which", "say"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def speak_text(text: str, *, voice: Optional[str] = None, rate: Optional[int] = None) -> bool:
    """Speak text aloud. Uses the configured TTS engine when available."""
    if not text or not text.strip():
        return False
    try:
        from app.config import load_settings
        from app.voice.tts import get_tts
        s = load_settings()
        get_tts(engine=s.tts_engine, voice=voice or s.tts_voice, rate=rate or s.tts_rate).speak(text)
        return True
    except Exception as e:
        log.warning("TTS failed: %s", e)
        return False


def get_available_voices() -> list[str]:
    """List available macOS voices (fallback engine)."""
    try:
        result = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=10)
        voices = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line and "#" in line:
                voice_name = line.split("#")[0].strip()
                if voice_name:
                    voices.append(voice_name)
        return voices
    except Exception:
        return []
