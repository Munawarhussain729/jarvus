"""Voice output: TTS using macOS `say` command."""
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
    """Speak text aloud using macOS TTS.

    Returns True if speech was initiated successfully.
    """
    if not text or not text.strip():
        return False

    cmd = ["say"]
    if voice:
        cmd.extend(["-v", voice])
    if rate:
        cmd.extend(["-r", str(rate)])
    cmd.append(text)

    try:
        subprocess.run(cmd, capture_output=True, timeout=120, check=False)
        return True
    except subprocess.TimeoutExpired:
        log.warning("TTS timed out")
        return False
    except Exception as e:
        log.warning("TTS failed: %s", e)
        return False


def get_available_voices() -> list[str]:
    """List available macOS voices."""
    try:
        result = subprocess.run(
            ["say", "-v", "?"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
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
