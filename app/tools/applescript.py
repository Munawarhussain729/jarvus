"""Shared helpers for running AppleScript safely."""
from __future__ import annotations

import subprocess

from app.tools.base import ToolResult


def escape(value: str) -> str:
    """Escape a string for safe interpolation inside AppleScript double quotes."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def run(script: str, *, timeout: int = 15, ok_payload: str = "done") -> ToolResult:
    """Run an AppleScript and normalize the result."""
    try:
        proc = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(ok=False, payload="AppleScript timed out")
    except Exception as e:
        return ToolResult(ok=False, payload=str(e))
    if proc.returncode == 0:
        return ToolResult(ok=True, payload=proc.stdout.strip() or ok_payload)
    return ToolResult(ok=False, payload=proc.stderr.strip() or "AppleScript failed")
