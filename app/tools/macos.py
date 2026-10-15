"""macOS-specific system control tools."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

from app.security.policy import RiskLevel
from app.security.sanitize import sanitize_tool_argument_string
from app.tools.base import Tool, ToolResult, ToolSpec


_APP_DIRS = ("/Applications", "/System/Applications", str(Path.home() / "Applications"))

# App dir listing changes rarely; cache resolutions for 5 min to skip the
# triple os.scandir on every 'open X'.
_app_cache: dict[str, str | None] = {}
_app_cache_at = 0.0
_APP_CACHE_TTL = 300.0


def _resolve_app(name: str) -> str | None:
    """Find the real app name (e.g. 'whats app' -> 'WhatsApp') in app dirs."""
    global _app_cache_at
    base = name[:-4] if name.lower().endswith(".app") else name
    want = base.lower().replace(" ", "")
    now = time.monotonic()
    if now - _app_cache_at > _APP_CACHE_TTL:
        _app_cache.clear()
        _app_cache_at = now
    if want in _app_cache:
        return _app_cache[want]
    found = None
    for d in _APP_DIRS:
        try:
            for entry in os.scandir(d):
                if not entry.is_dir():
                    continue
                stem = entry.name[:-4] if entry.name.lower().endswith(".app") else entry.name
                if stem.lower().replace(" ", "") == want:
                    found = stem
                    break
        except OSError:
            continue
        if found:
            break
    _app_cache[want] = found
    return found


class OpenAppTool(Tool):
    spec = ToolSpec(
        name="open_app",
        description="Open an application on macOS (e.g., 'WhatsApp', 'Safari', 'Terminal').",
        parameters_schema={
            "type": "object",
            "properties": {
                "app_name": {"type": "string", "description": "App name like 'WhatsApp.app' or just 'WhatsApp'"},
                "url": {"type": "string", "description": "Optional URL to open with app"},
            },
            "additionalProperties": False,
        },
        # MEDIUM: launching a resolved named app is low blast-radius - no
        # confirmation, so voice 'open X' just works. HIGH stays for
        # run_command/python_exec/web_fetch.
        risk=RiskLevel.MEDIUM,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            app_name = sanitize_tool_argument_string(arguments.get("app_name", ""), field_name="app_name") if arguments.get("app_name") else None
            url = sanitize_tool_argument_string(arguments.get("url", ""), field_name="url") if arguments.get("url") else None
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        if not app_name and not url:
            return ToolResult(ok=False, payload="need app_name or url")

        if app_name:
            # Voice transcripts arrive as 'whatsapp.' - strip punctuation and
            # resolve 'whats app'/'Word' against the real app dirs.
            cleaned = app_name.strip().strip(".,!?:;").strip()
            app_name = _resolve_app(cleaned) or cleaned

        args = ["open"]
        if app_name:
            if "/" not in app_name and not app_name.endswith(".app"):
                args.extend(["-a", app_name])
            else:
                args.append(app_name)
        if url:
            args.append(url)
        try:
            proc = subprocess.run(args, capture_output=True, text=True, timeout=30)
            out = (proc.stdout or "") + (proc.stderr or "")
            if proc.returncode != 0:
                out = out.strip() or "open failed"
                out += f" (tried '{app_name}'; check /Applications for the exact name)"
            return ToolResult(ok=proc.returncode == 0, payload=out or "opened", detail={"code": proc.returncode})
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))


class SayTool(Tool):
    spec = ToolSpec(
        name="say",
        description="Speak text using macOS 'say' command.",
        parameters_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "maxLength": 500},
            },
            "required": ["text"],
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            text = sanitize_tool_argument_string(arguments["text"], field_name="text")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        try:
            subprocess.run(["say", text], timeout=60, capture_output=True)
            return ToolResult(ok=True, payload=f"said: {text}")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
