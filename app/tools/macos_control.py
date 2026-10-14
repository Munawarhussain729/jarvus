"""Extended macOS control tools - volume, notifications, screenshot, clipboard, active app."""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import Any

from app.security.policy import RiskLevel
from app.security.sanitize import sanitize_tool_argument_string
from app.tools.applescript import escape, run
from app.tools.base import Tool, ToolResult, ToolSpec


class VolumeTool(Tool):
    spec = ToolSpec(
        name="volume",
        description="Get or set macOS output volume (0-100), or mute/unmute.",
        parameters_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["get", "set", "mute", "unmute"], "default": "get"},
                "level": {"type": "integer", "minimum": 0, "maximum": 100},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        action = arguments.get("action", "get")
        if action == "get":
            return run('output volume of (get volume settings)')
        if action == "set":
            try:
                level = max(0, min(100, int(arguments.get("level", 50))))
            except (TypeError, ValueError):
                return ToolResult(ok=False, payload="level must be 0-100")
            return run(f"set volume output volume {level}", ok_payload=f"Volume set to {level}")
        muted = "true" if action == "mute" else "false"
        return run(f"set volume output muted {muted}", ok_payload=f"{'Muted' if action == 'mute' else 'Unmuted'}")


class NotifyTool(Tool):
    spec = ToolSpec(
        name="send_notification",
        description="Show a macOS notification banner (sends, does not read).",
        parameters_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "title": {"type": "string", "default": "Jarvus"},
            },
            "required": ["text"],
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            text = escape(sanitize_tool_argument_string(arguments["text"], field_name="text"))
            title = escape(sanitize_tool_argument_string(str(arguments.get("title") or "Jarvus"), field_name="title"))
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        return run(f'display notification "{text}" with title "{title}"', ok_payload="Notification shown")


class ScreenshotTool(Tool):
    spec = ToolSpec(
        name="screenshot",
        description="Take a screenshot, saved under ~/.jarvus/screenshots/.",
        parameters_schema={
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Optional filename (default: timestamped)"},
            },
            "additionalProperties": False,
        },
        risk=RiskLevel.MEDIUM,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        name = arguments.get("filename") or f"shot_{int(time.time())}.png"
        try:
            name = sanitize_tool_argument_string(str(name), field_name="filename")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        if "/" in name or ".." in name:
            return ToolResult(ok=False, payload="filename must not contain path separators")
        if not name.endswith(".png"):
            name += ".png"
        out_dir = Path.home() / ".jarvus" / "screenshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / name
        try:
            proc = subprocess.run(["screencapture", "-x", str(path)], capture_output=True, text=True, timeout=15)
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        if proc.returncode == 0 and path.exists():
            return ToolResult(ok=True, payload=f"Screenshot saved: {path}")
        return ToolResult(ok=False, payload=proc.stderr.strip() or "screencapture failed")


class ClipboardTool(Tool):
    spec = ToolSpec(
        name="clipboard",
        description="Read or write the macOS clipboard.",
        parameters_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["read", "write"], "default": "read"},
                "text": {"type": "string", "description": "Text to copy (write only)"},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        action = arguments.get("action", "read")
        if action == "write":
            try:
                text = sanitize_tool_argument_string(str(arguments.get("text") or ""), field_name="text")
            except Exception as e:
                return ToolResult(ok=False, payload=str(e))
            proc = subprocess.run(["pbcopy"], input=text, capture_output=True, text=True, timeout=10)
            return ToolResult(ok=proc.returncode == 0, payload="Copied to clipboard" if proc.returncode == 0 else proc.stderr)
        try:
            proc = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=10)
            return ToolResult(ok=proc.returncode == 0, payload=proc.stdout[:8000] or "(clipboard empty)")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))


class BatteryTool(Tool):
    spec = ToolSpec(
        name="battery",
        description="Get battery level, charge status, and time remaining.",
        parameters_schema={"type": "object", "properties": {}, "additionalProperties": False},
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            proc = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10)
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        if proc.returncode != 0:
            return ToolResult(ok=False, payload=proc.stderr.strip() or "pmset failed")
        out = proc.stdout.strip()
        m = re.search(r"(\d+)%", out)
        if m:
            charging = "discharging" if "discharging" in out.lower() else ("charging" if "charging" in out.lower() or "attached" in out.lower() else "unknown")
            eta = ""
            em = re.search(r";\s*([^;]+)\s*remaining", out)
            if em:
                eta = f", {em.group(1).strip()} remaining"
            return ToolResult(ok=True, payload=f"Battery {m.group(1)}% ({charging}{eta})")
        return ToolResult(ok=True, payload=out or "No battery information")


class FrontAppTool(Tool):
    spec = ToolSpec(
        name="frontmost_app",
        description="Get the name of the currently active macOS application.",
        parameters_schema={"type": "object", "properties": {}, "additionalProperties": False},
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        return run(
            'tell application "System Events" to get name of first application process whose frontmost is true'
        )
