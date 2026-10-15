"""Timer and alarm tool using macOS."""
from __future__ import annotations

import subprocess
import threading
import time
from datetime import datetime, timedelta
from typing import Any

from app.security.policy import RiskLevel
from app.security.sanitize import sanitize_tool_argument_string
from app.tools.base import Tool, ToolResult, ToolSpec


class TimerTool(Tool):
    spec = ToolSpec(
        name="timer",
        description="Set a timer for N minutes. Speaks an alert when done.",
        parameters_schema={
            "type": "object",
            "properties": {
                "minutes": {"type": "number", "minimum": 0.1, "maximum": 120, "description": "Timer duration in minutes"},
                "label": {"type": "string", "description": "Optional label for the timer"},
            },
            "required": ["minutes"],
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            minutes = float(arguments["minutes"])
            label = sanitize_tool_argument_string(arguments.get("label", ""), field_name="label") if arguments.get("label") else ""
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))

        def _fire():
            time.sleep(minutes * 60)
            msg = f"Timer done: {label}" if label else "Timer done"
            subprocess.run(["say", msg], capture_output=True, timeout=30)

        t = threading.Thread(target=_fire, daemon=True)
        t.start()
        display = f"{minutes} minute{'s' if minutes != 1 else ''}"
        return ToolResult(ok=True, payload=f"Timer set for {display}" + (f" ({label})" if label else ""))


class AlarmTool(Tool):
    spec = ToolSpec(
        name="alarm",
        description="Set an alarm for a specific time (24h format, e.g. '07:30').",
        parameters_schema={
            "type": "object",
            "properties": {
                "time": {"type": "string", "description": "Alarm time in HH:MM 24h format"},
                "label": {"type": "string", "description": "Optional label"},
            },
            "required": ["time"],
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            time_str = sanitize_tool_argument_string(arguments["time"], field_name="time")
            label = sanitize_tool_argument_string(arguments.get("label", ""), field_name="label") if arguments.get("label") else ""
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))

        parts = time_str.split(":")
        if len(parts) != 2:
            return ToolResult(ok=False, payload="Time must be in HH:MM format")
        try:
            hour, minute = int(parts[0]), int(parts[1])
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError
        except ValueError:
            return ToolResult(ok=False, payload="Invalid time. Use HH:MM (00:00-23:59)")

        # Sleep straight to the target instead of polling every 20s forever.
        now = datetime.now()
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        delay = (target - now).total_seconds()

        def _fire():
            time.sleep(delay)
            msg = f"Alarm: {label}" if label else "Alarm"
            subprocess.run(["say", msg], capture_output=True, timeout=30)

        t = threading.Thread(target=_fire, daemon=True)
        t.start()
        return ToolResult(ok=True, payload=f"Alarm set for {time_str:0>5}" + (f" ({label})" if label else ""))
