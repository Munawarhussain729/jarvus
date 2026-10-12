"""Reminders and calendar tool using macOS reminders."""
from __future__ import annotations

import subprocess
from typing import Any

from app.security.policy import RiskLevel
from app.security.sanitize import sanitize_tool_argument_string
from app.tools.base import Tool, ToolResult, ToolSpec


class ReminderTool(Tool):
    spec = ToolSpec(
        name="reminder",
        description="Add a reminder to macOS Reminders app.",
        parameters_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Reminder text"},
                "list": {"type": "string", "description": "Reminder list name (default: Reminders)"},
                "due": {"type": "string", "description": "Due date/time (e.g., 'tomorrow 3pm', 'in 1 hour')"},
            },
            "required": ["text"],
            "additionalProperties": False,
        },
        risk=RiskLevel.MEDIUM,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            text = sanitize_tool_argument_string(arguments["text"], field_name="text")
            list_name = sanitize_tool_argument_string(arguments.get("list", "Reminders"), field_name="list") if arguments.get("list") else "Reminders"
            due = sanitize_tool_argument_string(arguments.get("due", ""), field_name="due") if arguments.get("due") else ""
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))

        script = f'''
        tell application "Reminders"
            tell list "{list_name}"
                make new reminder with properties {{name:"{text}"'''
        if due:
            script += f', due date date "{due}"'
        script += '''}
            end tell
        end tell'''

        try:
            proc = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if proc.returncode == 0:
                return ToolResult(ok=True, payload=f"Reminder added: {text}")
            return ToolResult(ok=False, payload=proc.stderr.strip() or "Failed to add reminder")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))


class CalendarTool(Tool):
    spec = ToolSpec(
        name="calendar",
        description="Add an event to macOS Calendar.",
        parameters_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Event title"},
                "start": {"type": "string", "description": "Start time (e.g., 'tomorrow 2pm')"},
                "end": {"type": "string", "description": "End time (e.g., 'tomorrow 3pm')"},
                "calendar": {"type": "string", "description": "Calendar name"},
            },
            "required": ["title", "start"],
            "additionalProperties": False,
        },
        risk=RiskLevel.MEDIUM,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            title = sanitize_tool_argument_string(arguments["title"], field_name="title")
            start = sanitize_tool_argument_string(arguments["start"], field_name="start")
            end = sanitize_tool_argument_string(arguments.get("end", ""), field_name="end") if arguments.get("end") else ""
            calendar = sanitize_tool_argument_string(arguments.get("calendar", ""), field_name="calendar") if arguments.get("calendar") else ""
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))

        script = f'''
        tell application "Calendar"
            tell calendar "{calendar}"
                set newEvent to make new event with properties {{summary:"{title}", start date date "{start}"'''
        if end:
            script += f', end date date "{end}"'
        script += '''}
            end tell
        end tell'''

        try:
            proc = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if proc.returncode == 0:
                return ToolResult(ok=True, payload=f"Calendar event added: {title}")
            return ToolResult(ok=False, payload=proc.stderr.strip() or "Failed to add event")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
