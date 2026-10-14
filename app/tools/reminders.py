"""Reminders and calendar tool using macOS reminders."""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

from app.security.policy import RiskLevel
from app.security.sanitize import sanitize_tool_argument_string
from app.tools.applescript import escape, run
from app.tools.base import Tool, ToolResult, ToolSpec


_WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
             "friday": 4, "saturday": 5, "sunday": 6}


def _parse_due(raw: str) -> datetime | None:
    """Parse common natural date phrases. AppleScript `date "x"` coercion is
    locale-fragile, so we parse in Python and set components explicitly."""
    s = raw.strip().lower()
    s = re.sub(r"^(?:at|on|by)\s+", "", s)
    now = datetime.now()

    m = re.match(r"^(?:in\s+)?(\d+(?:\.\d+)?)\s*(secs?|seconds?|mins?|minutes?|hrs?|hours?|days?|weeks?)\s*(?:from\s+now)?$", s)
    if m:
        unit = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks"}[m.group(2)[0]]
        return now + timedelta(**{unit: float(m.group(1))})

    hour = minute = None
    if re.search(r"\bnoon\b", s):
        hour, minute = 12, 0
    elif re.search(r"\bmidnight\b", s):
        hour, minute = 0, 0
    else:
        tm = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b", s)
        if tm:
            hour = int(tm.group(1)) % 12 + (12 if tm.group(3).startswith("p") else 0)
            minute = int(tm.group(2) or 0)
        else:
            tm = re.search(r"\b(\d{1,2}):(\d{2})\b", s)
            if tm:
                hour, minute = int(tm.group(1)), int(tm.group(2))
    # Strip only unambiguous time tokens (colon or am/pm or noon/midnight) -
    # a bare 1-2 digit token could be part of an ISO/M-D date.
    day = re.sub(r"\b(?:at\s+)?(?:\d{1,2}:\d{2}|\d{1,2}\s*(?:a\.?m\.?|p\.?m\.?)|noon|midnight)\b", "", s).strip(" ,")

    def at(dt: datetime, default_h: int = 9) -> datetime:
        return dt.replace(hour=hour if hour is not None else default_h,
                          minute=minute if minute is not None else 0,
                          second=0, microsecond=0)

    if not s:
        return None
    if not day or day == "today":
        return at(now + timedelta(hours=1)) if hour is None else at(now)
    if day == "tomorrow":
        return at(now + timedelta(days=1))
    if day == "tonight":
        return at(now, default_h=20)
    m = re.match(r"^(?:this\s+|next\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$", day)
    if m:
        delta = (_WEEKDAYS[m.group(1)] - now.weekday()) % 7 or 7
        return at(now + timedelta(days=delta))

    for fmt, has_time in (("%Y-%m-%d %H:%M", True), ("%Y-%m-%d", False),
                          ("%m/%d/%Y", False), ("%m/%d/%y", False)):
        try:
            d = datetime.strptime(day, fmt)
        except ValueError:
            continue
        return d if has_time and hour is None else at(d)
    return None


def _as_date(var: str, d: datetime) -> str:
    """AppleScript that sets variable `var` to `d` via explicit components."""
    secs = d.hour * 3600 + d.minute * 60
    return f"""set {var} to (current date)
        set day of {var} to 1
        set month of {var} to {d.month}
        set year of {var} to {d.year}
        set day of {var} to {d.day}
        set time of {var} to {secs}"""


def _opt(arguments: dict[str, Any], key: str, default: str) -> str:
    raw = arguments.get(key)
    if not raw:
        return default
    return escape(sanitize_tool_argument_string(raw, field_name=key))


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
            text = escape(sanitize_tool_argument_string(arguments["text"], field_name="text"))
            list_name = _opt(arguments, "list", "Reminders")
            due = _opt(arguments, "due", "")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))

        # Parse the due phrase in Python (AppleScript's `date "x"` coercion is
        # locale-fragile and fails at compile time), then set explicit fields.
        due_block = ""
        note = ""
        if due:
            d = _parse_due(arguments.get("due", ""))
            if d:
                due_block = f'''{_as_date("dueD", d)}
            set due date of newRem to dueD
            set dueSet to true'''
            else:
                note = f" (could not parse due date '{arguments.get('due')}')"

        # Unknown lists must not fail the reminder: resolve named list, fall
        # back to default.
        script = f'''
        tell application "Reminders"
            set targetList to missing value
            try
                if "{list_name}" is not "" then set targetList to list "{list_name}"
            end try
            if targetList is missing value then set targetList to default list
            tell targetList
                set newRem to make new reminder with properties {{name:"{text}"}}
            end tell
            set dueSet to false
            {due_block}
        end tell
        return dueSet'''
        ok = run(script, ok_payload="")
        if not ok.ok:
            return ok
        return ToolResult(ok=True, payload=f"Reminder added: {arguments['text']}{note}")


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
            title = escape(sanitize_tool_argument_string(arguments["title"], field_name="title"))
            calendar = _opt(arguments, "calendar", "")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))

        start = _parse_due(arguments.get("start", ""))
        if not start:
            return ToolResult(ok=False, payload=f"could not parse start time '{arguments.get('start')}'")
        end = _parse_due(arguments.get("end", "")) if arguments.get("end") else None
        if not end or end <= start:
            end = start + timedelta(hours=1)

        script = f'''
        {_as_date("sd", start)}
        {_as_date("ed", end)}
        tell application "Calendar"
            set targetCal to missing value
            try
                if "{calendar}" is not "" then set targetCal to calendar "{calendar}"
            end try
            if targetCal is missing value then set targetCal to calendar 1
            tell targetCal
                make new event with properties {{summary:"{title}", start date:sd, end date:ed}}
            end tell
        end tell'''
        return run(script, ok_payload=f"Calendar event added: {arguments['title']}")
