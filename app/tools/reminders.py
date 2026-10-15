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


def day_bounds(day: str = "today", days: int = 1) -> tuple[datetime, datetime] | None:
    """Local (start, end) datetimes covering `days` days starting on `day`
    ('today', 'tomorrow', 'week', weekday names, ISO/US dates)."""
    day = (day or "today").strip().lower()
    if day in ("week", "this week"):
        day, days = "today", max(days, 7)
    if day == "today":
        # _parse_due("today") means 'an hour from now' (reminder semantics) -
        # which rolls into tomorrow after 23:00. Day bounds always mean
        # calendar-today.
        d = datetime.now()
    else:
        d = _parse_due(day)
        if d is None:
            return None
    start = d.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=max(1, int(days)))


def _fmt_date(var: str) -> str:
    """AppleScript expr rendering date var as 'YYYY-MM-DD H:MM' (parseable by strptime)."""
    return (f'((year of {var} as integer) & "-" & (month of {var} as integer) & "-" '
            f'& (day of {var}) & " " & ((time of {var}) div 3600) & ":" '
            f'& (((time of {var}) mod 3600) div 60))')


_PERMISSION_MSG = (
    "macOS denied access - grant the terminal app Reminders/Calendar "
    "permission in System Settings > Privacy & Security, then retry."
)


def _check_perm(res: ToolResult) -> str | None:
    """Return a friendly message when osascript reported an Automation denial."""
    if res.ok:
        return None
    if "1743" in res.payload or "not authorized" in res.payload.lower():
        return _PERMISSION_MSG
    return res.payload


def _parse_row_dt(raw: str) -> datetime | None:
    try:
        return datetime.strptime(raw.strip(), "%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return None


def fetch_open_reminders(limit: int = 200) -> tuple[list[dict[str, Any]], str]:
    """(rows, error). rows: {'text','list','due': datetime|None}. Shared by the
    list_reminders tool and the daily digest - mock this in tests."""
    script = f'''
    tell application "Reminders"
        set out to ""
        repeat with l in lists
            set ln to name of l
            repeat with r in (get reminders of l whose completed is false)
                set nm to ""
                try
                    set nm to name of r
                end try
                if nm is not "" then
                    set dd to ""
                    try
                        set d to due date of r
                        set dd to {_fmt_date("d")}
                    end try
                    set out to out & nm & "|" & ln & "|" & dd & linefeed
                end if
            end repeat
        end repeat
        return out
    end tell'''
    res = run(script, timeout=30)
    err = _check_perm(res)
    if err:
        return [], err
    rows = []
    for line in res.payload.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3 and parts[0]:
            rows.append({"text": parts[0], "list": parts[1], "due": _parse_row_dt(parts[2])})
        if len(rows) >= limit:
            break
    return rows, ""


def fetch_calendar_events(start: datetime, end: datetime, limit: int = 50) -> tuple[list[dict[str, Any]], str]:
    """(rows, error). rows: {'title','calendar','start','end','all_day'}."""
    script = f'''
    {_as_date("sd", start)}
    {_as_date("ed", end)}
    tell application "Calendar"
        set out to ""
        repeat with c in calendars
            set cn to name of c
            repeat with e in (get every event of c whose start date ≥ sd and start date < ed)
                try
                    set sm to summary of e
                    set es to start date of e
                    set ee to end date of e
                    set allday to allday event of e
                    if sm is not "" then
                        set out to out & sm & "|" & cn & "|" & {_fmt_date("es")} & "|" & {_fmt_date("ee")} & "|" & allday & linefeed
                    end if
                end try
            end repeat
        end repeat
        return out
    end tell'''
    res = run(script, timeout=45)
    err = _check_perm(res)
    if err:
        return [], err
    rows = []
    for line in res.payload.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 4 and parts[0]:
            rows.append({
                "title": parts[0], "calendar": parts[1],
                "start": _parse_row_dt(parts[2]), "end": _parse_row_dt(parts[3]),
                "all_day": len(parts) > 4 and parts[4].lower() == "true",
            })
        if len(rows) >= limit:
            break
    return rows, ""


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


def _fmt_due(d: datetime | None) -> str:
    if d is None:
        return ""
    today = datetime.now().date()
    if d.date() == today:
        return f"today {d.strftime('%H:%M')}"
    if d.date() < today:
        return f"OVERDUE {d.strftime('%Y-%m-%d %H:%M')}"
    return d.strftime("%Y-%m-%d %H:%M")


class ListRemindersTool(Tool):
    spec = ToolSpec(
        name="list_reminders",
        description="List incomplete macOS Reminders, with due times. Filter: all, today, or overdue.",
        parameters_schema={
            "type": "object",
            "properties": {
                "when": {"type": "string", "enum": ["all", "today", "overdue"], "default": "all"},
                "list": {"type": "string", "description": "Reminder list name filter (optional)"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
            },
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        when = str(arguments.get("when") or "all")
        list_filter = str(arguments.get("list") or "").strip().lower()
        limit = min(50, max(1, int(arguments.get("limit") or 20)))
        rows, err = fetch_open_reminders()
        if err:
            return ToolResult(ok=False, payload=err)
        today = datetime.now().date()
        if when == "today":
            rows = [r for r in rows if r["due"] and r["due"].date() <= today]
        elif when == "overdue":
            rows = [r for r in rows if r["due"] and r["due"].date() < today]
        if list_filter:
            rows = [r for r in rows if list_filter in r["list"].lower()]
        rows.sort(key=lambda r: (r["due"] is None, r["due"] or datetime.max))
        rows = rows[:limit]
        if not rows:
            return ToolResult(ok=True, payload=f"No {when} reminders found.")
        lines = [f"- {r['text']}" + (f" [{_fmt_due(r['due'])}]" if r["due"] else "")
                 + (f" ({r['list']})" if r["list"] else "") for r in rows]
        return ToolResult(ok=True, payload="\n".join(lines))


class CalendarReadTool(Tool):
    spec = ToolSpec(
        name="calendar_read",
        description="List macOS Calendar events for a day/range (today, tomorrow, week, a date).",
        parameters_schema={
            "type": "object",
            "properties": {
                "day": {"type": "string", "description": "today | tomorrow | week | a date like '2026-10-20'"},
                "days": {"type": "integer", "minimum": 1, "maximum": 14, "default": 1},
            },
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        day = str(arguments.get("day") or "today")
        days = int(arguments.get("days") or 1)
        bounds = day_bounds(day, days)
        if not bounds:
            return ToolResult(ok=False, payload=f"could not parse day '{day}'")
        rows, err = fetch_calendar_events(*bounds)
        if err:
            return ToolResult(ok=False, payload=err)
        if not rows:
            return ToolResult(ok=True, payload=f"No events for {day}.")
        lines = []
        for r in rows:
            when = "all day" if r["all_day"] or not r["start"] else r["start"].strftime("%H:%M")
            lines.append(f"- [{when}] {r['title']}" + (f" ({r['calendar']})" if r["calendar"] else ""))
        return ToolResult(ok=True, payload="\n".join(lines))
