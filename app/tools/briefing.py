"""Daily briefing - a deterministic digest of today's tasks.

gather_day_digest() does NO LLM call: it merges today's calendar events,
due/overdue reminders, persisted todos and recent notes into one compact,
spoken-friendly summary. `read_fns` is injectable for tests.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from app.security.policy import RiskLevel
from app.tools.base import Tool, ToolResult, ToolSpec


def _fetch_weather() -> str | None:
    """Short weather line for the morning brief; None when unreachable."""
    try:
        from app.tools.weather import WeatherTool
        res = WeatherTool().run({"city": "", "format": "short"})
        return res.payload if res.ok and res.payload else None
    except Exception:
        return None


def _default_reads() -> dict[str, Callable]:
    from app.tools.reminders import fetch_calendar_events, fetch_open_reminders, day_bounds

    return {"reminders": fetch_open_reminders, "events": fetch_calendar_events,
            "bounds": day_bounds, "weather": _fetch_weather}


def _recent_notes(n: int = 3) -> list[str]:
    import os
    path = Path(os.environ.get("JARVUS_DATA_DIR", "~/.jarvus")).expanduser() / "notes.md"
    try:
        lines = [l.strip() for l in path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
        return lines[-n:]
    except OSError:
        return []


def gather_day_digest(memory=None, read_fns: dict[str, Callable] | None = None) -> str:
    """Compact spoken-friendly digest of today. `read_fns` provides
    {'reminders', 'events', 'bounds'} callables (default: macOS reads)."""
    fns = read_fns or _default_reads()
    today = datetime.now().date()
    errors: list[str] = []

    events: list[dict[str, Any]] = []
    bounds = fns["bounds"]("today")
    if bounds:
        events, err = fns["events"](*bounds)
        if err:
            errors.append(err)

    reminders, err = fns["reminders"]()
    if err:
        errors.append(err)
    due = [r for r in reminders if r["due"] and r["due"].date() <= today]
    due.sort(key=lambda r: r["due"])
    overdue = [r for r in due if r["due"].date() < today]
    undated = [r for r in reminders if not r["due"]]

    todos = memory.list_todos() if memory is not None else []

    parts: list[str] = []
    weather_fn = fns.get("weather")
    if weather_fn:
        try:
            weather = weather_fn()
        except Exception:
            weather = None
        if weather:
            parts.append(weather.rstrip(".") if len(weather) < 160 else weather[:160])
    if events:
        events.sort(key=lambda e: (e["start"] is None, e["start"] or datetime.max))
        first = events[0]
        t = "all day" if first["all_day"] or not first["start"] else first["start"].strftime("%H:%M")
        parts.append(f"{len(events)} event(s) today, first at {t}: {first['title']}")
    if due:
        top = due[0]
        label = f", {len(overdue)} overdue" if overdue else ""
        parts.append(f"{len(due)} reminder(s) due{label} - most urgent: {top['text']} at {top['due'].strftime('%H:%M')}")
    if todos:
        high = [t for t in todos if t["priority"] == "high"]
        parts.append(f"{len(todos)} open todo(s)" + (f", {len(high)} high priority" if high else ""))
    if undated:
        parts.append(f"{len(undated)} undated reminder(s) waiting")
    notes = _recent_notes()
    if notes:
        parts.append("latest note: " + notes[-1].split("] ", 1)[-1][:80])

    if not parts and not errors:
        return "Nothing on the schedule today - calendar, reminders and todo list are all clear."
    digest = ". ".join(parts) + "." if parts else ""
    if errors:
        hint = " ".join(dict.fromkeys(errors))
        digest = (digest + " " if digest else "") + f"({hint})"
    return digest.strip() or "Nothing scheduled today."


def maybe_auto_brief(memory, enabled: bool, read_fns: dict[str, Callable] | None = None) -> str | None:
    """Return today's digest once per local day, else None.

    Uses a LOCAL date (not UTC) - the brief window is the user's day.
    Never raises: read errors surface inside the digest itself.
    """
    if not enabled or memory is None or not hasattr(memory, "get_meta"):
        return None
    today = datetime.now().date().isoformat()
    if memory.get_meta("last_brief_date") == today:
        return None
    try:
        digest = gather_day_digest(memory, read_fns)
    except Exception:
        return None
    memory.set_meta("last_brief_date", today)
    return digest


class DailyBriefTool(Tool):
    spec = ToolSpec(
        name="daily_brief",
        description="Summarize today: calendar events, due/overdue reminders, open todos, latest note.",
        parameters_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def __init__(self, memory=None) -> None:
        self._memory = memory

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            return ToolResult(ok=True, payload=gather_day_digest(self._memory))
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
