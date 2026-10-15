"""Deterministic fast-path intents for common voice commands.

Matches only unambiguous, high-frequency patterns and returns a
(tool_name, arguments) pair routed through the normal ToolRouter
(confirmation + audit still apply). Everything else falls through to the LLM.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional


def _minutes(value: str, unit: str) -> float:
    v = float(value)
    if unit.startswith(("sec", "s")):
        return v / 60
    if unit.startswith(("hour", "hr", "h")):
        return v * 60
    return v


def match(text: str) -> Optional[tuple[str, str]]:
    """Return (tool_name, arguments_json) for a confident match, else None."""
    t = re.sub(r"\s+", " ", text.strip().lower())
    t = re.sub(r"^(hey |ok |okay )?(jarvus|jarvis)[,. ]*", "", t).strip()

    m = re.match(r"^(?:set|start|create)(?: a| an| me)? timer for (\d+(?:\.\d+)?)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?)\b(?:\s*(?:for|label(?:led)?|called|to)\s*(.+))?", t) \
        or re.match(r"^(?:set|start|create)(?: a| an| me)? (\d+(?:\.\d+)?)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?)\s+timer\b(?:\s*(?:for|to|called|named)\s*(.+))?", t)
    if m:
        minutes = _minutes(m.group(1), m.group(2))
        args: dict[str, Any] = {"minutes": max(0.1, min(120, minutes))}
        if m.group(3):
            args["label"] = m.group(3).strip()
        return "timer", json.dumps(args)

    if re.search(r"\b(?:what time|what's the time|current time|time is it|tell me the time)\b", t) or \
       re.search(r"\b(?:what day|what's the date|what date|day is it)\b", t):
        return "clock", "{}"

    m = re.match(r"^weather(?:\s+(?:in|for|at)\s+(.+))?$|^what(?:'s| is) the weather(?:\s+(?:in|for)\s+(.+))?", t)
    if m and (m.group(1) or m.group(2)):
        return "weather", json.dumps({"city": (m.group(1) or m.group(2)).strip(" ?")})

    if re.search(r"\b(?:read|check|show|see)\s+(?:my\s+)?notifications\b", t):
        return "read_notifications", "{}"

    # Daily digest / task analysis - deterministic, no LLM round trip.
    if re.search(r"\b(?:brief|briefing|analy[sz]e|review|plan)\b.*\b(?:my\s+)?(?:day|tasks?|schedule)\b", t) or \
       re.search(r"\b(?:what'?s on|what do i have|my)\s+(?:today|schedule|agenda|day)\b", t) or \
       re.search(r"\bhow'?s my day\b", t):
        return "daily_brief", "{}"

    if re.search(r"\b(?:list|show|read|check|what are)\s+(?:my\s+)?reminders\b", t):
        return "list_reminders", "{}"

    if re.search(r"\b(?:what'?s on|show|list|read|check)\s+(?:my\s+)?calendar\b", t):
        return "calendar_read", "{}"

    if re.search(r"\b(?:show|list|what are|check)\s+(?:my\s+)?(?:to-?dos?|tasks?)\b|\btask list\b", t):
        return "todo_write", json.dumps({"action": "list"})

    if re.search(r"\b(?:battery|charge level|power left|charging|percentage left|how much (?:charge|power))\b", t):
        return "battery", "{}"

    if re.search(r"\b(?:take|snap|capture)\s+(?:a\s+)?screenshot\b|^screenshot\b", t):
        return "screenshot", "{}"

    if re.search(r"\bwhat(?:'s| is) (?:playing|now playing)\b", t):
        return "music", json.dumps({"action": "now_playing"})

    m = re.match(r"^(play|pause|resume|next|previous|skip)\b(?:\s+(?:the\s+)?(?:song|track|music))?", t)
    if m:
        action = {"play": "play", "resume": "play", "pause": "pause",
                  "next": "next", "skip": "next", "previous": "previous"}[m.group(1)]
        return "music", json.dumps({"action": action})

    m = re.match(r"^(?:search(?:\s+(?:the\s+)?(?:web|online|internet))?|google|look up)(?:\s+for)?\s+(.+)$", t)
    if m and not re.search(r"\b(?:files?|folders?|notes?|documents?)\b", m.group(1)):
        return "web_search", json.dumps({"query": m.group(1).strip(" ?")})

    m = re.match(r"^remind me (?:to|about|that)\s+(.+)$", t)
    if m:
        return "reminder", json.dumps({"text": m.group(1).strip()})

    m = re.match(r"^(?:set\s+)?volume(?:\s+to)?\s+(\d{1,3})\s*%?$", t)
    if m:
        return "volume", json.dumps({"action": "set", "level": int(m.group(1))})
    if re.match(r"^mute\b", t):
        return "volume", json.dumps({"action": "mute"})
    if re.match(r"^unmute\b", t):
        return "volume", json.dumps({"action": "unmute"})

    m = re.match(r"^(?:open|launch|start)\s+([a-z0-9 .&'+-]{2,40})$", t)
    if m and not re.search(r"\b(file|folder|note|scratchpad|notification|photo)\b", m.group(1)):
        return "open_app", json.dumps({"app_name": m.group(1).strip()})

    return None


# ---- Heuristic tool prefilter (JARVUS_TOOL_PREFILTER=off to disable) ----
# Shrinks the schema payload the small local model has to reason over.
# CONSERVATIVE BY DESIGN: a false negative (dropping the needed tool) is
# unrecoverable - the model can never select it. So: union of ALL matching
# groups, a broad always-on core, and callers expand to the full set once a
# tool round has run (a tool result may need a tool outside the subset).

_ALWAYS_TOOLS = frozenset({
    "remember", "recall", "forget", "scratchpad", "plan", "todo_write",
    "daily_brief", "clock", "echo", "say",
})

_TOOL_GROUPS: list[tuple[re.Pattern, frozenset[str]]] = [
    (re.compile(r"\bfiles?\b|\bfolders?\b|\bdirector(?:y|ies)\b|\bdocuments?\b|\.(?:txt|md|py|csv|json|docx)\b"),
     {"read_file", "write_file", "list_dir", "glob_files", "create_docx"}),
    (re.compile(r"\b(?:terminal|shell|command|cmd|script|execut|python|code|process(?:es)?|kill|compile|build)\b"),
     {"run_command", "shell_read", "python_exec", "safe_shell", "ps_list"}),
    (re.compile(r"\b(?:app|application|open|launch|quit|close|front|active window)\b"),
     {"open_app", "frontmost_app"}),
    (re.compile(r"\bnotifications?\b|\bbanner\b"),
     {"send_notification", "read_notifications"}),
    (re.compile(r"\b(?:screenshots?|screen|clipboard|copy|paste)\b"),
     {"screenshot", "clipboard", "frontmost_app"}),
    (re.compile(r"\b(?:volume|mute|unmute|louder|quieter|sound)\b"),
     {"volume"}),
    (re.compile(r"\b(?:reminders?|remind|calendar|events?|meetings?|appointments?|schedule|agenda|"
                r"tasks?|to-?dos?|brief(?:ing)?|timer|alarm|wake me)\b"),
     {"reminder", "calendar", "list_reminders", "calendar_read", "timer", "alarm", "daily_brief"}),
    (re.compile(r"\b(?:music|songs?|tracks?|play|pause|skip|spotify|playlist|playing)\b"),
     {"music", "volume"}),
    (re.compile(r"\b(?:messages?|texts?|whatsapp|imessage|sms|email|e-mail|mail)\b"),
     {"message", "email"}),
    (re.compile(r"\b(?:search|google|look ?up|web|online|internet|fetch|url|website|browse|link)\b"),
     {"web_search", "web_fetch"}),
    (re.compile(r"\b(?:weather|forecast|temperature|rain|sunny|cloudy)\b"),
     {"weather"}),
    (re.compile(r"\b(?:system|battery|charging|charge|cpu|ram|disk|uptime|specs|memory usage)\b"),
     {"system_info", "ps_list", "battery"}),
    (re.compile(r"\b(?:word|docx|essay|report)\b"),
     {"create_docx", "write_file"}),
]


def tool_subset_for(text: str) -> frozenset[str] | None:
    """Tool names the user text plausibly needs, or None for the full payload.

    Union of every matching group plus the always-on core - never first-match.
    """
    t = text.lower()
    names = set(_ALWAYS_TOOLS)
    matched = False
    for pattern, tools in _TOOL_GROUPS:
        if pattern.search(t):
            matched = True
            names.update(tools)
    return frozenset(names) if matched else None
