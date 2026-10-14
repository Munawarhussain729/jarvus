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
