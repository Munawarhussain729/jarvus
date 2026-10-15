"""Deterministic fast-path intents for common voice commands.

Matches only unambiguous, high-frequency patterns and returns a
(tool_name, arguments) pair routed through the normal ToolRouter
(confirmation + audit still apply). Everything else falls through to the LLM.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from app.tools.internet import _COIN_ALIASES, _STOCK_NAME_MAP
from app.tools.macos_extra import _TLD_ALLOWLIST
from app.tools.utilities import MATH_ALLOWED_NAMES, _resolve_unit


def _minutes(value: str, unit: str) -> float:
    v = float(value)
    if unit.startswith(("sec", "s")):
        return v / 60
    if unit.startswith(("hour", "hr", "h")):
        return v * 60
    return v


# ---- speech-to-math normalisation ------------------------------------------

_MATH_WORDS = {
    "multiplied by": "*", "divided by": "/", "to the power of": "**",
    "square root of": "sqrt", "plus": "+", "minus": "-", "times": "*",
    "over": "/", "mod": "%", "x": "*", "point": ".",
}


def _normalize_math(s: str) -> Optional[str]:
    """Convert spoken math ('what's 15 percent of 80') to an expression, else None."""
    s = s.strip().rstrip("?.!")
    s = re.sub(r"(\d+(?:\.\d+)?)\s*(?:%|percent(?:age)?)\s*of\s*", r"(\1/100)*", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*(?:%|percent(?:age)?)", r"(\1/100)", s)
    s = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", s)
    s = re.sub(r"square root of\s*([0-9.]+)", r"sqrt(\1)", s)
    s = re.sub(r"sqrt\s*([0-9.]+)", r"sqrt(\1)", s)
    for word in sorted(_MATH_WORDS, key=len, reverse=True):
        s = re.sub(rf"\b{re.escape(word)}\b", _MATH_WORDS[word], s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*squared\b", r"\1**2", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*cubed\b", r"\1**3", s)
    s = s.replace(" ", "").replace(",", "").replace("^", "**")
    if not re.fullmatch(r"[0-9+\-*/().%a-z]+", s) or not re.search(r"\d", s):
        return None
    for name in re.findall(r"[a-z]+", s):
        if name not in MATH_ALLOWED_NAMES:
            return None
    return s


# ---- currency / unit words --------------------------------------------------

_CURRENCY_WORDS = {
    "dollar": "USD", "dollars": "USD", "usd": "USD", "buck": "USD", "bucks": "USD",
    "euro": "EUR", "euros": "EUR", "eur": "EUR",
    "pound": "GBP", "pounds": "GBP", "gbp": "GBP", "quid": "GBP",
    "yen": "JPY", "jpy": "JPY", "yuan": "CNY", "cny": "CNY", "rmb": "CNY",
    "rupee": "INR", "rupees": "INR", "inr": "INR",
    "franc": "CHF", "francs": "CHF", "chf": "CHF",
    "won": "KRW", "krw": "KRW", "peso": "MXN", "pesos": "MXN", "mxn": "MXN",
    "cad": "CAD", "aud": "AUD", "nzd": "NZD", "brl": "BRL", "reais": "BRL",
    "zar": "ZAR", "rand": "ZAR", "sek": "SEK", "nok": "NOK", "dkk": "DKK",
    "pln": "PLN", "hkd": "HKD", "sgd": "SGD", "rub": "RUB", "ruble": "RUB", "rubles": "RUB",
}

_COIN_NAMES = frozenset(_COIN_ALIASES) | frozenset(_COIN_ALIASES.values())

_CONVERT_RE = re.compile(
    r"^(?:convert\s+|how much is\s+|how many\s+|what'?s\s+|whats\s+)?"
    r"(-?\d+(?:\.\d+)?)\s*([a-z°'\"/]+)\s+(?:to|into|in)\s+([a-z°'\"]+)\s*[?.!]?$")


def _currency_code(word: str) -> str:
    return _CURRENCY_WORDS.get(word, word.upper())


_COIN_RE = re.compile(
    r"\b(bitcoin|btc|ethereum|eth|dogecoin|doge|solana|sol|cardano|ada|ripple|xrp|"
    r"litecoin|ltc|binancecoin|bnb|polkadot|dot|avalanche|avax|tether|usdt|"
    r"crypto(?:currenc(?:y|ies))?)\b")

_PRICE_WORDS_RE = re.compile(
    r"\b(price|prices|worth|value|cost|much|rate|today|now|current|check|trading at)\b")

_FILE_SEARCH_RE = re.compile(
    r"^(?:find|locate|search(?:\s+for)?|look for|where'?s|where is|wheres)\s+"
    r"(?:my\s+)?(?:the\s+)?"
    r"(files?|documents?|docs?|pdfs?|photos?|pictures?|images?|spreadsheets?|folders?)\b\s*(.*)$")

_URL_INTENT_RE = re.compile(
    r"^(?:open|go to|goto|visit|browse(?:\s+to)?|load)\s+"
    r"(?:the\s+)?(?:website|site|page|url|link)?\s*"
    r"([a-z0-9-]+(?:\.[a-z0-9-]+)+[^\s]*)$")

_FILE_EXT_RE = re.compile(r"\.(?:txt|md|py|pdf|csv|json|docx?|xlsx?|pptx?|png|jpe?g|zip|mov|mp[34])$")

# Things 'what is X' shouldn't route to wikipedia.
_WIKI_STOPS = frozenset({
    "it", "this", "that", "you", "up", "wrong", "next", "today", "tomorrow",
    "the time", "the date", "the weather", "the news", "going on", "happening",
    "for dinner", "for lunch", "on tv", "on my calendar", "the plan",
})


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

    # "what day is <date>" is date math, not the clock - check before clock.
    m = re.match(r"^what day (?:of the week )?(?:is|was|will)\s+(?!it\b|today\b|tomorrow\b)(.+?)[?.!]?$", t)
    if m:
        return "date_calc", json.dumps({"action": "day_of_week", "date": m.group(1).strip()})

    if re.search(r"\b(?:what time|what(?:'s| is) the time|whats the time|current time|time is it|tell me the time)\b", t) or \
       re.search(r"\b(?:what day|what(?:'s| is) the date|whats the date|what date|day is it)\b", t):
        return "clock", "{}"

    m = re.match(r"^weather(?:\s+(?:in|for|at)\s+(.+?))?[?.!]?$", t) \
        or re.match(r"^(?:what(?:'s| is)|whats|how'?s) (?:the )?weather(?:\s+like)?(?:\s+(?:in|for|at)\s+(.+?))?[?.!]?$", t)
    if m:
        city = next((g for g in m.groups() if g), "")
        return "weather", json.dumps({"city": city.strip(" ?")})
    if re.match(r"^(?:will it|is it going to)\s+(?:rain|snow|be sunny|be cloudy|storm)", t):
        return "weather", "{}"

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

    # --- math / conversion (deterministic, so always fast-path) ---
    m = re.match(r"^(?:what'?s|whats|what is|calculate|compute|solve|evaluate|how much is)\s+(.+?)\s*[?.!]?$", t)
    if m:
        expr = _normalize_math(m.group(1))
        if expr:
            return "calculate", json.dumps({"expression": expr})

    m = re.match(r"^how many ([a-z°'\"]+)\s+(?:are )?(?:in|is in)\s+(?:a |an |one )?([a-z°'\"]+)\s*[?.!]?$", t)
    if m and _resolve_unit(m.group(1)) and _resolve_unit(m.group(2)):
        return "unit_convert", json.dumps({"value": 1, "from_unit": m.group(2), "to_unit": m.group(1)})

    m = _CONVERT_RE.match(t)
    if m:
        amount, frm, to = float(m.group(1)), m.group(2).strip(), m.group(3).strip()
        if frm in _COIN_NAMES or to in _COIN_NAMES:
            coin, other = (frm, to) if frm in _COIN_NAMES else (to, frm)
            return "crypto", json.dumps({"coins": coin, "vs_currency": _currency_code(other).lower()})
        if to in _CURRENCY_WORDS:
            return "currency", json.dumps({"amount": amount, "from_currency": _currency_code(frm),
                                           "to_currency": _currency_code(to)})
        if _resolve_unit(frm) or _resolve_unit(to):
            return "unit_convert", json.dumps({"value": amount, "from_unit": frm, "to_unit": to})

    # --- live facts ---
    m = re.search(r"\bdays? between\s+(.+?)\s+and\s+(.+?)[?.!]?$", t)
    if m:
        return "date_calc", json.dumps({"action": "days_between",
                                        "date": m.group(1).strip(), "date2": m.group(2).strip()})
    m = re.search(r"\b(?:how many days? (?:until|till|to)|days? (?:until|till|to))\s+(.+?)[?.!]?$", t)
    if m:
        return "date_calc", json.dumps({"action": "days_until", "date": m.group(1).strip()})
    m = re.match(r"^(\d+)\s*days?\s+from (?:now|today)\s*[?.!]?$", t) \
        or re.match(r"^(?:what'?s|whats|what will (?:the date )?be)\s+(\d+)\s*days?\s+from (?:now|today)", t)
    if m:
        return "date_calc", json.dumps({"action": "add_days", "date": "today", "days": int(m.group(1))})

    coins = _COIN_RE.findall(t)
    if coins and (_PRICE_WORDS_RE.search(t) or len(t.split()) <= 3):
        return "crypto", json.dumps({"coins": ",".join(dict.fromkeys(coins))})

    m = re.match(r"^(?:what'?s|whats|how'?s|check|get|show(?: me)?|the)?\s*(.+?)\s+(?:stock|share)\s*(?:price|value)?\s*[?.!]?$", t) \
        or re.match(r"^stock price (?:of|for)\s+(.+?)\s*[?.!]?$", t)
    if m:
        raw = re.sub(r"^(?:the|my)\s+", "", m.group(1).strip())
        symbol = _STOCK_NAME_MAP.get(raw, raw.upper())
        if raw not in ("the", "a", "an", "this", "market") \
                and re.fullmatch(r"[a-zA-Z0-9.^=-]{1,12}", symbol):
            return "stock", json.dumps({"symbol": raw})

    tm = re.search(r"\b(?:news|headlines)\s+(?:about|on|for)\s+(.+?)\s*[?.!]?$", t)
    if re.search(r"\bhacker news\b", t) or tm or \
       (re.search(r"\b(?:news|headlines)\b", t)
        and re.match(r"^(?:what'?s|whats|tell me|give me|show me|read|get|check|any|is|are|the|latest|top|today)", t)):
        args = {"topic": tm.group(1).strip()} if tm else {}
        return "news", json.dumps(args)

    if re.search(r"\bjokes?\b|\bmake me laugh\b|\bsomething funny\b", t):
        return "joke", "{}"

    m = re.search(r"\b(?:my|public|external|wan|local)\s+ip\b|\bip address\b|\bwi-?fi\b|"
                  r"\bwhat network\b|\bmy network\b|\bam i online\b|\binternet connection\b", t)
    if m:
        action = "public" if re.search(r"\b(?:public|external|wan)\b", t) else "all"
        return "network_info", json.dumps({"action": action})
    m = re.match(r"^(?:where is|locate|lookup|look up|trace|who owns)\s+(?:ip\s+|the ip\s+)?(\d{1,3}(?:\.\d{1,3}){3}|[a-z0-9-]+(?:\.[a-z0-9-]+)+)\s*[?.!]?$", t)
    if m:
        return "network_info", json.dumps({"action": "lookup", "target": m.group(1)})

    m = re.match(r"^(?:wiki(?:pedia)?\s+(?:for|about|on)\s+|wiki\s+|tell me about\s+|who (?:is|was|are|were)\s+|what (?:is|are|was|were)\s+(?:a|an|the)\s+)(.+?)\s*[?.!]?$", t) \
        or re.match(r"^what (?:is|are|was|were)\s+([a-z][a-z' -]{2,50})\s*[?.!]?$", t)
    if m:
        topic = m.group(1).strip()
        if not re.search(r"\b(?:my|your)\b", topic) and topic not in _WIKI_STOPS:
            return "wikipedia", json.dumps({"topic": topic})

    # --- files & macOS control ---
    m = _FILE_SEARCH_RE.match(t)
    if m:
        query = re.sub(r"^(?:named|called|titled|about|for|containing|matching|like)\s+", "",
                       m.group(2).strip())
        if query:
            args: dict[str, Any] = {"query": query.strip("\"'")}
            if re.search(r"\b(?:contain|content|inside|text)\w*", m.group(2)):
                args["content"] = True
            return "find_files", json.dumps(args)

    m = _URL_INTENT_RE.match(t)
    if m:
        host = m.group(1).split("/")[0].split("?")[0]
        if host.rsplit(".", 1)[-1].lower() in _TLD_ALLOWLIST:
            return "open_url", json.dumps({"url": m.group(1)})

    if re.search(r"\bdark\s*mode\b", t):
        if re.match(r"^is\b", t) or re.search(r"\bstatus\b", t):
            action = "get"
        elif re.search(r"\b(?:off|disable|deactivate|light)\b", t):
            action = "off"
        elif re.search(r"\b(?:on|enable|activate)\b", t):
            action = "on"
        else:
            action = "toggle"
        return "dark_mode", json.dumps({"action": action})

    if re.search(r"\block\s+(?:the\s+|my\s+)?(?:screen|mac|computer|laptop|display)\b|^lock(?:\s+(?:it|this|up))?$", t):
        return "lock_screen", "{}"
    if re.search(r"\b(?:sleep|turn off|shut off)\s+(?:the\s+|my\s+)?(?:display|screens?|monitors?)\b", t):
        return "sleep_display", "{}"
    if re.search(r"\bempty\s+(?:the\s+|my\s+)?(?:trash|bin|rubbish)\b", t):
        return "empty_trash", "{}"

    if re.search(r"\bflip\s+(?:a\s+)?coin\b|\bcoin flip\b|\bcoin toss\b|\btoss a coin\b", t):
        return "random", json.dumps({"action": "coin"})
    m = re.search(r"\broll\s+(?:a\s+|the\s+|me\s+)?(?:d(\d+)|die|dice)\b", t)
    if m:
        return "random", json.dumps({"action": "dice", "sides": int(m.group(1) or 6)})
    m = re.match(r"^(?:(?:give me|pick|generate|choose)\s+)?a?\s*random number(?:\s+between\s+(-?\d+)\s+and\s+(-?\d+))?", t)
    if m:
        args = {"action": "number"}
        if m.group(1):
            args["min"], args["max"] = int(m.group(1)), int(m.group(2))
        return "random", json.dumps(args)
    if re.search(r"\b(?:generate|create|make|give me)\s+(?:me\s+)?(?:a\s+)?(?:new\s+|secure\s+|random\s+|strong\s+)*password\b", t):
        return "random", json.dumps({"action": "password"})

    m = re.match(r"^(?:run|execute|start|launch)\s+(?:the\s+)?shortcut\s+(?:called\s+|named\s+)?(.+)$", t)
    if m:
        return "run_shortcut", json.dumps({"action": "run", "name": m.group(1).strip()})
    if re.match(r"^(?:list|show|what)\s+(?:my\s+|are my\s+)?shortcuts\b", t):
        return "run_shortcut", json.dumps({"action": "list"})

    m = re.match(r"^(?:search(?:\s+(?:the\s+)?(?:web|online|internet))?|google|look up)(?:\s+for)?\s+(.+)$", t)
    if m and not re.search(r"\b(?:files?|folders?|notes?|documents?)\b", m.group(1)):
        return "web_search", json.dumps({"query": m.group(1).strip(" ?")})

    # 'remind me in 20 minutes to call mom' -> timed reminder (tool parses due).
    m = re.match(r"^remind me in\s+(.+?)\s+(?:to|that)\s+(.+)$", t)
    if m:
        return "reminder", json.dumps({"text": m.group(2).strip(), "due": f"in {m.group(1).strip()}"})
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
    if m:
        target = m.group(1).strip()
        # 'open report.txt' is a file, 'open github.com' already matched open_url;
        # don't let either fall into open_app.
        if not re.search(r"\b(file|folder|note|scratchpad|notification|photo|shortcut)\b", target) \
                and not _FILE_EXT_RE.search(target):
            return "open_app", json.dumps({"app_name": target})

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
    (re.compile(r"\bfiles?\b|\bfolders?\b|\bdirector(?:y|ies)\b|\bdocuments?\b|\bfind\b|\blocate\b|\bspotlight\b|\.(?:txt|md|py|csv|json|docx)\b"),
     {"read_file", "write_file", "list_dir", "glob_files", "find_files", "create_docx"}),
    (re.compile(r"\b(?:terminal|shell|command|cmd|script|execut|python|code|process(?:es)?|kill|compile|build)\b"),
     {"run_command", "shell_read", "python_exec", "safe_shell", "ps_list"}),
    (re.compile(r"\b(?:app|application|open|launch|quit|close|front|active window|shortcut)\b"),
     {"open_app", "frontmost_app", "open_url", "run_shortcut"}),
    (re.compile(r"\bnotifications?\b|\bbanner\b"),
     {"send_notification", "read_notifications"}),
    (re.compile(r"\b(?:screenshots?|screen|clipboard|copy|paste)\b"),
     {"screenshot", "clipboard", "frontmost_app"}),
    (re.compile(r"\b(?:volume|mute|unmute|louder|quieter|sound)\b"),
     {"volume"}),
    (re.compile(r"\b(?:reminders?|remind|calendar|events?|meetings?|appointments?|schedule|agenda|"
                r"tasks?|to-?dos?|brief(?:ing)?|timer|alarm|wake me)\b"),
     {"reminder", "calendar", "list_reminders", "calendar_read", "timer", "alarm", "daily_brief", "date_calc"}),
    (re.compile(r"\b(?:music|songs?|tracks?|play|pause|skip|spotify|playlist|playing)\b"),
     {"music", "volume"}),
    (re.compile(r"\b(?:messages?|texts?|whatsapp|imessage|sms|email|e-mail|mail)\b"),
     {"message", "email"}),
    (re.compile(r"\b(?:search|google|look ?up|web|online|internet|fetch|url|website|browse|link|wiki(?:pedia)?)\b"),
     {"web_search", "web_fetch", "open_url", "wikipedia"}),
    (re.compile(r"\b(?:weather|forecast|temperature|rain|sunny|cloudy)\b"),
     {"weather"}),
    (re.compile(r"\b(?:system|battery|charging|charge|cpu|ram|disk|uptime|specs|memory usage)\b"),
     {"system_info", "ps_list", "battery"}),
    (re.compile(r"\b(?:word|docx|essay|report)\b"),
     {"create_docx", "write_file"}),
    (re.compile(r"\b(?:calculate|compute|math|arithmetic|percent|convert|conversion|units?|miles?|"
                r"kilometers?|pounds?|kilograms?|ounces?|celsius|fahrenheit|kelvin|liters?|litres?|"
                r"gallons?|meters?|metres?|feet|inches|days? (?:until|between|from)|day of the week|"
                r"coin|dice|die|random|password|uuid|pick|base64|encode|decode|hash|sha256|md5|"
                r"uppercase|lowercase|word count|reverse)\b"),
     {"calculate", "unit_convert", "date_calc", "random", "text_tools"}),
    (re.compile(r"\b(?:news|headlines|jokes?|funny|wikipedia|wiki|stocks?|shares?|ticker|crypto|"
                r"bitcoin|ethereum|dogecoin|currenc(?:y|ies)|exchange rate|dollars?|euros?|pounds?|"
                r"yen|rupees?|francs?|pesos?|won|price|quote)\b"),
     {"news", "joke", "wikipedia", "stock", "crypto", "currency"}),
    (re.compile(r"\b(?:ip|network|wi-?fi|ssid|gateway|online|connected|isp|vpn)\b"),
     {"network_info"}),
    (re.compile(r"\b(?:dark ?mode|light ?mode|appearance|lock|locked|sleep|display|monitor|"
                r"trash|rubbish|bin)\b"),
     {"dark_mode", "lock_screen", "sleep_display", "empty_trash"}),
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
