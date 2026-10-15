"""Deterministic utility tools - math, units, dates, randomness, text transforms.

All local and instant; no network, no model needed. Preferring these over LLM
mental math is both faster and more accurate.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import math
import operator
import re
import secrets
import urllib.parse
import uuid
from datetime import date, datetime, timedelta
from typing import Any

from app.security.policy import RiskLevel
from app.security.sanitize import sanitize_tool_argument_string
from app.tools.base import Tool, ToolResult, ToolSpec


# ---- calculate -------------------------------------------------------------

_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}

MATH_FUNCS = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "log": math.log, "ln": math.log, "log10": math.log10, "log2": math.log2,
    "exp": math.exp, "floor": math.floor, "ceil": math.ceil,
    "factorial": math.factorial, "gcd": math.gcd,
    "degrees": math.degrees, "radians": math.radians,
}
MATH_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
MATH_ALLOWED_NAMES = frozenset(MATH_FUNCS) | frozenset(MATH_CONSTS)
_MAX_NODES = 200
_MAX_POW_EXPONENT = 10_000


def _eval_node(node: ast.AST, depth: int = 0) -> float:
    if depth > 50:
        raise ValueError("expression too deep")
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, depth + 1)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError("only numbers allowed")
    if isinstance(node, ast.Name):
        if node.id in MATH_CONSTS:
            return MATH_CONSTS[node.id]
        raise ValueError(f"unknown name: {node.id}")
    if isinstance(node, ast.BinOp):
        fn = _BINOPS.get(type(node.op))
        if fn is None:
            raise ValueError("unsupported operator")
        left, right = _eval_node(node.left, depth + 1), _eval_node(node.right, depth + 1)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW_EXPONENT:
            raise ValueError("exponent too large")
        return fn(left, right)
    if isinstance(node, ast.UnaryOp):
        fn = _UNARY.get(type(node.op))
        if fn is None:
            raise ValueError("unsupported unary operator")
        return fn(_eval_node(node.operand, depth + 1))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in MATH_FUNCS:
            raise ValueError("unsupported function")
        if node.keywords:
            raise ValueError("keyword args not supported")
        args = [_eval_node(a, depth + 1) for a in node.args]
        return MATH_FUNCS[node.func.id](*args)
    raise ValueError(f"unsupported expression: {type(node).__name__}")


def _format_number(v: float) -> str:
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        return str(int(v))
    return f"{v:,g}"


class CalcTool(Tool):
    spec = ToolSpec(
        name="calculate",
        description="Evaluate a math expression accurately: + - * / // % ** (), functions "
                    "(sqrt, sin, cos, log, ln, exp, floor, ceil, factorial, gcd, abs, round, "
                    "min, max, sum) and constants pi, e, tau. ALWAYS use this for arithmetic.",
        parameters_schema={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "e.g. '2+2*3', 'sqrt(16)', '15/100*80'"},
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            expr = sanitize_tool_argument_string(str(arguments["expression"]), field_name="expression").strip()
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        if not expr:
            return ToolResult(ok=False, payload="empty expression")
        try:
            tree = ast.parse(expr, mode="eval")
            if len(list(ast.walk(tree))) > _MAX_NODES:
                return ToolResult(ok=False, payload="expression too complex")
            result = _eval_node(tree)
        except (ValueError, SyntaxError) as e:
            return ToolResult(ok=False, payload=f"cannot evaluate: {e}")
        except ZeroDivisionError:
            return ToolResult(ok=False, payload="division by zero")
        except (OverflowError, ArithmeticError) as e:
            return ToolResult(ok=False, payload=f"math error: {e}")
        if isinstance(result, complex) or (isinstance(result, float) and math.isnan(result)):
            return ToolResult(ok=False, payload="result is not a real number")
        return ToolResult(ok=True, payload=f"{expr} = {_format_number(result)}")


# ---- unit conversion -------------------------------------------------------

# kind -> {unit: factor to base} ; base = meter, gram, liter, byte, m/s, second
_UNIT_TABLES: dict[str, dict[str, float]] = {
    "length": {"mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0,
               "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344},
    "weight": {"mg": 0.001, "g": 1.0, "kg": 1000.0, "t": 1e6,
               "oz": 28.349523125, "lb": 453.59237, "st": 6350.29318},
    "volume": {"ml": 0.001, "l": 1.0, "tsp": 0.00492892, "tbsp": 0.0147868,
               "floz": 0.0295735, "cup": 0.236588, "pt": 0.473176,
               "qt": 0.946353, "gal": 3.78541},
    "data": {"b": 1.0, "kb": 1024.0, "mb": 1024**2, "gb": 1024**3, "tb": 1024**4, "pb": 1024**5},
    "speed": {"mps": 1.0, "kmh": 1 / 3.6, "mph": 0.44704, "kn": 0.514444},
    "time": {"ms": 0.001, "s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0, "wk": 604800.0},
}

_UNIT_ALIASES = {
    "millimeter": "mm", "millimeters": "mm", "millimetre": "mm", "millimetres": "mm",
    "centimeter": "cm", "centimeters": "cm", "centimetre": "cm", "centimetres": "cm",
    "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    "kilometer": "km", "kilometers": "km", "kilometre": "km", "kilometres": "km",
    "inch": "in", "inches": "in", '"': "in",
    "foot": "ft", "feet": "ft", "'": "ft",
    "yard": "yd", "yards": "yd",
    "mile": "mi", "miles": "mi",
    "milligram": "mg", "milligrams": "mg",
    "gram": "g", "grams": "g", "gramme": "g", "grammes": "g",
    "kilogram": "kg", "kilograms": "kg", "kilo": "kg", "kilos": "kg",
    "ton": "t", "tons": "t", "tonne": "t", "tonnes": "t",
    "ounce": "oz", "ounces": "oz",
    "pound": "lb", "pounds": "lb", "lbs": "lb",
    "stone": "st", "stones": "st",
    "milliliter": "ml", "milliliters": "ml", "millilitre": "ml", "millilitres": "ml",
    "liter": "l", "liters": "l", "litre": "l", "litres": "l",
    "teaspoon": "tsp", "teaspoons": "tsp", "tablespoon": "tbsp", "tablespoons": "tbsp",
    "fluid ounce": "floz", "fluid ounces": "floz",
    "cup": "cup", "cups": "cup", "pint": "pt", "pints": "pt",
    "quart": "qt", "quarts": "qt", "gallon": "gal", "gallons": "gal",
    "fahrenheit": "f", "celsius": "c", "centigrade": "c", "kelvin": "k",
    "byte": "b", "bytes": "b", "kilobyte": "kb", "kilobytes": "kb",
    "megabyte": "mb", "megabytes": "mb", "gigabyte": "gb", "gigabytes": "gb",
    "terabyte": "tb", "terabytes": "tb", "petabyte": "pb", "petabytes": "pb",
    "kilometers per hour": "kmh", "kilometres per hour": "kmh", "kph": "kmh",
    "miles per hour": "mph", "meters per second": "mps", "metres per second": "mps",
    "m/s": "mps", "knot": "kn", "knots": "kn",
    "millisecond": "ms", "milliseconds": "ms",
    "second": "s", "seconds": "s", "sec": "s", "secs": "s",
    "minute": "min", "minutes": "min", "mins": "min",
    "hour": "h", "hours": "h", "hr": "h", "hrs": "h",
    "day": "d", "days": "d", "week": "wk", "weeks": "wk",
    "°c": "c", "°f": "f",
}

# kind-ambiguous units resolved by the OTHER unit's kind (e.g. lb<->kg).
def _resolve_unit(raw: str) -> tuple[str, str] | None:
    """Return (kind, unit) for a user-typed unit name, or None."""
    u = raw.strip().lower().rstrip(".")
    u = _UNIT_ALIASES.get(u, u)
    for kind, table in _UNIT_TABLES.items():
        if u in table:
            return kind, u
    if u in ("c", "f", "k"):
        return "temperature", u
    return None


def _temp_to_c(v: float, u: str) -> float:
    if u == "c":
        return v
    if u == "f":
        return (v - 32) * 5 / 9
    return v - 273.15


def _temp_from_c(v: float, u: str) -> float:
    if u == "c":
        return v
    if u == "f":
        return v * 9 / 5 + 32
    return v + 273.15


class UnitConvertTool(Tool):
    spec = ToolSpec(
        name="unit_convert",
        description="Convert between units: length, weight, volume, temperature (c/f/k), "
                    "data size, speed, time. Deterministic - always prefer this over guessing.",
        parameters_schema={
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "from_unit": {"type": "string", "description": "e.g. 'km', 'miles', 'lb', 'fahrenheit'"},
                "to_unit": {"type": "string"},
            },
            "required": ["value", "from_unit", "to_unit"],
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            value = float(arguments["value"])
            frm = _resolve_unit(str(arguments["from_unit"]))
            to = _resolve_unit(str(arguments["to_unit"]))
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        if frm is None or to is None:
            bad = "from_unit" if frm is None else "to_unit"
            return ToolResult(ok=False, payload=f"unknown unit: {arguments.get(bad)}")
        kind_f, unit_f = frm
        kind_t, unit_t = to
        if kind_f != kind_t:
            return ToolResult(ok=False, payload=f"cannot convert {unit_f} ({kind_f}) to {unit_t} ({kind_t})")
        if kind_f == "temperature":
            result = _temp_from_c(_temp_to_c(value, unit_f), unit_t)
        else:
            result = value * _UNIT_TABLES[kind_f][unit_f] / _UNIT_TABLES[kind_f][unit_t]
        return ToolResult(ok=True, payload=f"{_format_number(value)} {unit_f} = {_format_number(round(result, 6))} {unit_t}")


# ---- date math -------------------------------------------------------------

_MONTHS = "january february march april may june july august september october november december".split()


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    return d + timedelta(days=(weekday - d.weekday()) % 7 + 7 * (n - 1))


def _easter(year: int) -> date:
    """Gregorian Easter Sunday (Meeus/Jones/Butcher)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e, f = b // 4, b % 4, (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    return date(year, (h + l - 7 * m + 114) // 31, (h + l - 7 * m + 114) % 31 + 1)


def _fixed_or_rolling(month: int, day: int) -> date:
    today = date.today()
    d = date(today.year, month, day)
    return d if d >= today else date(today.year + 1, month, day)


_HOLIDAYS = {
    "christmas": (12, 25), "christmas day": (12, 25), "christmas eve": (12, 24),
    "boxing day": (12, 26),
    "new year": (1, 1), "new year's": (1, 1), "new year's day": (1, 1),
    "new years day": (1, 1), "new year's eve": (12, 31), "new years eve": (12, 31),
    "halloween": (10, 31), "valentine's day": (2, 14), "valentines day": (2, 14),
    "valentine's": (2, 14), "valentines": (2, 14),
    "independence day": (7, 4), "july 4th": (7, 4), "fourth of july": (7, 4),
    "st patrick's day": (3, 17), "st. patrick's day": (3, 17),
    "april fools": (4, 1), "april fool's day": (4, 1),
}


def _parse_date(raw: str) -> date | None:
    s = raw.strip().lower().rstrip("?.!")
    s = re.sub(r"\s+", " ", s)
    today = date.today()
    if s in _HOLIDAYS:
        return _fixed_or_rolling(*_HOLIDAYS[s])
    if s == "thanksgiving":
        d = _nth_weekday(today.year, 11, 3, 4)  # 4th Thursday of November
        return d if d >= today else _nth_weekday(today.year + 1, 11, 3, 4)
    if s == "easter":
        d = _easter(today.year)
        return d if d >= today else _easter(today.year + 1)
    if s in ("today", "now"):
        return today
    if s == "tomorrow":
        return today + timedelta(days=1)
    if s == "yesterday":
        return today - timedelta(days=1)
    m = re.match(r"^in (\d+) days?$", s)
    if m:
        return today + timedelta(days=int(m.group(1)))
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d %Y", "%B %d, %Y", "%b %d %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    # "december 25" / "25 december" -> next occurrence
    m = re.match(r"^([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?$", s) or re.match(r"^(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([a-z]+)$", s)
    if m:
        a, b = m.group(1), m.group(2)
        mon = a if a.isalpha() else b
        day = int(b if a.isalpha() else a)
        for i, name in enumerate(_MONTHS, 1):
            if name.startswith(mon):
                try:
                    d = date(today.year, i, day)
                except ValueError:
                    return None
                if d < today:
                    d = date(today.year + 1, i, day)
                return d
    return None


def _fmt_delta(days: int) -> str:
    if days == 0:
        return "today"
    if days > 0:
        return f"in {days} day{'s' if days != 1 else ''}"
    return f"{-days} day{'s' if days != -1 else ''} ago"


class DateCalcTool(Tool):
    spec = ToolSpec(
        name="date_calc",
        description="Date math: days until/between dates, day of week, add/subtract days. "
                    "Accepts 'today', 'tomorrow', '2026-12-25', '12/25/2026', 'december 25'.",
        parameters_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["days_until", "days_between", "add_days", "day_of_week"]},
                "date": {"type": "string", "description": "First/only date"},
                "date2": {"type": "string", "description": "Second date (days_between)"},
                "days": {"type": "integer", "description": "Days to add, may be negative (add_days)"},
            },
            "required": ["action", "date"],
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        action = arguments.get("action")
        d1 = _parse_date(str(arguments.get("date") or ""))
        if d1 is None:
            return ToolResult(ok=False, payload=f"cannot parse date: {arguments.get('date')!r}")
        if action == "days_until":
            delta = (d1 - date.today()).days
            return ToolResult(ok=True, payload=f"{d1.strftime('%A, %B %d %Y')} is {_fmt_delta(delta)}")
        if action == "day_of_week":
            return ToolResult(ok=True, payload=f"{d1.isoformat()} is a {d1.strftime('%A')}")
        if action == "add_days":
            try:
                n = int(arguments.get("days") or 0)
            except (TypeError, ValueError):
                return ToolResult(ok=False, payload="days must be an integer")
            out = d1 + timedelta(days=n)
            return ToolResult(ok=True, payload=f"{out.strftime('%A, %B %d %Y')} ({out.isoformat()})")
        if action == "days_between":
            d2 = _parse_date(str(arguments.get("date2") or ""))
            if d2 is None:
                return ToolResult(ok=False, payload=f"cannot parse date2: {arguments.get('date2')!r}")
            delta = (d2 - d1).days
            return ToolResult(ok=True, payload=f"{abs(delta)} day{'s' if abs(delta) != 1 else ''} between {d1.isoformat()} and {d2.isoformat()}")
        return ToolResult(ok=False, payload=f"unknown action: {action}")


# ---- randomness ------------------------------------------------------------

_PW_ALPHABET = "abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_PW_SYMBOLS = "!@#$%&*+-=?~"


class RandomTool(Tool):
    spec = ToolSpec(
        name="random",
        description="Secure randomness: flip a coin, roll dice, random number in a range, "
                    "uuid, generate a password, or pick a random item from a list.",
        parameters_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["coin", "dice", "number", "uuid", "password", "pick"]},
                "sides": {"type": "integer", "minimum": 2, "maximum": 1000, "default": 6},
                "min": {"type": "integer", "default": 1},
                "max": {"type": "integer", "default": 100},
                "length": {"type": "integer", "minimum": 4, "maximum": 128, "default": 16},
                "items": {"type": "string", "description": "Comma-separated options (pick)"},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        action = arguments.get("action")
        try:
            if action == "coin":
                return ToolResult(ok=True, payload=secrets.choice(["Heads", "Tails"]))
            if action == "dice":
                sides = max(2, min(1000, int(arguments.get("sides") or 6)))
                return ToolResult(ok=True, payload=f"d{sides}: {secrets.randbelow(sides) + 1}")
            if action == "number":
                lo, hi = int(arguments.get("min", 1)), int(arguments.get("max", 100))
                if lo > hi:
                    lo, hi = hi, lo
                return ToolResult(ok=True, payload=str(lo + secrets.randbelow(hi - lo + 1)))
            if action == "uuid":
                return ToolResult(ok=True, payload=str(uuid.uuid4()))
            if action == "password":
                n = max(4, min(128, int(arguments.get("length") or 16)))
                alphabet = _PW_ALPHABET + _PW_SYMBOLS
                # guarantee at least one symbol
                pw = [secrets.choice(_PW_ALPHABET) for _ in range(n - 1)] + [secrets.choice(_PW_SYMBOLS)]
                secrets.SystemRandom().shuffle(pw)
                return ToolResult(ok=True, payload="".join(pw))
            if action == "pick":
                items = [i.strip() for i in str(arguments.get("items") or "").split(",") if i.strip()]
                if not items:
                    return ToolResult(ok=False, payload="items required for pick")
                return ToolResult(ok=True, payload=secrets.choice(items))
        except (TypeError, ValueError) as e:
            return ToolResult(ok=False, payload=f"bad argument: {e}")
        return ToolResult(ok=False, payload=f"unknown action: {action}")


# ---- text transforms -------------------------------------------------------

class TextTool(Tool):
    spec = ToolSpec(
        name="text_tools",
        description="Text transforms: base64/url encode & decode, sha256/md5 hash, "
                    "word count, upper/lower/title case, reverse.",
        parameters_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string",
                           "enum": ["base64_encode", "base64_decode", "url_encode", "url_decode",
                                    "sha256", "md5", "word_count", "upper", "lower", "title", "reverse"]},
                "text": {"type": "string"},
            },
            "required": ["action", "text"],
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            text = sanitize_tool_argument_string(str(arguments["text"]), field_name="text")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        action = arguments.get("action")
        try:
            if action == "base64_encode":
                out = base64.b64encode(text.encode()).decode()
            elif action == "base64_decode":
                out = base64.b64decode(text, validate=True).decode("utf-8", "replace")
            elif action == "url_encode":
                out = urllib.parse.quote_plus(text)
            elif action == "url_decode":
                out = urllib.parse.unquote_plus(text)
            elif action == "sha256":
                out = hashlib.sha256(text.encode()).hexdigest()
            elif action == "md5":
                out = hashlib.md5(text.encode()).hexdigest()
            elif action == "word_count":
                words = len(text.split())
                lines = text.count("\n") + 1
                out = f"{words} words, {len(text)} chars, {lines} lines"
            elif action == "upper":
                out = text.upper()
            elif action == "lower":
                out = text.lower()
            elif action == "title":
                out = text.title()
            elif action == "reverse":
                out = text[::-1]
            else:
                return ToolResult(ok=False, payload=f"unknown action: {action}")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        return ToolResult(ok=True, payload=out[:4000])
