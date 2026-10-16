"""Live internet facts - currency, crypto, stocks, news, wikipedia, jokes, network info.

All use free, key-less JSON endpoints via urllib (no extra deps).
"""
from __future__ import annotations

import json
import re
import subprocess
import urllib.parse
import urllib.request
from typing import Any

from app.security.policy import RiskLevel
from app.security.sanitize import sanitize_tool_argument_string
from app.tools.base import Tool, ToolResult, ToolSpec

_UA_JSON = {"User-Agent": "Jarvus/1.0 (local assistant)", "Accept": "application/json"}
_UA_BROWSER = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def _get_json(url: str, timeout: int = 10, headers: dict | None = None) -> Any:
    req = urllib.request.Request(url, headers={**_UA_JSON, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _get_text(url: str, timeout: int = 10, headers: dict | None = None) -> str:
    req = urllib.request.Request(url, headers={**_UA_JSON, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def _fmt_price(v: float) -> str:
    """Human-friendly price: 82,307 / 0.00001234."""
    if v >= 100:
        return f"{v:,.2f}"
    if v >= 1:
        return f"{v:,.4f}".rstrip("0").rstrip(".")
    return f"{v:.6g}"


def _clean_code(raw: str, *, field: str) -> str:
    code = sanitize_tool_argument_string(raw, field_name=field).strip().upper().rstrip(".?")
    if not re.fullmatch(r"[A-Z]{3}", code):
        raise ValueError(f"{field} must be a 3-letter code like USD")
    return code


# ---- currency --------------------------------------------------------------

class CurrencyTool(Tool):
    spec = ToolSpec(
        name="currency",
        description="Convert between currencies or get exchange rates (Frankfurter, ECB data). "
                    "Use for 'how much is 50 euros in dollars', 'exchange rate', etc.",
        parameters_schema={
            "type": "object",
            "properties": {
                "amount": {"type": "number", "default": 1},
                "from_currency": {"type": "string", "description": "3-letter code, e.g. USD", "default": "USD"},
                "to_currency": {"type": "string", "description": "3-letter code, or 'all' for every rate"},
            },
            "required": ["to_currency"],
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            amount = float(arguments.get("amount") or 1)
            base = _clean_code(str(arguments.get("from_currency") or "USD"), field="from_currency")
            target_raw = str(arguments.get("to_currency") or "").strip()
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        try:
            if target_raw.lower() in ("all", "", "*"):
                data = _get_json(f"https://api.frankfurter.dev/v1/latest?base={base}")
                rates = data.get("rates") or {}
                if not rates:
                    return ToolResult(ok=False, payload=f"unknown currency: {base}")
                pairs = ", ".join(f"{k} {_fmt_price(v * amount)}" for k, v in sorted(rates.items()))
                return ToolResult(ok=True, payload=f"{amount:g} {base} = {pairs}")
            targets = [_clean_code(c, field="to_currency")
                       for c in re.split(r"[,\s]+", target_raw) if c.strip()]
            symbols = ",".join(targets)
            data = _get_json(
                f"https://api.frankfurter.dev/v1/latest?base={base}&symbols={urllib.parse.quote(symbols)}")
            rates = data.get("rates") or {}
            if not rates:
                return ToolResult(ok=False, payload="no rates returned (check the currency codes)")
            parts = [f"{_fmt_price(v * amount)} {k}" for k, v in sorted(rates.items())]
            return ToolResult(ok=True, payload=f"{amount:g} {base} = {' = '.join(parts)}")
        except Exception as e:
            return ToolResult(ok=False, payload=f"currency lookup failed: {e}")


# ---- crypto ----------------------------------------------------------------

_COIN_ALIASES = {
    "btc": "bitcoin", "eth": "ethereum", "doge": "dogecoin", "sol": "solana",
    "ada": "cardano", "xrp": "ripple", "ltc": "litecoin", "bnb": "binancecoin",
    "usdt": "tether", "usdc": "usd-coin", "dot": "polkadot", "avax": "avalanche-2",
}


class CryptoTool(Tool):
    spec = ToolSpec(
        name="crypto",
        description="Get live cryptocurrency prices with 24h change (CoinGecko). "
                    "Accepts names or tickers: bitcoin/btc, ethereum/eth, dogecoin/doge, solana/sol.",
        parameters_schema={
            "type": "object",
            "properties": {
                "coins": {"type": "string", "description": "Comma-separated coins, e.g. 'bitcoin,ethereum'", "default": "bitcoin"},
                "vs_currency": {"type": "string", "default": "usd"},
            },
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        raw = str(arguments.get("coins") or "bitcoin").lower()
        ids = [_COIN_ALIASES.get(c.strip(), c.strip()) for c in raw.split(",") if c.strip()][:10]
        if not ids or not all(re.fullmatch(r"[a-z0-9-]+", i) for i in ids):
            return ToolResult(ok=False, payload="invalid coin names")
        try:
            vs = _clean_code(str(arguments.get("vs_currency") or "usd"), field="vs_currency").lower()
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        from app.tools._cache import get as _cget, put as _cput
        ckey = f"crypto:{','.join(sorted(ids))}:{vs}"
        cached = _cget(ckey)
        if cached is not None:
            return ToolResult(ok=True, payload=cached)
        url = ("https://api.coingecko.com/api/v3/simple/price"
               f"?ids={urllib.parse.quote(','.join(ids))}&vs_currencies={vs}&include_24hr_change=true")
        try:
            data = _get_json(url, timeout=15)
        except Exception as e:
            return ToolResult(ok=False, payload=f"crypto lookup failed: {e}")
        lines = []
        for coin in ids:
            entry = data.get(coin)
            if not entry or vs not in entry:
                lines.append(f"{coin}: not found")
                continue
            chg = entry.get(f"{vs}_24h_change")
            chg_s = f" ({chg:+.1f}% 24h)" if isinstance(chg, (int, float)) else ""
            lines.append(f"{coin}: {_fmt_price(entry[vs])} {vs.upper()}{chg_s}")
        payload = "\n".join(lines)
        _cput(ckey, payload, ttl_s=120)  # live prices: short TTL
        return ToolResult(ok=True, payload=payload)


# ---- stocks ----------------------------------------------------------------

_STOCK_NAME_MAP = {
    "apple": "AAPL", "google": "GOOGL", "alphabet": "GOOGL", "microsoft": "MSFT",
    "tesla": "TSLA", "amazon": "AMZN", "nvidia": "NVDA", "meta": "META",
    "netflix": "NFLX", "amd": "AMD", "intel": "INTC", "disney": "DIS",
    "s&p 500": "^GSPC", "s&p": "^GSPC", "dow": "^DJI", "nasdaq": "^IXIC",
    "bitcoin": "BTC-USD", "gold": "GC=F", "s&p500": "^GSPC",
}


class StockTool(Tool):
    spec = ToolSpec(
        name="stock",
        description="Get a live stock/index quote (Yahoo Finance). Accepts tickers (AAPL) "
                    "or common names (apple, tesla, s&p 500).",
        parameters_schema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Ticker or company name"},
            },
            "required": ["symbol"],
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            raw = sanitize_tool_argument_string(str(arguments["symbol"]), field_name="symbol").strip().lower()
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        symbol = _STOCK_NAME_MAP.get(raw, raw.upper())
        if not re.fullmatch(r"[A-Z0-9.^=-]{1,12}", symbol):
            return ToolResult(ok=False, payload=f"invalid symbol: {arguments['symbol']!r}")
        from app.tools._cache import get as _cget, put as _cput
        ckey = f"stock:{symbol}"
        cached = _cget(ckey)
        if cached is not None:
            return ToolResult(ok=True, payload=cached)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}?interval=1d&range=5d"
        try:
            data = _get_json(url, timeout=12, headers=_UA_BROWSER)
            meta = (data.get("chart", {}).get("result") or [{}])[0].get("meta", {})
        except Exception as e:
            return ToolResult(ok=False, payload=f"quote lookup failed: {e}")
        price = meta.get("regularMarketPrice")
        prev = meta.get("previousClose") or meta.get("chartPreviousClose")
        if price is None:
            return ToolResult(ok=False, payload=f"no quote for {symbol}")
        cur = meta.get("currency", "USD")
        name = meta.get("shortName") or meta.get("longName") or symbol
        chg = ""
        if prev:
            pct = (price - prev) / prev * 100
            chg = f" ({pct:+.2f}% vs prev close {_fmt_price(prev)})"
        payload = f"{name} [{symbol}]: {_fmt_price(price)} {cur}{chg}"
        _cput(ckey, payload, ttl_s=120)
        return ToolResult(ok=True, payload=payload)


# ---- news ------------------------------------------------------------------

class NewsTool(Tool):
    spec = ToolSpec(
        name="news",
        description="Top tech news headlines (Hacker News front page), or HN stories "
                    "about a topic. For general web results use web_search instead.",
        parameters_schema={
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Optional topic to search within news"},
                "count": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            },
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            count = max(1, min(10, int(arguments.get("count") or 5)))
            topic = str(arguments.get("topic") or "").strip()
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        if topic:
            url = ("https://hn.algolia.com/api/v1/search?tags=story"
                   f"&hitsPerPage={count}&query={urllib.parse.quote(topic)}")
        else:
            url = f"https://hn.algolia.com/api/v1/search?tags=front_page&hitsPerPage={count}"
        try:
            data = _get_json(url, timeout=12)
        except Exception as e:
            return ToolResult(ok=False, payload=f"news lookup failed: {e}")
        hits = data.get("hits") or []
        if not hits:
            return ToolResult(ok=True, payload="No headlines found.")
        lines = []
        for i, h in enumerate(hits[:count], 1):
            title = h.get("title") or "(untitled)"
            points = h.get("points") or 0
            link = h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}"
            lines.append(f"{i}. {title} — {points} pts\n   {link}")
        return ToolResult(ok=True, payload="\n".join(lines))


# ---- wikipedia -------------------------------------------------------------

class WikipediaTool(Tool):
    spec = ToolSpec(
        name="wikipedia",
        description="Get a short Wikipedia summary of a person, place or topic.",
        parameters_schema={
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Article title, e.g. 'Alan Turing'"},
            },
            "required": ["topic"],
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            topic = sanitize_tool_argument_string(str(arguments["topic"]), field_name="topic").strip()
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        if not topic:
            return ToolResult(ok=False, payload="topic required")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(topic.replace(' ', '_'))}"
        try:
            data = _get_json(url, timeout=12)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return ToolResult(ok=False, payload=f"No Wikipedia article for '{topic}'")
            return ToolResult(ok=False, payload=f"wikipedia lookup failed: {e}")
        except Exception as e:
            return ToolResult(ok=False, payload=f"wikipedia lookup failed: {e}")
        extract = (data.get("extract") or "").strip()
        if not extract:
            return ToolResult(ok=False, payload=f"No summary available for '{topic}'")
        link = (data.get("content_urls") or {}).get("desktop", {}).get("page", "")
        title = data.get("title") or topic
        if len(extract) > 900:
            extract = extract[:900].rsplit(" ", 1)[0] + "…"
        return ToolResult(ok=True, payload=f"{title}: {extract}\n{link}".strip())


# ---- joke ------------------------------------------------------------------

class JokeTool(Tool):
    spec = ToolSpec(
        name="joke",
        description="Tell a random dad joke.",
        parameters_schema={"type": "object", "properties": {}, "additionalProperties": False},
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            joke = _get_text("https://icanhazdadjoke.com/",
                             timeout=10, headers={"Accept": "text/plain"}).strip()
        except Exception as e:
            return ToolResult(ok=False, payload=f"joke lookup failed: {e}")
        return ToolResult(ok=True, payload=joke or "No joke today.")


# ---- network info ----------------------------------------------------------

_IP_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
_HOST_RE = re.compile(r"^[a-z0-9.-]+\.[a-z]{2,}$", re.I)


def _local_network_info() -> list[str]:
    lines: list[str] = []
    for iface, label in (("en0", "en0 (usually Wi-Fi)"), ("en1", "en1")):
        try:
            proc = subprocess.run(["ipconfig", "getifaddr", iface],
                                  capture_output=True, text=True, timeout=5)
            ip = proc.stdout.strip()
            if proc.returncode == 0 and ip:
                lines.append(f"{label}: {ip}")
        except Exception:
            continue
    try:
        proc = subprocess.run(["ipconfig", "getsummary", "en0"],
                              capture_output=True, text=True, timeout=5)
        m = re.search(r"^\s*SSID\s*:\s*(.+)$", proc.stdout, re.M)
        if m:
            lines.append(f"Wi-Fi SSID: {m.group(1).strip()}")
    except Exception:
        pass
    try:
        proc = subprocess.run(["route", "-n", "get", "default"],
                              capture_output=True, text=True, timeout=5)
        m = re.search(r"gateway:\s*(\S+)", proc.stdout)
        if m:
            lines.append(f"Gateway: {m.group(1)}")
    except Exception:
        pass
    return lines


def _public_ip_info(target: str = "") -> list[str]:
    url = f"http://ip-api.com/json/{urllib.parse.quote(target)}?fields=status,message,query,city,regionName,country,isp,org"
    data = _get_json(url, timeout=8)
    if data.get("status") != "success":
        raise RuntimeError(data.get("message") or "lookup failed")
    where = ", ".join(x for x in (data.get("city"), data.get("regionName"), data.get("country")) if x)
    label = target or "this machine"
    line = f"{label}: {data.get('query')}"
    if where:
        line += f" — {where}"
    if data.get("isp"):
        line += f" ({data['isp']})"
    return [line]


class NetworkInfoTool(Tool):
    spec = ToolSpec(
        name="network_info",
        description="Network status: local IP, Wi-Fi SSID, gateway, public IP and its "
                    "approximate location. 'lookup' geolocates any IP or domain.",
        parameters_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["all", "local", "public", "lookup"], "default": "all"},
                "target": {"type": "string", "description": "IP or domain (lookup only)"},
            },
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        action = arguments.get("action") or "all"
        lines: list[str] = []
        if action == "lookup":
            try:
                target = sanitize_tool_argument_string(str(arguments.get("target") or ""), field_name="target").strip()
            except Exception as e:
                return ToolResult(ok=False, payload=str(e))
            if not (_IP_RE.match(target) or _HOST_RE.match(target)):
                return ToolResult(ok=False, payload="target must be an IP or domain")
            try:
                lines = _public_ip_info(target)
            except Exception as e:
                return ToolResult(ok=False, payload=f"lookup failed: {e}")
        else:
            if action in ("all", "local"):
                lines += _local_network_info()
            if action in ("all", "public"):
                try:
                    lines += _public_ip_info()
                except Exception as e:
                    lines.append(f"public IP: unavailable ({e})")
        return ToolResult(ok=True, payload="\n".join(lines) or "no network info")
