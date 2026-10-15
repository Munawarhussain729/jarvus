"""Tests for the extended skill set (no network/model required; HTTP is stubbed)."""
from __future__ import annotations

import json
from datetime import date, timedelta

import pytest


class TestCalcTool:
    def test_basic(self):
        from app.tools.utilities import CalcTool
        t = CalcTool()
        assert t.run({"expression": "2+2*3"}).payload.endswith("= 8")
        assert t.run({"expression": "sqrt(16)"}).payload.endswith("= 4")
        assert "2.5" in t.run({"expression": "10/4"}).payload
        assert t.run({"expression": "(1/100)*80"}).ok

    def test_safety(self):
        from app.tools.utilities import CalcTool
        t = CalcTool()
        for bad in ["__import__('os')", "open('x')", "import os", "[1,2]", "x=1",
                    "globals()", "'a'+'b'"]:
            assert not t.run({"expression": bad}).ok, bad

    def test_math_errors(self):
        from app.tools.utilities import CalcTool
        t = CalcTool()
        assert not t.run({"expression": "1/0"}).ok
        assert not t.run({"expression": "2**1000000"}).ok
        assert not t.run({"expression": ""}).ok


class TestUnitConvert:
    def test_length_and_alias(self):
        from app.tools.utilities import UnitConvertTool
        t = UnitConvertTool()
        res = t.run({"value": 1, "from_unit": "mile", "to_unit": "km"})
        assert res.ok and "1.60934" in res.payload

    def test_temperature(self):
        from app.tools.utilities import UnitConvertTool
        t = UnitConvertTool()
        res = t.run({"value": 100, "from_unit": "c", "to_unit": "fahrenheit"})
        assert res.ok and "212" in res.payload

    def test_mismatched_kinds_and_unknown(self):
        from app.tools.utilities import UnitConvertTool
        t = UnitConvertTool()
        assert not t.run({"value": 1, "from_unit": "kg", "to_unit": "miles"}).ok
        assert not t.run({"value": 1, "from_unit": "furlong", "to_unit": "m"}).ok

    def test_data(self):
        from app.tools.utilities import UnitConvertTool
        res = UnitConvertTool().run({"value": 1, "from_unit": "gb", "to_unit": "mb"})
        assert res.ok and "1024" in res.payload


class TestDateCalc:
    def test_days_until(self):
        from app.tools.utilities import DateCalcTool
        t = DateCalcTool()
        assert "today" in t.run({"action": "days_until", "date": "today"}).payload
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        assert "in 1 day" in t.run({"action": "days_until", "date": tomorrow}).payload

    def test_day_of_week_and_between(self):
        from app.tools.utilities import DateCalcTool
        t = DateCalcTool()
        res = t.run({"action": "day_of_week", "date": "2026-10-15"})
        assert res.ok and "Thursday" in res.payload
        res = t.run({"action": "days_between", "date": "2026-01-01", "date2": "2026-01-11"})
        assert "10 days" in res.payload

    def test_add_days_and_named_month(self):
        from app.tools.utilities import DateCalcTool
        t = DateCalcTool()
        res = t.run({"action": "add_days", "date": "today", "days": 7})
        assert res.ok and (date.today() + timedelta(days=7)).isoformat() in res.payload
        res = t.run({"action": "days_until", "date": "december 25"})
        assert res.ok and "December 25" in res.payload
        assert not t.run({"action": "days_until", "date": "blargh"}).ok


class TestRandomTool:
    def test_actions(self):
        from app.tools.utilities import RandomTool
        t = RandomTool()
        assert t.run({"action": "coin"}).payload in ("Heads", "Tails")
        assert 1 <= int(t.run({"action": "dice", "sides": 6}).payload.split(": ")[1]) <= 6
        assert 5 <= int(t.run({"action": "number", "min": 5, "max": 10}).payload) <= 10
        assert len(t.run({"action": "uuid"}).payload) == 36
        assert len(t.run({"action": "password", "length": 20}).payload) == 20
        assert t.run({"action": "pick", "items": "a,b,c"}).payload in ("a", "b", "c")
        assert not t.run({"action": "pick", "items": ""}).ok


class TestTextTool:
    def test_roundtrip_and_counts(self):
        from app.tools.utilities import TextTool
        t = TextTool()
        enc = t.run({"action": "base64_encode", "text": "hello"}).payload
        assert t.run({"action": "base64_decode", "text": enc}).payload == "hello"
        assert "4 words" in t.run({"action": "word_count", "text": "one two three four"}).payload
        assert t.run({"action": "sha256", "text": "x"}).ok
        assert t.run({"action": "upper", "text": "abc"}).payload == "ABC"


class TestInternetToolsStubbed:
    """HTTP is monkeypatched at the module helper - no real network."""

    def _patch(self, monkeypatch, payload):
        import app.tools.internet as I
        monkeypatch.setattr(I, "_get_json", lambda url, timeout=10, headers=None: payload)

    def test_currency(self, monkeypatch):
        self._patch(monkeypatch, {"rates": {"EUR": 0.9, "GBP": 0.75}})
        from app.tools.internet import CurrencyTool
        res = CurrencyTool().run({"amount": 100, "from_currency": "usd", "to_currency": "eur"})
        assert res.ok and "90" in res.payload and "EUR" in res.payload
        assert not CurrencyTool().run({"amount": 1, "from_currency": "usd", "to_currency": "XX"}).ok

    def test_crypto(self, monkeypatch):
        self._patch(monkeypatch, {"bitcoin": {"usd": 80000, "usd_24h_change": 2.5}})
        from app.tools.internet import CryptoTool
        res = CryptoTool().run({"coins": "btc"})
        assert res.ok and "80,000" in res.payload and "+2.5%" in res.payload

    def test_stock(self, monkeypatch):
        self._patch(monkeypatch, {"chart": {"result": [{"meta": {
            "symbol": "AAPL", "regularMarketPrice": 250.0, "previousClose": 240.0,
            "currency": "USD", "exchangeName": "NMS", "shortName": "Apple Inc."}}]}})
        from app.tools.internet import StockTool
        res = StockTool().run({"symbol": "apple"})
        assert res.ok and "AAPL" in res.payload and "250" in res.payload and "+" in res.payload

    def test_news(self, monkeypatch):
        self._patch(monkeypatch, {"hits": [{"title": "AI News", "points": 100,
                                            "url": "https://x", "objectID": "1"}]})
        from app.tools.internet import NewsTool
        res = NewsTool().run({"count": 1})
        assert res.ok and "AI News" in res.payload

    def test_wikipedia(self, monkeypatch):
        self._patch(monkeypatch, {"title": "Alan Turing", "extract": "A mathematician.",
                                  "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Alan_Turing"}}})
        from app.tools.internet import WikipediaTool
        res = WikipediaTool().run({"topic": "alan turing"})
        assert res.ok and "mathematician" in res.payload

    def test_network_lookup_validation(self):
        from app.tools.internet import NetworkInfoTool
        assert not NetworkInfoTool().run({"action": "lookup", "target": "not a host!!"}).ok


class TestMacosExtra:
    def test_open_url_validation(self, monkeypatch):
        import app.tools.macos_extra as M
        calls = []
        monkeypatch.setattr(M.subprocess, "run",
                            lambda *a, **k: calls.append(a[0]) or type("P", (), {"returncode": 0, "stderr": ""})())
        from app.tools.macos_extra import OpenUrlTool
        t = OpenUrlTool()
        assert t.run({"url": "github.com"}).ok
        assert calls and calls[0] == ["open", "https://github.com"]
        assert not t.run({"url": "report.txt"}).ok          # .txt not a web TLD
        assert not t.run({"url": "has space.com"}).ok
        assert not t.run({"url": "file:///etc/passwd"}).ok

    def test_dark_mode_parses_applescript(self, monkeypatch):
        import app.tools.macos_extra as M
        from app.tools.base import ToolResult
        monkeypatch.setattr(M, "run", lambda *a, **k: ToolResult(ok=True, payload="true"))
        from app.tools.macos_extra import DarkModeTool
        res = DarkModeTool().run({"action": "get"})
        assert res.ok and "on" in res.payload

    def test_find_files_rejects_bad_scope(self):
        from app.tools.macos_extra import FindFilesTool
        assert not FindFilesTool().run({"query": "x", "scope": "/etc"}).ok
        assert not FindFilesTool().run({"query": ""}).ok


class TestNewIntents:
    def test_math_phrases(self):
        from app.assistant.intents import match
        assert match("what's 2 plus 2") == ("calculate", '{"expression": "2+2"}')
        assert match("calculate 12*5")[0] == "calculate"
        name, args = match("what's 15 percent of 80")
        assert name == "calculate" and "15/100" in json.loads(args)["expression"]
        # non-math 'what is' goes to wikipedia, not the calculator
        assert match("what is the capital of france")[0] == "wikipedia"

    def test_convert_and_currency(self):
        from app.assistant.intents import match
        name, args = match("convert 5 miles to km")
        assert name == "unit_convert" and json.loads(args)["from_unit"] == "miles"
        name, args = match("convert 100 dollars to euros")
        assert name == "currency" and json.loads(args)["to_currency"] == "EUR"
        name, args = match("how many cm in a meter")
        assert name == "unit_convert"

    def test_crypto_stock_news_joke(self):
        from app.assistant.intents import match
        name, args = match("bitcoin price")
        assert name == "crypto" and json.loads(args)["coins"] == "bitcoin"
        assert match("how much is ethereum")[0] == "crypto"
        name, args = match("apple stock price")
        assert name == "stock" and json.loads(args)["symbol"] == "apple"
        assert match("what's the news")[0] == "news"
        assert match("tell me a joke") == ("joke", "{}")

    def test_wikipedia_and_ip(self):
        from app.assistant.intents import match
        name, args = match("who is alan turing")
        assert name == "wikipedia" and json.loads(args)["topic"] == "alan turing"
        assert match("tell me about the roman empire")[0] == "wikipedia"
        assert match("tell me about my day") == ("daily_brief", "{}")
        name, args = match("look up 8.8.8.8")
        assert name == "network_info" and json.loads(args)["action"] == "lookup"

    def test_macos_and_random(self):
        from app.assistant.intents import match
        assert match("open github.com") == ("open_url", '{"url": "github.com"}')
        assert match("open safari") == ("open_app", '{"app_name": "safari"}')
        assert match("open report.txt") is None
        assert match("turn on dark mode") == ("dark_mode", '{"action": "on"}')
        assert match("dark mode") == ("dark_mode", '{"action": "toggle"}')
        assert match("lock my screen") == ("lock_screen", "{}")
        assert match("flip a coin") == ("random", '{"action": "coin"}')
        assert match("roll a d20") == ("random", '{"action": "dice", "sides": 20}')
        assert match("generate a password") == ("random", '{"action": "password"}')
        assert match("run shortcut meeting mode") == ("run_shortcut", '{"action": "run", "name": "meeting mode"}')
        # bare 'stock' / 'the stock' must not resolve to a symbol
        assert match("check the stock") is None
        name, args = match("remind me in 20 minutes to call mom")
        assert name == "reminder" and json.loads(args)["due"] == "in 20 minutes"

    def test_date_and_file_search(self):
        from app.assistant.intents import match
        name, args = match("how many days until christmas")
        assert name == "date_calc" and json.loads(args)["action"] == "days_until"
        name, args = match("find file named report")
        assert name == "find_files" and json.loads(args)["query"] == "report"
        name, args = match("find documents containing invoice")
        assert name == "find_files" and json.loads(args).get("content") is True

    def test_new_prefilter_groups(self):
        from app.assistant.intents import tool_subset_for
        s = tool_subset_for("what's the bitcoin price")
        assert {"crypto", "stock", "currency"} <= set(s)
        s = tool_subset_for("convert 5 miles to km")
        assert "unit_convert" in s
        s = tool_subset_for("find the file i downloaded")
        assert "find_files" in s
        s = tool_subset_for("lock the screen")
        assert "lock_screen" in s
        assert tool_subset_for("what's the meaning of life") is None
