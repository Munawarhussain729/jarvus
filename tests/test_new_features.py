"""Tests for the new Jarvus capabilities (no mic / network / model required)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.tools.applescript import escape
from app.tools.filesystem import _validate_path


class TestAppleScriptEscape:
    def test_quotes_escaped(self):
        assert escape('say "hi"') == 'say \\"hi\\"'

    def test_backslash_escaped(self):
        assert escape("a\\b") == "a\\\\b"

    def test_combined(self):
        # input x\"y -> backslash doubled, quote escaped -> x\\\"y
        assert escape('x\\"y') == 'x\\\\\\"y'


class TestPathSandbox:
    def test_home_allowed(self):
        p = _validate_path("~/somefile.txt")
        assert str(p).startswith(str(Path.home()))

    def test_tmp_allowed(self):
        p = _validate_path("/tmp/x.txt")
        assert str(p).startswith(("/tmp", "/private/tmp"))  # macOS resolves /tmp -> /private/tmp

    def test_etc_rejected(self):
        with pytest.raises(ValueError):
            _validate_path("/etc/passwd")

    def test_extra_roots_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("JARVUS_FS_ROOTS", str(tmp_path))
        p = _validate_path(str(tmp_path / "ok.txt"))
        assert p.parent == tmp_path

    def test_traversal_rejected(self):
        with pytest.raises(ValueError):
            _validate_path("/tmp/../../etc/passwd")


class TestMemoryTools:
    @pytest.fixture()
    def memory(self, tmp_path):
        from app.memory.enhanced import EnhancedMemory
        m = EnhancedMemory(tmp_path / "test.db")
        yield m
        m.close()

    def test_remember_recall_forget(self, memory):
        from app.tools.memory_tools import ForgetTool, RecallTool, RememberTool
        assert RememberTool(memory).run({"key": "Name", "value": "Alex"}).ok
        # keys are normalised to lowercase
        assert memory.get_fact("name") == "Alex"
        res = RecallTool(memory).run({"key": "name"})
        assert res.ok and "Alex" in res.payload
        assert ForgetTool(memory).run({"key": "name"}).ok
        assert memory.get_fact("name") is None

    def test_recall_all_lists_facts(self, memory):
        from app.tools.memory_tools import RecallTool, RememberTool
        RememberTool(memory).run({"key": "a", "value": "1"})
        RememberTool(memory).run({"key": "b", "value": "2"})
        payload = RecallTool(memory).run({}).payload
        assert "a: 1" in payload and "b: 2" in payload

    def test_facts_prompt_block(self, memory):
        memory.set_fact("name", "Alex")
        block = memory.facts_prompt_block()
        assert "name: Alex" in block
        empty = memory.delete_fact("name")
        assert empty and memory.facts_prompt_block() == ""


class TestVoiceConfirmer:
    def _make_voice(self, answer: str):
        from app.voice.session import JarvusVoice

        listener = type("L", (), {})()
        listener.paused = __import__("threading").Event()
        listener.listen_once = lambda timeout=8.0: answer
        listener.callback = None
        listener.expect_command = __import__("threading").Event()

        tts = type("T", (), {"speak": lambda self, t: None})()
        agent = type("A", (), {})()  # confirmer path never touches the agent
        v = JarvusVoice(agent, listener=listener, tts=tts, always_speak=False)
        return v

    @pytest.mark.parametrize("answer", ["yes", "yeah", "confirm", "go ahead", "yes.", "Okay!", "yes please"])
    def test_approvals(self, answer):
        assert self._make_voice(answer).voice_confirmer("run_command", "x") is True

    @pytest.mark.parametrize("answer", ["no", "nope", "", "maybe", "yes I think"])
    def test_rejections(self, answer):
        assert self._make_voice(answer).voice_confirmer("run_command", "x") is False


class TestCleanForSpeech:
    def test_strips_tool_markers_and_markdown(self):
        from app.voice.session import clean_for_speech
        out = clean_for_speech("[tool ok] done\n```code``` `inline` Jarvus> hello")
        assert "[" not in out and "`" not in out and "hello" in out

    def test_truncates(self):
        from app.voice.session import clean_for_speech
        assert len(clean_for_speech("x" * 5000)) <= 600


class TestServerConfirmer:
    def test_request_scoped_confirm(self):
        from app import server
        # default: denied
        assert server._request_confirmer("run_command", "x") is False
        server._confirm_ctx.allow = True
        assert server._request_confirmer("run_command", "x") is True
        server._confirm_ctx.allow = False


class TestQuickIntents:
    def test_timer_variants(self):
        from app.assistant.intents import match
        assert match("set a 5 minute timer for tea") == ("timer", '{"minutes": 5.0, "label": "tea"}')
        name, args = match("set a timer for 30 seconds")
        assert name == "timer" and '"minutes": 0.5' in args

    def test_no_match_falls_through(self):
        from app.assistant.intents import match
        assert match("tell me a joke") is None
        assert match("open the file report.txt") is None

    def test_clock_and_notifications(self):
        from app.assistant.intents import match
        assert match("what time is it") == ("clock", "{}")
        assert match("read my notifications") == ("read_notifications", "{}")
        assert match("hey jarvus, can you read my notifications") == ("read_notifications", "{}")

    def test_battery_and_web_and_reminder(self):
        from app.assistant.intents import match
        assert match("how much battery is left") == ("battery", "{}")
        assert match("is my macbook charging") == ("battery", "{}")
        assert match("look up iphone 18") == ("web_search", '{"query": "iphone 18"}')
        assert match("search my files for receipts") is None
        assert match("remind me to call mom") == ("reminder", '{"text": "call mom"}')


class TestDueDateParser:
    def test_relative_and_named_days(self):
        from app.tools.reminders import _parse_due
        now = __import__("datetime").datetime.now()
        assert (_parse_due("in 30 minutes") - now).total_seconds() > 1700
        assert _parse_due("tomorrow 3pm").hour == 15
        assert _parse_due("tonight").hour == 20
        assert _parse_due("next tuesday").weekday() == 1

    def test_iso_and_us_dates(self):
        from app.tools.reminders import _parse_due
        d = _parse_due("2026-10-20 15:00")
        assert (d.year, d.month, d.day, d.hour) == (2026, 10, 20, 15)
        assert _parse_due("10/25/2026").day == 25

    def test_unparseable_returns_none(self):
        from app.tools.reminders import _parse_due
        assert _parse_due("blargh") is None
        assert _parse_due("") is None


class TestNotesTool:
    def test_add_list_search(self, tmp_path, monkeypatch):
        monkeypatch.setenv("JARVUS_DATA_DIR", str(tmp_path))
        from app.tools.notes import NoteTool
        t = NoteTool()
        assert t.run({"action": "add", "text": "buy milk"}).ok
        assert "buy milk" in t.run({"action": "list"}).payload
        assert "buy milk" in t.run({"action": "search", "text": "milk"}).payload
        assert "No notes matching" in t.run({"action": "search", "text": "eggs"}).payload
