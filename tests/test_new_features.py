"""Tests for the new Jarvus capabilities (no mic / network / model required)."""
from __future__ import annotations

import os
from datetime import datetime, timedelta
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

    @pytest.mark.parametrize("answer", ["yes", "yeah", "confirm", "go ahead", "yes.", "Okay!", "yes please",
                                        "jarvus, yes.", "hey jarvis okay", "jarvus yes"])
    def test_approvals(self, answer):
        assert self._make_voice(answer).voice_confirmer("run_command", "x") is True

    @pytest.mark.parametrize("answer", ["no", "nope", "", "maybe", "yes I think"])
    def test_rejections(self, answer):
        assert self._make_voice(answer).voice_confirmer("run_command", "x") is False


class TestHallucinationFilter:
    def test_whisper_silence_artifacts_dropped(self):
        from app.voice.stt import _is_hallucination
        for t in ["Thank you.", "Thanks for watching!", "All right.", "Bye", "Hmm", "you", "..."]:
            assert _is_hallucination(t), t

    def test_real_speech_kept(self):
        from app.voice.stt import _is_hallucination
        for t in ["open safari", "yes", "no", "okay", "set a timer", "Microsoft Word"]:
            assert not _is_hallucination(t), t


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
        # 'what is <concept>' now routes to wikipedia
        assert match("what is the meaning of life") == ("wikipedia", '{"topic": "meaning of life"}')
        assert match("hmm, let me think about that") is None
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
        name, args = match("search my files for receipts")
        assert name == "find_files" and '"query": "receipts"' in args
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


class TestTaskIntents:
    def test_daily_brief_phrases(self):
        from app.assistant.intents import match
        for phrase in ["brief me on my day", "what's on today", "what do i have today",
                       "analyse my tasks", "how's my day", "what's my schedule"]:
            assert match(phrase) == ("daily_brief", "{}"), phrase

    def test_read_intents(self):
        from app.assistant.intents import match
        assert match("list my reminders") == ("list_reminders", "{}")
        assert match("what's on my calendar") == ("calendar_read", "{}")
        name, args = match("show my todos")
        assert name == "todo_write" and '"action": "list"' in args

    def test_reminder_create_still_distinct(self):
        from app.assistant.intents import match
        assert match("remind me to call mom") == ("reminder", '{"text": "call mom"}')


class TestPersistentTodos:
    @pytest.fixture()
    def memory(self, tmp_path):
        from app.memory.enhanced import EnhancedMemory
        m = EnhancedMemory(tmp_path / "test.db")
        yield m
        m.close()

    def test_set_add_complete_list(self, memory):
        from app.tools.plan import TodoWriteTool
        t = TodoWriteTool(memory)
        assert t.run({"todos": [{"content": "ship it", "status": "pending", "priority": "high"}]}).ok
        assert "ship it" in memory.list_todos()[0]["content"]
        assert t.run({"action": "add", "content": "call bob", "priority": "low"}).ok
        assert len(memory.list_todos()) == 2
        assert t.run({"action": "complete", "content": "ship"}).ok
        open_todos = memory.list_todos()
        assert len(open_todos) == 1 and open_todos[0]["content"] == "call bob"
        assert "call bob" in t.run({"action": "list"}).payload

    def test_complete_no_match(self, memory):
        from app.tools.plan import TodoWriteTool
        res = TodoWriteTool(memory).run({"action": "complete", "content": "ghost"})
        assert res.ok and "No open todos" in res.payload

    def test_plan_persists(self, memory):
        from app.tools.plan import PlanTool
        assert PlanTool(memory).run({"goal": "launch", "steps": ["a", "b"]}).ok
        cur = memory._conn.execute("SELECT name, status FROM plans")
        row = cur.fetchone()
        assert row["name"] == "launch" and row["status"] == "active"


class TestReadTools:
    """AppleScript is mocked at the module where it's bound (app.tools.reminders.run)."""

    def _mock_run(self, monkeypatch, payload="", ok=True):
        from app.tools.base import ToolResult
        import app.tools.reminders as R
        monkeypatch.setattr(R, "run", lambda script, timeout=15, ok_payload="done":
                            ToolResult(ok=ok, payload=payload))
        return R

    def test_list_reminders_parses_and_filters(self, monkeypatch):
        now = datetime.now()
        past = (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M")
        today = now.strftime("%Y-%m-%d %H:%M")
        self._mock_run(monkeypatch, f"Buy milk|Reminders|{past}\nCall mom|Home|{today}\nSomeday|Misc|\n")
        from app.tools.reminders import ListRemindersTool
        t = ListRemindersTool()
        res = t.run({"when": "all"})
        assert res.ok and "Buy milk" in res.payload and "OVERDUE" in res.payload and "Someday" in res.payload
        res = t.run({"when": "today"})
        assert "Call mom" in res.payload and "Someday" not in res.payload
        res = t.run({"when": "overdue"})
        assert "Buy milk" in res.payload and "Call mom" not in res.payload

    def test_list_reminders_permission_denied(self, monkeypatch):
        self._mock_run(monkeypatch, payload="execution error: Not authorized to send Apple events (-1743)", ok=False)
        from app.tools.reminders import ListRemindersTool
        res = ListRemindersTool().run({})
        assert not res.ok and "Privacy" in res.payload

    def test_calendar_read_parses(self, monkeypatch):
        now = datetime.now()
        self._mock_run(monkeypatch,
                       f"Standup|Work|{now.strftime('%Y-%m-%d %H:%M')}|{now.strftime('%Y-%m-%d %H:%M')}|false\n")
        from app.tools.reminders import CalendarReadTool
        res = CalendarReadTool().run({"day": "today"})
        assert res.ok and "Standup" in res.payload

    def test_day_bounds(self):
        from app.tools.reminders import day_bounds
        s, e = day_bounds("today")
        now = datetime.now()
        assert (s.hour, s.minute) == (0, 0) and s.date() == now.date() and (e - s).days == 1
        s2, e2 = day_bounds("week")
        assert (e2 - s2).days == 7
        assert day_bounds("blargh") is None


class TestDailyBrief:
    def _memory(self, tmp_path):
        from app.memory.enhanced import EnhancedMemory
        return EnhancedMemory(tmp_path / "t.db")

    def _fns(self, events=None, reminders=None, ev_err="", rem_err=""):
        now = datetime.now()
        return {
            "bounds": lambda d="today", days=1: (now.replace(hour=0, minute=0, second=0, microsecond=0),
                                                 now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)),
            "events": lambda s, e: (events or [], ev_err),
            "reminders": lambda limit=200: (reminders or [], rem_err),
        }

    def test_empty_day(self, tmp_path, monkeypatch):
        monkeypatch.setenv("JARVUS_DATA_DIR", str(tmp_path / "nodata"))
        from app.tools.briefing import gather_day_digest
        out = gather_day_digest(self._memory(tmp_path), self._fns())
        assert "Nothing on the schedule" in out

    def test_merges_events_reminders_todos(self, tmp_path, monkeypatch):
        monkeypatch.setenv("JARVUS_DATA_DIR", str(tmp_path / "nope"))
        mem = self._memory(tmp_path)
        mem.add_todo("finish report", "pending", "high")
        now = datetime.now()
        fns = self._fns(
            events=[{"title": "Standup", "calendar": "Work", "start": now, "end": now, "all_day": False}],
            reminders=[{"text": "Pay bill", "list": "Home",
                        "due": now - timedelta(days=1), }],
        )
        from app.tools.briefing import gather_day_digest
        out = gather_day_digest(mem, fns)
        assert "event" in out and "Standup" in out and "Pay bill" in out and "todo" in out
        mem.close()

    def test_permission_error_surfaced_not_raised(self, tmp_path, monkeypatch):
        monkeypatch.setenv("JARVUS_DATA_DIR", str(tmp_path / "nodata"))
        from app.tools.briefing import gather_day_digest
        out = gather_day_digest(None, self._fns(rem_err="macOS denied access"))
        assert "denied" in out


class TestAutoBrief:
    def test_fires_once_per_local_day(self, tmp_path, monkeypatch):
        monkeypatch.setenv("JARVUS_DATA_DIR", str(tmp_path / "nodata"))
        from app.memory.enhanced import EnhancedMemory
        from app.tools.briefing import maybe_auto_brief
        mem = EnhancedMemory(tmp_path / "t.db")
        now = datetime.now()
        fns = {
            "bounds": lambda d="today", days=1: (now, now + timedelta(days=1)),
            "events": lambda s, e: ([], ""),
            "reminders": lambda limit=200: ([], ""),
        }
        # First call today fires (digest or nothing-scheduled message)
        assert maybe_auto_brief(mem, enabled=True, read_fns=fns) is not None
        # Same local day -> no repeat
        assert maybe_auto_brief(mem, enabled=True, read_fns=fns) is None
        # Local-date rollover -> fires again (not UTC: stored as local ISO date)
        yesterday = (datetime.now().date() - timedelta(days=1)).isoformat()
        mem.set_meta("last_brief_date", yesterday)
        assert maybe_auto_brief(mem, enabled=True, read_fns=fns) is not None
        # Disabled / memory without meta -> never fires
        assert maybe_auto_brief(mem, enabled=False, read_fns=fns) is None
        mem.close()


class TestToolPrefilter:
    """False negatives are the silent-failure risk: a dropped tool is
    unrecoverable, so every ambiguous command must keep its tools."""

    def test_no_keyword_returns_full_set(self):
        from app.assistant.intents import tool_subset_for
        assert tool_subset_for("hmm, let me think about that") is None
        assert tool_subset_for("what is the meaning of life") is None

    def test_union_not_first_match_open_file(self):
        from app.assistant.intents import tool_subset_for
        s = tool_subset_for("open the file report.txt")
        # 'open' matches the app group AND 'file' matches file tools - both kept
        assert {"open_app", "read_file", "write_file"} <= set(s)

    def test_task_keywords_keep_task_tools(self):
        from app.assistant.intents import tool_subset_for
        for phrase, needed in [
            ("remind me to water the plants", "reminder"),
            ("what's on my calendar tomorrow", "calendar_read"),
            ("put 'buy eggs' on my task list", "todo_write"),
            ("set a timer for tea", "timer"),
        ]:
            s = tool_subset_for(phrase)
            assert s is not None and needed in s, (phrase, needed)

    def test_core_always_present(self):
        from app.assistant.intents import tool_subset_for
        s = tool_subset_for("what is my battery at")
        assert {"remember", "recall", "scratchpad", "todo_write", "daily_brief"} <= set(s)
        assert "battery" in s

    def test_payload_subset_via_registry(self, tmp_path):
        from app.bootstrap import build_registry
        from app.memory.enhanced import EnhancedMemory
        reg = build_registry(EnhancedMemory(tmp_path / "t.db"))
        full = reg.openai_tools_payload()
        sub = reg.tools_payload_for({"clock", "timer"})
        assert {t["function"]["name"] for t in sub} == {"clock", "timer"}
        assert len(reg.tools_payload_for(None)) == len(full)
        # cached payload returns fresh lists - mutating one doesn't corrupt cache
        sub.append({"x": 1})
        assert len(reg.openai_tools_payload()) == len(full)
