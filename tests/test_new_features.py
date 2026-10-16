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


class TestContentCleaning:
    """Provider-side safety net: think blocks and leaked reasoning never reach speech."""

    def test_think_block_stripped(self):
        from app.models.local_provider import clean_content
        assert clean_content("<think>long reasoning</think>Done, saved it.") == "Done, saved it."

    def test_unclosed_think_drops_tail(self):
        from app.models.local_provider import clean_content
        assert clean_content("<think>still reasoning") == ""

    def test_reasoning_leak_becomes_empty(self):
        from app.models.local_provider import clean_content
        assert clean_content("Hmm, the user wants me to say OK. I need to") == ""
        assert clean_content("The user asked me to make a game.") == ""

    def test_normal_reply_untouched(self):
        from app.models.local_provider import clean_content
        assert clean_content("Sure, your file is saved.") == "Sure, your file is saved."
        assert clean_content(None) is None


class TestCreateCodeTool:
    def test_writes_and_checks_python(self, tmp_path, monkeypatch):
        import app.tools.code as code_mod
        monkeypatch.setattr(code_mod, "PROJECTS_DIR", tmp_path)
        r = code_mod.CreateCodeTool().run({"filename": "snake.py", "code": "x = 1\nprint(x)\n"})
        assert r.ok and "syntax OK" in r.payload
        assert (tmp_path / "snake.py").exists()

    def test_syntax_error_reported(self, tmp_path, monkeypatch):
        import app.tools.code as code_mod
        monkeypatch.setattr(code_mod, "PROJECTS_DIR", tmp_path)
        r = code_mod.CreateCodeTool().run({"filename": "bad.py", "code": "def broken(:\n"})
        assert not r.ok and "syntax error" in r.payload.lower()
        assert not (tmp_path / "bad.py").exists()  # bad code never touches disk

    def test_bad_code_does_not_clobber_existing(self, tmp_path, monkeypatch):
        import app.tools.code as code_mod
        monkeypatch.setattr(code_mod, "PROJECTS_DIR", tmp_path)
        good = code_mod.CreateCodeTool().run({"filename": "keep.py", "code": "x = 1\n"})
        assert good.ok
        bad = code_mod.CreateCodeTool().run({"filename": "keep.py", "code": "def broken(:\n"})
        assert not bad.ok
        assert (tmp_path / "keep.py").read_text() == "x = 1\n"

    def test_path_traversal_stripped(self, tmp_path, monkeypatch):
        import app.tools.code as code_mod
        monkeypatch.setattr(code_mod, "PROJECTS_DIR", tmp_path)
        r = code_mod.CreateCodeTool().run({"filename": "../evil.py", "code": "pass\n"})
        assert r.ok  # basename only -> stays in projects dir
        assert (tmp_path / "evil.py").exists()


class _FakeListener:
    def __init__(self):
        import threading
        self.paused = threading.Event()
        self.state = "idle"


class _FakeTTS:
    def __init__(self):
        self.spoken: list[str] = []

    def speak(self, text):
        self.spoken.append(text)


def _make_session():
    from app.voice.session import JarvusVoice
    v = JarvusVoice.__new__(JarvusVoice)
    v.tts = _FakeTTS()
    v.listener = _FakeListener()
    import queue, threading
    v._speech_q = queue.Queue(maxsize=8)
    v._speaker_stop = threading.Event()
    v._speaker_thread = None
    v._ensure_speaker()
    return v


class TestSpeechPipeline:
    def test_sentences_emitted_incrementally(self):
        from app.voice.session import _SpeechPipeline
        sess = _make_session()
        p = _SpeechPipeline(sess)
        p.feed("Saved the file. It")
        p.feed(" has two parts.")
        p.finish()
        sess._speech_q.join()
        assert sess.tts.spoken == ["Saved the file.", "It has two parts."]
        sess._speaker_stop.set()

    def test_flush_speaks_remainder(self):
        from app.voice.session import _SpeechPipeline
        sess = _make_session()
        p = _SpeechPipeline(sess)
        p.feed("All done")
        p.finish()
        sess._speech_q.join()
        assert sess.tts.spoken == ["All done"]
        sess._speaker_stop.set()

    def test_think_block_never_spoken_even_when_split(self):
        from app.voice.session import _SpeechPipeline
        sess = _make_session()
        p = _SpeechPipeline(sess)
        p.feed("<thi")      # partial tag - must be held back
        p.feed("nk>secret monologue</thi")
        p.feed("nk>Done. ")
        p.finish()
        sess._speech_q.join()
        assert sess.tts.spoken == ["Done."]
        sess._speaker_stop.set()

    def test_unclosed_think_dropped_at_finish(self):
        from app.voice.session import _SpeechPipeline
        sess = _make_session()
        p = _SpeechPipeline(sess)
        p.feed("Working. <think> hmm the user")
        p.finish()
        sess._speech_q.join()
        assert sess.tts.spoken == ["Working."]
        sess._speaker_stop.set()

    def test_no_duplicate_speech_on_retry(self):
        from app.voice.session import _SpeechPipeline
        sess = _make_session()
        p = _SpeechPipeline(sess)
        p.feed("Done. ")
        p.feed("Done. Again. ")
        p.finish()
        sess._speech_q.join()
        assert sess.tts.spoken == ["Done.", "Again."]
        sess._speaker_stop.set()

    def test_discard_pending_drops_partial(self):
        from app.voice.session import _SpeechPipeline
        sess = _make_session()
        p = _SpeechPipeline(sess)
        p.feed("Let me create th")   # no sentence end - pending
        p.discard_pending()
        p.feed("File saved. ")
        p.finish()
        sess._speech_q.join()
        assert sess.tts.spoken == ["File saved."]
        sess._speaker_stop.set()


class TestVAD:
    def test_silence_and_click_rejected(self):
        import numpy as np
        from app.voice.stt import has_speech, SAMPLE_RATE
        rng = np.random.default_rng(0)
        silence = rng.normal(0, 0.0005, SAMPLE_RATE).astype(np.float32)
        assert not has_speech(silence, rms_gate=0.003)
        click = np.zeros(SAMPLE_RATE, dtype=np.float32)
        click[4000:4800] = 0.3  # loud transient < 12% of frames
        assert not has_speech(click, rms_gate=0.003)

    def test_voiced_audio_passes(self):
        import numpy as np
        from app.voice.stt import has_speech, SAMPLE_RATE
        voiced = np.tile(np.random.default_rng(1).normal(0, 0.02, int(SAMPLE_RATE * .3)),
                         4)[:SAMPLE_RATE].astype(np.float32)
        assert has_speech(voiced, rms_gate=0.003)


class TestSchemaValidation:
    def _router(self):
        from app.assistant.router import ToolRouter
        from app.tools.registry import ToolRegistry
        from app.audit.logger import AuditLogger
        from app.memory.store import MemoryStore
        from pathlib import Path
        import tempfile
        from app.tools.weather import WeatherTool
        reg = ToolRegistry([WeatherTool()])
        mem = MemoryStore(Path(tempfile.mkdtemp()) / "m.db")
        return ToolRouter(reg, AuditLogger(mem), confirmer=lambda n, s: True)

    def test_bad_enum_rejected_with_schema(self):
        r = self._router()
        body = r.handle_tool_call(tool_name="weather", arguments_json='{"format": "banana"}')
        assert "[tool error]" in body and "enum" in body.lower() or "one of" in body

    def test_wrong_type_rejected(self):
        r = self._router()
        body = r.handle_tool_call(tool_name="weather", arguments_json='{"city": 42}')
        assert "[tool error]" in body

    def test_valid_args_pass(self, monkeypatch):
        import app.tools.weather as w
        monkeypatch.setattr(w.WeatherTool, "run", lambda self, a: type("R", (), {"ok": True, "payload": "ok", "detail": {}})())
        r = self._router()
        body = r.handle_tool_call(tool_name="weather", arguments_json='{"city": "Paris"}')
        assert "[tool ok]" in body


class TestProfiles:
    def test_lean_profile_applies(self, monkeypatch, tmp_path):
        # Run in a dir with no .env so only monkeypatched env vars count.
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("JARVUS_PROFILE", "lean")
        monkeypatch.delenv("JARVUS_LLM_NUM_CTX", raising=False)
        monkeypatch.delenv("JARVUS_MAX_TOKENS", raising=False)
        monkeypatch.delenv("JARVUS_LLM_KEEP_ALIVE", raising=False)
        monkeypatch.delenv("JARVUS_STT_MODEL", raising=False)
        from app.config import load_settings
        s = load_settings()
        assert s.max_tokens == 256 and s.llm_num_ctx == 8192

    def test_explicit_var_beats_profile(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("JARVUS_PROFILE", "lean")
        monkeypatch.setenv("JARVUS_MAX_TOKENS", "999")
        from app.config import load_settings
        s = load_settings()
        assert s.max_tokens == 999


class TestNewIntents:
    def test_good_morning_is_daily_brief(self):
        from app.assistant.intents import match
        assert match("good morning")[0] == "daily_brief"

    def test_volume_relative(self):
        from app.assistant.intents import match
        import json
        assert json.loads(match("volume up")[1])["action"] == "up"
        d = json.loads(match("turn it down by 20")[1])
        assert d["action"] == "down" and d["level"] == 20

    def test_code_request_reaches_create_code_file(self):
        from app.assistant.intents import tool_subset_for
        s = tool_subset_for("make a snake game in python")
        assert "create_code_file" in s and "write_file" in s


class TestSTTFilters:
    def test_decoder_loop_rejected(self):
        from app.voice.stt import _is_hallucination, _keep_segment
        junk = "we will use this link to find all of this link " * 12
        assert _is_hallucination(junk)
        assert not _keep_segment({"no_speech_prob": 0.1, "avg_logprob": -0.3,
                                  "compression_ratio": 8.0, "text": junk})
        assert _keep_segment({"no_speech_prob": 0.1, "avg_logprob": -0.3,
                              "compression_ratio": 1.4, "text": "set a timer"})
