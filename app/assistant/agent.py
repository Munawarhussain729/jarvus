"""Full agent with extended capabilities."""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any

from app.assistant.core import AssistantSession, SYSTEM_PROMPT
from app.assistant.router import ToolRouter
from app.memory.enhanced import EnhancedMemory
from app.memory.store import MemoryStore
from app.models.base import ModelProvider


AGENT_SYSTEM_PROMPT = """You are Jarvus, an autonomous local AI agent running 100% locally with strict security. Think of yourself as JARVIS from Iron Man - helpful, capable, proactive, concise.

CAPABILITIES (call tools with valid arguments only):
- open_app / open_url / say / volume / send_notification / read_notifications / screenshot / clipboard / frontmost_app: macOS control & notifications
- dark_mode / lock_screen / sleep_display / run_shortcut / empty_trash: more macOS control
- read_file / write_file / list_dir / glob_files / find_files: file operations & Spotlight search (sandboxed)
- run_command (HIGH risk) / shell_read / python_exec / create_code_file: execution & writing programs/games to ~/Jarvus/projects
- plan / todo_write: planning & persistent task list (set/add/complete/list)
- system_info / ps_list / clock / date_calc / battery / network_info: system, time & date, network
- weather / timer / alarm / reminder / calendar: everyday tasks
- list_reminders / calendar_read / daily_brief: see and analyse today's tasks
- music / message / email: apps and communication
- web_search / web_fetch / wikipedia / news / currency / crypto / stock / joke: internet & live facts
- calculate / unit_convert: math and unit conversion - ALWAYS use these for arithmetic or conversions, never do them mentally
- random / text_tools: coin flips, dice, passwords, uuid, base64/hash/case transforms
- scratchpad: the user's own notes file (NOT system notifications)
- remember / recall / forget: long-term memory of user facts
- devin_projects / devin_project / devin_send: list & register Devin projects, and send prompts to Devin sessions (local CLI or cloud)

PRINCIPLES:
- CRITICAL: when the user asks you to DO something a tool covers, you MUST emit the tool call in this turn. Never reply with 'let me', 'I will', 'I can' or describe the action without calling the tool.
- Take action immediately when asked - use tools, don't just talk.
- Keep replies SHORT and spoken-friendly (this is often a voice interface).
- Voice input may be imperfect. If the request seems garbled, ask a one-line clarifying question instead of guessing.
- NEVER pick a tool just because its name sounds similar - 'scratchpad' is the user's notes file, 'read_notifications' reads system notifications, 'send_notification' only SENDS a banner.
- 'Make / write / create' a program, game, or script means call create_code_file with the filename and full code - it saves to ~/Jarvus/projects and syntax-checks in one call.
- If no tool fits the request, say what you can't do in one short sentence and ask a clarifying question.
- Use 'remember' proactively when the user shares personal facts, preferences, or context.
- HIGH risk tools require user confirmation - mention it briefly if cancelled.
""" + "\n\n" + SYSTEM_PROMPT

MAX_CONTEXT_MESSAGES = 40
# Hard prompt budget (~6k tokens of role content) - a small num_ctx can't hold
# 40 fat messages; the model would start answering its own truncated history.
MAX_CONTEXT_CHARS = 24000
# Cap tool results fed back to the model: keeps the small local model's context
# lean (faster inference) and stops one huge payload from evicting real history.
MAX_TOOL_RESULT_CHARS = 4000
_CODE_MAX_TOKENS = 1024  # bigger output budget for code-creation turns
# Requests that need the bigger generation budget.
_CODE_REQUEST_RE = re.compile(
    r"\b(?:make|write|create|build|code|generate|fix)\b.*"
    r"\b(?:game|script|program|code|app|python|bot)\b", re.IGNORECASE)

log = logging.getLogger("jarvus.agent")

# Background auto-learning: after each turn a small extraction pass pulls
# durable user facts into the facts table, so Jarvus personalises over time
# without the user saying "remember this".
_LEARN_SYSTEM_PROMPT = """You maintain a fact store about the user of a personal assistant.
From the exchange below, extract ONLY durable facts about the user worth
remembering across sessions: name, preferences, projects they work on, habits,
tools/settings they stated.
Rules:
- Output ONLY a JSON array like [{"key":"editor","value":"prefers vim"}] or [] when nothing durable exists.
- At most 3 facts. Keys are short snake_case slugs; values are one short phrase.
- Never store passwords, tokens, API keys, one-off requests, or assistant-internal details.
- Questions, commands, and small talk usually yield []."""


def parse_fact_candidates(text: str, max_facts: int = 3) -> list[tuple[str, str]]:
    """Parse the extractor's JSON array into (key, value) pairs, defensively."""
    m = re.search(r"\[.*\]", text or "", re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    out: list[tuple[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        key = re.sub(r"[^a-z0-9_.-]", "_", str(item.get("key", "")).lower().strip())[:40].strip("_.-")
        value = str(item.get("value", "")).strip()[:300]
        if not key or not value:
            continue
        # Never persist anything that looks like a credential.
        if any(s in key for s in ("password", "token", "secret", "api_key", "apikey")):
            continue
        out.append((key, value))
        if len(out) >= max_facts:
            break
    return out


class JarvusAgent(AssistantSession):
    def __init__(self, *args, tool_prefilter: bool = True, auto_learn: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._system_prompt = AGENT_SYSTEM_PROMPT
        self._tool_prefilter = tool_prefilter
        self._auto_learn = auto_learn

    def _learn_from_turn(self, user_text: str, reply: str) -> None:
        """Background pass: pull durable user facts out of the exchange."""
        if not isinstance(self._memory, EnhancedMemory):
            return
        try:
            turn = self._provider.generate(
                system_prompt=_LEARN_SYSTEM_PROMPT,
                messages=[{"role": "user",
                           "content": f"User: {user_text}\nAssistant: {reply}"}],
                tools=None,
                max_tokens=200,
            )
            for key, value in parse_fact_candidates(turn.content or ""):
                self._memory.set_fact(key, value)
                log.info("auto-learned fact: %s", key)
        except Exception as e:
            log.debug("auto-learn extraction failed: %s", e)

    def _prompt_with_facts(self) -> str:
        if isinstance(self._memory, EnhancedMemory):
            block = self._memory.facts_prompt_block()
            if block:
                return f"{self._system_prompt}\n\n{block}"
        return self._system_prompt

    def _trim_context(self) -> None:
        """Keep recent messages within a message-count AND a char (~token)
        budget, without orphaning tool results."""
        while len(self._messages) > MAX_CONTEXT_MESSAGES:
            self._messages.pop(0)
        while (sum(len(m.get("content") or "") for m in self._messages) > MAX_CONTEXT_CHARS
               and len(self._messages) > 8):
            self._messages.pop(0)
        while self._messages and (
            self._messages[0].get("role") == "tool"
            or (self._messages[0].get("role") == "assistant" and self._messages[0].get("tool_calls"))
        ):
            self._messages.pop(0)

    _TOOL_BODY_KEEP_RECENT = 4  # most recent tool messages stay verbatim

    def _compact_tool_history(self, before_idx: int) -> None:
        """Elide bodies of old 'tool' role messages (keep the status prefix).

        Old tool outputs are the biggest context growth driver - a web page or
        dir listing fetched 10 turns ago still rides along in every prompt.
        Only messages from previous turns (< before_idx) are touched; the
        audit log keeps every output in full.
        """
        tool_idx = [i for i, m in enumerate(self._messages)
                    if m.get("role") == "tool" and i < before_idx]
        for i in tool_idx[:-self._TOOL_BODY_KEEP_RECENT]:
            content = self._messages[i].get("content") or ""
            if len(content) > 200:
                prefix = content.split("]", 1)[0] + "]" if content.startswith("[tool") else ""
                self._messages[i]["content"] = f"{prefix} (earlier output elided)".strip()

    def process_user_text(self, text: str, *, on_delta=None, speech=None) -> str:
        """speech: optional _SpeechPipeline - aborts per-round pending text when
        a round ends in tool calls so half-sentences never get spoken."""
        """on_delta: optional callback receiving reply text chunks as they
        stream (voice mode uses it to speak the first sentence early)."""
        from app.security.sanitize import sanitize_user_message, InputRejected

        try:
            cleaned = sanitize_user_message(text)
        except InputRejected as e:
            return f"[input rejected] {e}"
        if not cleaned:
            return "[input rejected] Empty"

        self._memory.append_message("user", cleaned)
        self._messages.append({"role": "user", "content": cleaned})

        # Deterministic fast path for common commands - instant and reliable,
        # still routed through ToolRouter so policy/confirmation/audit apply.
        from app.assistant.intents import match as match_intent
        hit = match_intent(cleaned)
        if hit:
            tool_name, args_json = hit
            body = self._router.handle_tool_call(tool_name=tool_name, arguments_json=args_json)
            if len(body) > MAX_TOOL_RESULT_CHARS:
                body = body[:MAX_TOOL_RESULT_CHARS] + "\n... (truncated)"
            reply = re.sub(r"^\[tool (ok|error|cancelled)\]\s*", "", body)
            self._memory.append_message("assistant", reply)
            self._messages.append({"role": "assistant", "content": reply})
            return reply

        self._trim_context()
        turn_start = len(self._messages)
        self._compact_tool_history(turn_start)
        # Heuristic prefilter: shrink the schema payload for the small local
        # model. Union-matching + always-on core; expanded to the full set once
        # a tool round has run (results may need a tool outside the subset).
        from app.assistant.intents import tool_subset_for
        subset = tool_subset_for(cleaned) if self._tool_prefilter else None
        tools = self._registry.tools_payload_for(subset)
        subset_applied = subset is not None
        system_prompt = self._prompt_with_facts()  # once per turn, not per round
        nudged = False
        rounds = 0
        tool_ms = 0
        had_tool_error = False
        cancelled = False
        t0 = time.monotonic()
        logged = False
        # Same-turn dedupe: a repair retry can't re-fire a call that already
        # succeeded (e.g. write_file again while fixing another tool's args).
        executed: dict[tuple[str, str], str] = {}
        # Code-creation turns need a bigger generation budget - 384 tokens
        # truncates most real programs.
        wants_code = bool(_CODE_REQUEST_RE.search(cleaned)) or (
            subset is not None and "create_code_file" in subset)
        gen_budget = _CODE_MAX_TOKENS if wants_code else None
        while rounds < self._max_tool_rounds:
            rounds += 1
            if rounds > 1 and subset_applied and (had_tool_error or nudged):
                # Expand the schema set only when the subset actually failed.
                tools = self._registry.tools_payload_for(None)
                subset_applied = False
            self._compact_tool_history(turn_start)
            turn = self._provider.generate(
                system_prompt=system_prompt,
                messages=self._messages,
                tools=tools,
                on_delta=on_delta,
                max_tokens=gen_budget,
            )
            if not logged:
                logged = True
                # Feeds the Phase 4 decision: only a FIRST call of exactly one
                # no-arg tool counts as a System One candidate.
                single_noarg = (
                    len(turn.tool_calls or []) == 1
                    and (turn.tool_calls[0].arguments_json or "").strip() in ("", "{}")
                )
                log.info("llm_turn regex_hit=False latency_ms=%d resolved_single_noarg=%s tools=%d",
                         int((time.monotonic() - t0) * 1000), single_noarg, len(tools))
            if turn.tool_calls:
                if speech is not None:
                    speech.discard_pending()  # don't speak half-sentences of preamble
                self._memory.append_message("assistant", turn.content or f"(tools: {len(turn.tool_calls)})")
                self._messages.append({
                    "role": "assistant",
                    "content": turn.content,
                    "tool_calls": [
                        {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments_json}}
                        for tc in turn.tool_calls
                    ],
                })
                for tc in turn.tool_calls:
                    key = (tc.name, tc.arguments_json or "")
                    if key in executed:
                        body = executed[key]
                        log.info("tool_dedupe skipped repeat of %s", tc.name)
                    else:
                        tt = time.monotonic()
                        body = self._router.handle_tool_call(
                            tool_name=tc.name, arguments_json=tc.arguments_json)
                        tool_ms += int((time.monotonic() - tt) * 1000)
                        if body.startswith("[tool ok]"):
                            executed[key] = body
                        if "[tool error]" in body:
                            had_tool_error = True
                        if "[tool cancelled]" in body:
                            cancelled = True
                    if len(body) > MAX_TOOL_RESULT_CHARS:
                        body = body[:MAX_TOOL_RESULT_CHARS] + "\n... (truncated)"
                    self._messages.append({"role": "tool", "tool_call_id": tc.id, "content": body})
                self._trim_context()
                continue
            reply = (turn.content or "").strip()
            # Small local models sometimes return empty content or promise an
            # action instead of calling the tool. Nudge once so the turn
            # actually produces something; for action verbs the nudge names
            # the candidate tools so round 2 lands the call.
            if not nudged and (not reply or _looks_like_unfulfilled_action(reply)):
                nudged = True
                if reply:
                    self._messages.append({"role": "assistant", "content": reply})
                nudge = "Call the appropriate tool now, or give a short spoken reply - never respond with empty text."
                if _ACTION_RE.search(cleaned):
                    names = [t["function"]["name"] for t in tools][:10]
                    nudge = (f"That request needs a tool call - emit one of [{', '.join(names)}] "
                             "with valid JSON arguments right now, not a description.")
                self._messages.append({"role": "user", "content": nudge})
                continue
            if not reply:
                log.warning("llm_turn produced empty content")
                reply = "Sorry, I didn't catch that - could you say it again?"
            outcome = ("cancelled" if cancelled else
                       "error" if had_tool_error else
                       "clarify" if reply.strip().endswith("?") else "ok")
            log.info("turn_done rounds=%d tool_ms=%d wall_ms=%d outcome=%s",
                     rounds, tool_ms, int((time.monotonic() - t0) * 1000), outcome)
            try:
                from app.telemetry import record
                record("turn_done", rounds=rounds, tool_ms=tool_ms,
                       wall_ms=int((time.monotonic() - t0) * 1000), outcome=outcome)
            except Exception:
                pass
            self._memory.append_message("assistant", reply)
            self._messages.append({"role": "assistant", "content": reply})
            if self._auto_learn:
                # Off the critical path: extraction runs while the reply is
                # already being spoken/printed. Never blocks the turn.
                threading.Thread(target=self._learn_from_turn,
                                 args=(cleaned, reply), daemon=True).start()
            return reply
        return "[Jarvus] Max rounds exceeded"


_ACTION_RE = re.compile(
    r"\b(make|write|create|build|open|launch|set|send|play|start|stop|turn|"
    r"delete|remind|add|run|schedule|generate|code|take|capture)\b", re.IGNORECASE)


_PROMISE_PATTERNS = ("let me ", "i'll ", "i will ", "i can help you", "i can check",
                   "going to ", "allow me", "i'll check", "i'm going to",
                   "one moment", "please wait", "working on it", "give me a second",
                   "i'm checking", "let me check")


def _looks_like_unfulfilled_action(reply: str) -> bool:
    low = reply.lower()
    return any(p in low for p in _PROMISE_PATTERNS)
