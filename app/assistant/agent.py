"""Full agent with extended capabilities."""
from __future__ import annotations

import logging
import re
import time
from typing import Any

from app.assistant.core import AssistantSession, SYSTEM_PROMPT
from app.assistant.router import ToolRouter
from app.memory.enhanced import EnhancedMemory
from app.memory.store import MemoryStore
from app.models.base import ModelProvider


AGENT_SYSTEM_PROMPT = """You are Jarvus, an autonomous local AI agent running 100% locally with strict security. Think of yourself as JARVIS from Iron Man - helpful, capable, proactive, concise.

CAPABILITIES (call tools with valid arguments only):
- open_app / say / volume / send_notification / read_notifications / screenshot / clipboard / frontmost_app: macOS control & notifications
- read_file / write_file / list_dir / glob_files: file operations (sandboxed)
- run_command (HIGH risk) / shell_read / python_exec: execution
- plan / todo_write: planning & persistent task list (set/add/complete/list)
- system_info / ps_list / clock / battery: system info, time & date, battery
- weather / timer / alarm / reminder / calendar: everyday tasks
- list_reminders / calendar_read / daily_brief: see and analyse today's tasks
- music / message / email: apps and communication
- web_search / web_fetch: internet
- scratchpad: the user's own notes file (NOT system notifications)
- remember / recall / forget: long-term memory of user facts

PRINCIPLES:
- CRITICAL: when the user asks you to DO something a tool covers, you MUST emit the tool call in this turn. Never reply with 'let me', 'I will', 'I can' or describe the action without calling the tool.
- Take action immediately when asked - use tools, don't just talk.
- Keep replies SHORT and spoken-friendly (this is often a voice interface).
- Voice input may be imperfect. If the request seems garbled, ask a one-line clarifying question instead of guessing.
- NEVER pick a tool just because its name sounds similar - 'scratchpad' is the user's notes file, 'read_notifications' reads system notifications, 'send_notification' only SENDS a banner.
- If no tool fits the request, say what you can't do in one short sentence and ask a clarifying question.
- Use 'remember' proactively when the user shares personal facts, preferences, or context.
- HIGH risk tools require user confirmation - mention it briefly if cancelled.
""" + "\n\n" + SYSTEM_PROMPT

MAX_CONTEXT_MESSAGES = 40

log = logging.getLogger("jarvus.agent")


class JarvusAgent(AssistantSession):
    def __init__(self, *args, tool_prefilter: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._system_prompt = AGENT_SYSTEM_PROMPT
        self._tool_prefilter = tool_prefilter

    def _prompt_with_facts(self) -> str:
        if isinstance(self._memory, EnhancedMemory):
            block = self._memory.facts_prompt_block()
            if block:
                return f"{self._system_prompt}\n\n{block}"
        return self._system_prompt

    def _trim_context(self) -> None:
        """Keep the last MAX_CONTEXT_MESSAGES messages without orphaning tool results."""
        while len(self._messages) > MAX_CONTEXT_MESSAGES:
            self._messages.pop(0)
        while self._messages and (
            self._messages[0].get("role") == "tool"
            or (self._messages[0].get("role") == "assistant" and self._messages[0].get("tool_calls"))
        ):
            self._messages.pop(0)

    def process_user_text(self, text: str) -> str:
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
            reply = re.sub(r"^\[tool (ok|error|cancelled)\]\s*", "", body)
            self._memory.append_message("assistant", reply)
            self._messages.append({"role": "assistant", "content": reply})
            return reply

        self._trim_context()
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
        t0 = time.monotonic()
        logged = False
        while rounds < self._max_tool_rounds:
            rounds += 1
            if rounds > 1 and subset_applied:
                tools = self._registry.tools_payload_for(None)
                subset_applied = False
            turn = self._provider.generate(
                system_prompt=system_prompt,
                messages=self._messages,
                tools=tools,
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
                    body = self._router.handle_tool_call(tool_name=tc.name, arguments_json=tc.arguments_json)
                    self._messages.append({"role": "tool", "tool_call_id": tc.id, "content": body})
                self._trim_context()
                continue
            reply = (turn.content or "").strip() or "(no text)"
            # Small local models sometimes promise an action instead of calling
            # the tool. Nudge once so the call actually happens.
            if not nudged and _looks_like_unfulfilled_action(reply):
                nudged = True
                self._messages.append({"role": "assistant", "content": reply})
                self._messages.append({
                    "role": "user",
                    "content": "Call the appropriate tool now - do not just describe the action.",
                })
                continue
            self._memory.append_message("assistant", reply)
            self._messages.append({"role": "assistant", "content": reply})
            return reply
        return "[Jarvus] Max rounds exceeded"


_PROMISE_PATTERNS = ("let me ", "i'll ", "i will ", "i can help you", "i can check",
                   "going to ", "allow me", "i'll check", "i'm going to")


def _looks_like_unfulfilled_action(reply: str) -> bool:
    low = reply.lower()
    return any(p in low for p in _PROMISE_PATTERNS)
