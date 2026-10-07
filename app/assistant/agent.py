"""Full agent with extended capabilities."""
from __future__ import annotations

from typing import Any

from app.assistant.core import AssistantSession, SYSTEM_PROMPT
from app.assistant.router import ToolRouter
from app.memory.store import MemoryStore
from app.models.base import ModelProvider


AGENT_SYSTEM_PROMPT = """You are Jarvus/Typhoon, an autonomous local AI agent running 100% locally-first with strict security. Think of yourself as JARVIS from Iron Man - helpful, capable, and proactive.

CAPABILITIES:
- open_app: Open macOS applications (WhatsApp, Safari, Chrome, etc.)
- say: Speak text aloud
- read_file/write_file/list_dir/glob_files: File operations (sandboxed)
- run_command: Run controlled commands (HIGH risk - needs confirmation)
- shell_read: Read-only shell commands (including 'open' for apps)
- python_exec: Execute Python code
- plan/todo_write: Planning and task management
- system_info/ps_list: System information

PRINCIPLES:
- Be proactive and take action when asked. If user says "open X", use open_app tool immediately.
- For opening apps, prefer 'open_app' tool over shell commands.
- Think step by step. Use tools don't just talk.
- Security: HIGH risk tools require confirmation - be transparent about that.
- Local-first, privacy-focused.
- Concise but helpful.
""" + "\n\n" + SYSTEM_PROMPT


class JarvusAgent(AssistantSession):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._system_prompt = AGENT_SYSTEM_PROMPT

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
        tools = self._registry.openai_tools_payload()
        rounds = 0
        while rounds < self._max_tool_rounds:
            rounds += 1
            turn = self._provider.generate(
                system_prompt=self._system_prompt,
                messages=self._messages,
                tools=tools,
            )
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
                continue
            reply = (turn.content or "").strip() or "(no text)"
            self._memory.append_message("assistant", reply)
            self._messages.append({"role": "assistant", "content": reply})
            return reply
        return "[Jarvus] Max rounds exceeded"
