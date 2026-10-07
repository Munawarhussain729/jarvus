"""Assistant orchestration: model (reasoning) vs router (execution)."""

from __future__ import annotations

from typing import Any

from app.assistant.router import ToolRouter
from app.memory.store import MemoryStore
from app.models.base import ModelProvider
from app.security.sanitize import InputRejected, sanitize_user_message
from app.tools.registry import ToolRegistry


SYSTEM_PROMPT = """You are Typhoon, a local-first assistant running on the user's Mac.

Security rules you must follow:
- Never tell the user to run shell commands you invent. Actions go only through declared tools.
- Prefer concise, accurate answers. If a tool is needed, call it with valid arguments only.

Available capabilities are exposed as function tools; you cannot execute anything outside them.
"""


class AssistantSession:
    """One REPL session: in-memory message list + optional persistence."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        registry: ToolRegistry,
        router: ToolRouter,
        memory: MemoryStore,
        max_tool_rounds: int = 6,
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._router = router
        self._memory = memory
        self._max_tool_rounds = max_tool_rounds
        self._messages: list[dict[str, Any]] = []

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self._messages

    def process_user_text(self, text: str) -> str:
        try:
            cleaned = sanitize_user_message(text)
        except InputRejected as e:
            return f"[input rejected] {e}"

        if not cleaned:
            return "[input rejected] Empty message."

        self._memory.append_message("user", cleaned)
        self._messages.append({"role": "user", "content": cleaned})

        tools = self._registry.openai_tools_payload()
        rounds = 0
        while rounds < self._max_tool_rounds:
            rounds += 1
            turn = self._provider.generate(
                system_prompt=SYSTEM_PROMPT,
                messages=self._messages,
                tools=tools,
            )

            if turn.tool_calls:
                # Persist assistant tool-call request as a compact line (MVP)
                self._memory.append_message(
                    "assistant",
                    turn.content or f"(tool_calls: {len(turn.tool_calls)})",
                )
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": turn.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments_json},
                        }
                        for tc in turn.tool_calls
                    ],
                }
                self._messages.append(assistant_msg)

                for tc in turn.tool_calls:
                    body = self._router.handle_tool_call(
                        tool_name=tc.name,
                        arguments_json=tc.arguments_json,
                    )
                    self._messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": body,
                        }
                    )
                continue

            reply = (turn.content or "").strip() or "(no text reply)"
            self._memory.append_message("assistant", reply)
            self._messages.append({"role": "assistant", "content": reply})
            return reply

        return "[Typhoon] Stopped: maximum tool rounds exceeded."
