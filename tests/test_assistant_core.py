"""Assistant loop with a fake model provider (no API key)."""

from __future__ import annotations

from typing import Any

import pytest

from app.assistant.core import AssistantSession
from app.assistant.router import ToolRouter
from app.audit.logger import AuditLogger
from app.memory.store import MemoryStore
from app.models.base import AssistantTurn, ModelProvider, ToolCall
from app.tools.registry import ToolRegistry


class FakeProvider(ModelProvider):
    def __init__(self, turns: list[AssistantTurn]) -> None:
        self._turns = list(turns)
        self.calls = 0

    def generate(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> AssistantTurn:
        if self.calls >= len(self._turns):
            return AssistantTurn(content="done", tool_calls=[])
        t = self._turns[self.calls]
        self.calls += 1
        return t


@pytest.fixture
def tmp_db(tmp_path):
    p = tmp_path / "x.db"
    store = MemoryStore(p)
    yield store
    store.close()


def test_process_user_text_echo_tool(tmp_db: MemoryStore) -> None:
    registry = ToolRegistry.default()
    audit = AuditLogger(tmp_db)
    router = ToolRouter(registry, audit)
    provider = FakeProvider(
        [
            AssistantTurn(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="echo",
                        arguments_json='{"message": "hi"}',
                    )
                ],
            ),
            AssistantTurn(content="Thanks.", tool_calls=[]),
        ]
    )
    session = AssistantSession(
        provider=provider,
        registry=registry,
        router=router,
        memory=tmp_db,
    )
    reply = session.process_user_text("ping")
    assert "Thanks" in reply
