"""Model provider abstraction — swap OpenAI for local inference later."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolCall:
    id: str
    name: str
    arguments_json: str


@dataclass
class AssistantTurn:
    """One model response: assistant text and/or structured tool calls."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class ModelProvider(ABC):
    @abstractmethod
    def generate(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        on_delta: Callable[[str], None] | None = None,
    ) -> AssistantTurn:
        """Call remote or local model; must not execute tools.

        on_delta (optional): receives content text chunks as they stream in,
        so voice mode can start speaking the first sentence while the rest
        generates. Providers without streaming support may ignore it.
        """
