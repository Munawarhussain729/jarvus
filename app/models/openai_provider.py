"""OpenAI Chat Completions provider (network only here — no listeners)."""

# TODO: Add a local ModelProvider (e.g. MLX, llama.cpp) using the same AssistantTurn contract.

from __future__ import annotations

from typing import Any

from openai import OpenAI

from app.models.base import AssistantTurn, ModelProvider, ToolCall


class OpenAIProvider(ModelProvider):
    def __init__(self, *, api_key: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        on_delta=None,
        max_tokens: int | None = None,
    ) -> AssistantTurn:
        api_messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        api_messages.extend(messages)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message
        content = msg.content
        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments_json=tc.function.arguments or "{}",
                    )
                )
        return AssistantTurn(content=content, tool_calls=tool_calls)
