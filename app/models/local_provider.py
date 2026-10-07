"""Local model provider - uses Ollama or llama.cpp HTTP if available."""
from __future__ import annotations

from typing import Any

from app.models.base import AssistantTurn, ModelProvider, ToolCall


class LocalProvider(ModelProvider):
    def __init__(self, *, base_url: str = "http://localhost:11434/v1", model: str = "qwen3:4b", api_key: str = "ollama", temperature: float = 0.1, max_tokens: int = 256) -> None:
        try:
            from openai import OpenAI
        except Exception:
            raise RuntimeError("openai required for local provider via OpenAI-compatible API")
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> AssistantTurn:
        api_messages = [{"role": "system", "content": system_prompt}] + messages
        kwargs = {
            "model": self._model,
            "messages": api_messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "top_p": 0.9,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        try:
            resp = self._client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            msg = choice.message
            tool_calls = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments_json=tc.function.arguments or "{}"))
            return AssistantTurn(content=msg.content, tool_calls=tool_calls)
        except Exception as e:
            return AssistantTurn(content=f"[local model error] {e}", tool_calls=[])
