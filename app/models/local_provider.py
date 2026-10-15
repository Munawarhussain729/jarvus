"""Local model provider - Ollama native API with OpenAI-compatible fallback.

Works against:
- Ollama (native /api/chat): uses think=false, num_ctx, keep_alive. Recommended.
- Any OpenAI-compatible endpoint (llama.cpp server, LM Studio, vLLM): /v1/chat/completions.
"""
from __future__ import annotations

import json
from typing import Any

from app.models.base import AssistantTurn, ModelProvider, ToolCall


def _json_or_original(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value if value is not None else {}


class LocalProvider(ModelProvider):
    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:8b",
        temperature: float = 0.2,
        max_tokens: int = 512,
        num_ctx: int = 16384,
        think: bool = False,
        keep_alive: str = "30m",
    ) -> None:
        self._base = base_url.rstrip("/")
        # Normalise: accept either the Ollama root or a /v1 compat URL
        if self._base.endswith("/v1"):
            self._base = self._base[: -len("/v1")]
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._num_ctx = num_ctx
        self._think = think
        self._keep_alive = keep_alive
        self._ollama_native: bool | None = None  # detected lazily
        self._http = None  # lazily-shared httpx.Client (keep-alive pooling)
        self._oai = None  # lazily-shared OpenAI compat client

    def _client(self):
        if self._http is None:
            import httpx
            self._http = httpx.Client(timeout=180.0)
        return self._http

    def _detect_backend(self) -> bool:
        if self._ollama_native is not None:
            return self._ollama_native
        try:
            r = self._client().get(f"{self._base}/api/version", timeout=2.0)
            self._ollama_native = r.status_code == 200
        except Exception:
            self._ollama_native = False
        return self._ollama_native

    def generate(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> AssistantTurn:
        api_messages = [{"role": "system", "content": system_prompt}] + messages
        if self._detect_backend():
            return self._generate_ollama(self._to_native_history(api_messages), tools)
        return self._generate_openai_compat(api_messages, tools)

    @staticmethod
    def _to_native_history(api_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Ollama's native /api/chat expects tool_call function arguments as objects,
        while our session stores them as JSON strings (OpenAI shape)."""
        out: list[dict[str, Any]] = []
        for msg in api_messages:
            if msg.get("tool_calls"):
                msg = dict(msg)
                msg["tool_calls"] = [
                    {
                        **tc,
                        "function": {
                            **tc.get("function", {}),
                            "arguments": _json_or_original(tc.get("function", {}).get("arguments")),
                        },
                    }
                    for tc in msg["tool_calls"]
                ]
            out.append(msg)
        return out

    def _generate_ollama(self, api_messages: list[dict[str, Any]], tools) -> AssistantTurn:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "stream": False,
            "think": self._think,
            "keep_alive": self._keep_alive,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
                "num_ctx": self._num_ctx,
                "top_p": 0.9,
            },
        }
        if tools:
            payload["tools"] = tools
        try:
            r = self._client().post(f"{self._base}/api/chat", json=payload)
            r.raise_for_status()
            msg = r.json().get("message", {})
            tool_calls = []
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                args = fn.get("arguments")
                if not isinstance(args, str):
                    args = json.dumps(args or {})
                tool_calls.append(ToolCall(
                    id=tc.get("id") or f"call_{len(tool_calls)}",
                    name=fn.get("name", ""),
                    arguments_json=args,
                ))
            return AssistantTurn(content=msg.get("content"), tool_calls=tool_calls)
        except Exception as e:
            return AssistantTurn(content=f"[local model error] {e}", tool_calls=[])

    def _generate_openai_compat(self, api_messages: list[dict[str, Any]], tools) -> AssistantTurn:
        try:
            from openai import OpenAI
        except Exception:
            return AssistantTurn(content="[local model error] openai package required for non-Ollama endpoints", tool_calls=[])
        if self._oai is None:
            self._oai = OpenAI(base_url=f"{self._base}/v1", api_key="local")
        client = self._oai
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        try:
            resp = client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            tool_calls = []
            for tc in msg.tool_calls or []:
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments_json=tc.function.arguments or "{}"))
            return AssistantTurn(content=msg.content, tool_calls=tool_calls)
        except Exception as e:
            return AssistantTurn(content=f"[local model error] {e}", tool_calls=[])
