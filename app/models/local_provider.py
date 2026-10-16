"""Local model provider - Ollama native API with OpenAI-compatible fallback.

Works against:
- Ollama (native /api/chat): uses think=false, num_ctx, keep_alive. Recommended.
- Any OpenAI-compatible endpoint (llama.cpp server, LM Studio, vLLM): /v1/chat/completions.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable

from app.models.base import AssistantTurn, ModelProvider, ToolCall

log = logging.getLogger("jarvus.provider.local")

# How long a single LLM request may take before we give up. Voice turns must
# fail fast - a silent 3-minute hang feels like Jarvus died.
REQUEST_TIMEOUT = 45.0

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)
# A thinking variant answering with its monologue instead of the reply -
# matches the start of typical reasoning prose, never a normal answer.
_REASONING_LEAK_RE = re.compile(
    r"^(?:hmm|okay|ok|alright)[,.]?\s+(?:the user|let me|i need|i should|so)\b"
    r"|^(?:the user|let me (?:think|analyze|figure|check|see|work)|i need to|"
    r"i should (?:first|probably)|first,? (?:i|let)|wait,? (?:let|i|the)|"
    r"to (?:answer|respond|do this|help)|analy[sz]ing|let'?s (?:see|think|analyze))\b",
    re.IGNORECASE,
)


def clean_content(content: str | None) -> str | None:
    """Strip <think> blocks; return '' if what remains is reasoning monologue.

    Thinking-style models occasionally put chain-of-thought in `content` even
    with think=false. Returning '' lets the agent's empty-reply nudge retry.
    """
    if content is None:
        return None
    text = _THINK_BLOCK_RE.sub("", content)
    if _UNCLOSED_THINK_RE.search(text):
        # '<think> ...' with no closing tag: everything after is monologue.
        text = _UNCLOSED_THINK_RE.sub("", text)
    text = text.strip()
    if text and _REASONING_LEAK_RE.match(text):
        return ""
    return text


def _json_or_original(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value if value is not None else {}


def _collect_tool_calls(msg: dict[str, Any]) -> list[ToolCall]:
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
    return tool_calls


class LocalProvider(ModelProvider):
    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:4b-instruct-2507",
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
            self._http = httpx.Client(timeout=REQUEST_TIMEOUT)
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

    def warmup(self, *, keep_alive: str | None = None) -> None:
        """Load the model and log speed + reasoning-leak status. Safe to call
        on a background thread at startup."""
        if keep_alive:
            self._keep_alive = keep_alive
        if not self._detect_backend():
            return  # OpenAI-compatible backend: nothing to warm
        import httpx
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": "Reply with exactly: ready"}],
            "stream": False,
            "think": False,
            "keep_alive": self._keep_alive,
            "options": {"temperature": 0.0, "num_predict": 8, "num_ctx": 2048},
        }
        try:
            t0 = time.monotonic()
            r = self._client().post(f"{self._base}/api/chat", json=payload)
            r.raise_for_status()
            d = r.json()
            wall = time.monotonic() - t0
            eval_n, eval_d = d.get("eval_count") or 0, d.get("eval_duration") or 0
            tps = eval_n / (eval_d / 1e9) if eval_d else 0.0
            raw = (d.get("message") or {}).get("content") or ""
            leaked = bool(_REASONING_LEAK_RE.match(raw.strip()) or "<think>" in raw.lower())
            log.info("LLM warmup: %s ready in %.1fs (%.0f tok/s)%s",
                     self._model, wall, tps,
                     " - WARNING: model leaked reasoning; use an instruct variant" if leaked else "")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                log.warning("LLM warmup: model '%s' not pulled - run: ollama pull %s",
                            self._model, self._model)
            else:
                log.warning("LLM warmup failed: %s", e)
        except Exception as e:
            log.warning("LLM warmup failed: %s", e)

    def generate(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        on_delta: Callable[[str], None] | None = None,
    ) -> AssistantTurn:
        api_messages = [{"role": "system", "content": system_prompt}] + messages
        if self._detect_backend():
            return self._generate_ollama(self._to_native_history(api_messages), tools, on_delta=on_delta)
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

    _TIMEOUT_REPLY = "[Jarvus] That's taking too long - the model timed out. Try a shorter request."

    def _generate_ollama(self, api_messages: list[dict[str, Any]], tools,
                         *, on_delta: Callable[[str], None] | None = None) -> AssistantTurn:
        import httpx
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "stream": bool(on_delta),
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
        last_err: Exception | None = None
        for attempt in range(2):
            try:
                if on_delta is not None:
                    return self._ollama_stream(payload, on_delta)
                r = self._client().post(f"{self._base}/api/chat", json=payload)
                r.raise_for_status()
                msg = r.json().get("message", {})
                return AssistantTurn(content=clean_content(msg.get("content")),
                                     tool_calls=_collect_tool_calls(msg))
            except httpx.TimeoutException as e:
                # No retry: a second 45s wait just doubles the silence.
                log.warning("LLM request timed out after %.0fs", REQUEST_TIMEOUT)
                return AssistantTurn(content=self._TIMEOUT_REPLY, tool_calls=[])
            except Exception as e:
                last_err = e
                if attempt == 0:
                    time.sleep(0.4)  # one quick retry on transient errors
        return AssistantTurn(content=f"[local model error] {last_err}", tool_calls=[])

    def _ollama_stream(self, payload: dict[str, Any],
                       on_delta: Callable[[str], None]) -> AssistantTurn:
        """Streaming /api/chat: speakable text flows to on_delta per chunk while
        tool_calls still land in the final message."""
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        with self._client().stream("POST", f"{self._base}/api/chat", json=payload) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = d.get("message") or {}
                delta = msg.get("content") or ""
                if delta:
                    content_parts.append(delta)
                    on_delta(delta)
                if msg.get("tool_calls"):
                    tool_calls = _collect_tool_calls(msg)
        return AssistantTurn(content=clean_content("".join(content_parts)),
                             tool_calls=tool_calls)

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
        last_err: Exception | None = None
        for attempt in range(2):
            try:
                resp = client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message
                tool_calls = []
                for tc in msg.tool_calls or []:
                    tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments_json=tc.function.arguments or "{}"))
                return AssistantTurn(content=clean_content(msg.content), tool_calls=tool_calls)
            except Exception as e:
                last_err = e
                if attempt == 0:
                    time.sleep(0.4)  # one quick retry on transient errors
        return AssistantTurn(content=f"[local model error] {last_err}", tool_calls=[])
