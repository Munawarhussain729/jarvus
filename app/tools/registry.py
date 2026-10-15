"""Central allowlisted tool registry — the only path to execution."""

from __future__ import annotations

from typing import Any

from app.security.sanitize import InputRejected, sanitize_tool_argument_string
from app.security.policy import RiskLevel, is_tool_name_allowed
from app.tools.base import Tool, ToolResult, ToolSpec
from app.tools.safe_shell import SafeShellTool


class ToolRegistry:
    """Maps tool name -> Tool instance; builds OpenAI tool definitions."""

    def __init__(self, tools: list[Tool]) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools:
            name = t.spec.name
            if name in self._tools:
                raise ValueError(f"Duplicate tool name: {name}")
            self._tools[name] = t
        self._tools_payload = self._build_payload()

    def _build_payload(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters_schema,
                },
            }
            for spec in (t.spec for t in self._tools.values())
        ]

    @classmethod
    def default(cls) -> ToolRegistry:
        return cls([EchoTool(), SafeShellTool()])

    def allowlist(self) -> frozenset[str]:
        return frozenset(self._tools.keys())

    def get(self, name: str) -> Tool | None:
        if not is_tool_name_allowed(name, self.allowlist()):
            return None
        return self._tools.get(name)

    def openai_tools_payload(self) -> list[dict[str, Any]]:
        # Cached at construction (registry is static). Callers must not mutate
        # the returned dicts - build a fresh list to subset.
        return list(self._tools_payload)

    def tools_payload_for(self, names: set[str] | None) -> list[dict[str, Any]]:
        """Subset of the cached payload (fresh list) or the full payload."""
        if not names:
            return list(self._tools_payload)
        return [t for t in self._tools_payload if t["function"]["name"] in names]


class EchoTool(Tool):
    """Low-risk demo tool: echoes a string (validates length)."""

    spec = ToolSpec(
        name="echo",
        description="Echo back a short user-provided message (for testing tools).",
        parameters_schema={
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Short text to echo.",
                    "maxLength": 2000,
                }
            },
            "required": ["message"],
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        msg = arguments.get("message")
        if not isinstance(msg, str):
            return ToolResult(ok=False, payload="message must be a string", detail={})
        try:
            msg = sanitize_tool_argument_string(msg, field_name="message")
        except InputRejected as e:
            return ToolResult(ok=False, payload=str(e), detail={})
        return ToolResult(ok=True, payload=msg, detail={})
