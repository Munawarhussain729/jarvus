"""Memory tools - let Jarvus remember and recall facts about the user."""
from __future__ import annotations

from typing import Any

from app.memory.enhanced import EnhancedMemory
from app.security.policy import RiskLevel
from app.security.sanitize import sanitize_tool_argument_string
from app.tools.base import Tool, ToolResult, ToolSpec


def _key(value: Any, field: str = "key") -> str:
    key = sanitize_tool_argument_string(str(value or ""), field_name=field).strip().lower()
    if not key:
        raise ValueError(f"{field} required")
    return key


class RememberTool(Tool):
    def __init__(self, memory: EnhancedMemory) -> None:
        self._memory = memory

    spec = ToolSpec(
        name="remember",
        description="Store a fact about the user long-term (name, preference, birthday, project...). "
                    "Use proactively when the user states personal info worth remembering.",
        parameters_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Short fact label, e.g. 'name', 'favorite_color'"},
                "value": {"type": "string", "description": "The fact to store"},
            },
            "required": ["key", "value"],
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            key = _key(arguments.get("key"))
            value = sanitize_tool_argument_string(str(arguments.get("value", "")), field_name="value")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        self._memory.set_fact(key, value)
        return ToolResult(ok=True, payload=f"Remembered: {key} = {value}")


class RecallTool(Tool):
    def __init__(self, memory: EnhancedMemory) -> None:
        self._memory = memory

    spec = ToolSpec(
        name="recall",
        description="Recall a stored fact by key, or list all known facts if key omitted.",
        parameters_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Fact key to recall (optional)"},
            },
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        key = (arguments.get("key") or "").strip().lower()
        if not key:
            facts = self._memory.list_facts()
            if not facts:
                return ToolResult(ok=True, payload="No facts stored yet.")
            return ToolResult(ok=True, payload="\n".join(f"{k}: {v}" for k, v in facts))
        value = self._memory.get_fact(key)
        if value is None:
            return ToolResult(ok=True, payload=f"Nothing stored for '{key}'.")
        return ToolResult(ok=True, payload=f"{key}: {value}")


class ForgetTool(Tool):
    def __init__(self, memory: EnhancedMemory) -> None:
        self._memory = memory

    spec = ToolSpec(
        name="forget",
        description="Delete a stored fact by key.",
        parameters_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string"},
            },
            "required": ["key"],
            "additionalProperties": False,
        },
        risk=RiskLevel.MEDIUM,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            key = _key(arguments.get("key"))
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        if self._memory.delete_fact(key):
            return ToolResult(ok=True, payload=f"Forgot: {key}")
        return ToolResult(ok=False, payload=f"No fact stored for '{key}'.")
