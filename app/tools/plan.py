"""Planning tools for complex task management — persisted to SQLite."""
from __future__ import annotations

from typing import Any

from app.security.policy import RiskLevel
from app.security.sanitize import sanitize_tool_argument_string
from app.tools.base import Tool, ToolResult, ToolSpec


class PlanTool(Tool):
    spec = ToolSpec(
        name="plan",
        description="Create a named multi-step plan for complex work (persisted).",
        parameters_schema={
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
                "steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ordered steps",
                },
            },
            "required": ["goal", "steps"],
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def __init__(self, memory=None) -> None:
        self._memory = memory

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            goal = sanitize_tool_argument_string(arguments["goal"], field_name="goal")
            steps = arguments["steps"]
            if not isinstance(steps, list):
                raise ValueError("steps must be array")
            lines = [f"Goal: {goal}", ""]
            for i, s in enumerate(steps, 1):
                lines.append(f"{i}. {str(s)[:200]}")
            payload = "\n".join(lines)
            if self._memory is not None:
                self._memory.save_plan(goal, payload)
            return ToolResult(ok=True, payload=payload)
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))


class TodoWriteTool(Tool):
    spec = ToolSpec(
        name="todo_write",
        description="Manage the task list (todos) - persisted across sessions. "
                    "'set' replaces the whole list; 'add' adds one item; "
                    "'complete' marks items done; 'list' shows current tasks.",
        parameters_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["set", "add", "complete", "list"], "default": "set"},
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "cancelled"]},
                            "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                        },
                        "required": ["content", "status", "priority"],
                        "additionalProperties": False,
                    },
                },
                "content": {"type": "string", "description": "Todo text (add) or text to match (complete)"},
                "priority": {"type": "string", "enum": ["high", "medium", "low"], "default": "medium"},
            },
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def __init__(self, memory=None) -> None:
        self._memory = memory

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        action = arguments.get("action") or "set"
        if self._memory is None:
            return self._render(arguments.get("todos", []))

        try:
            if action == "list":
                return self._render(self._memory.list_todos(include_done=True))
            if action == "add":
                content = sanitize_tool_argument_string(str(arguments.get("content") or ""), field_name="content").strip()
                if not content:
                    return ToolResult(ok=False, payload="content required for add")
                self._memory.add_todo(content, "pending", arguments.get("priority") or "medium")
                return ToolResult(ok=True, payload=f"Todo added: {content}")
            if action == "complete":
                content = sanitize_tool_argument_string(str(arguments.get("content") or ""), field_name="content").strip()
                if not content:
                    return ToolResult(ok=False, payload="content required for complete")
                n = self._memory.complete_todo(content)
                if n:
                    return ToolResult(ok=True, payload=f"Completed {n} todo(s) matching '{content}'.")
                return ToolResult(ok=True, payload=f"No open todos matching '{content}'.")
            todos = arguments.get("todos", [])
            clean = [
                {"content": str(t["content"])[:300],
                 "status": str(t.get("status") or "pending"),
                 "priority": str(t.get("priority") or "medium")}
                for t in todos[:50] if isinstance(t, dict) and t.get("content")
            ]
            self._memory.set_todos(clean)
            return self._render(clean)
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))

    @staticmethod
    def _render(todos: list[dict[str, Any]]) -> ToolResult:
        if not todos:
            return ToolResult(ok=True, payload="Todo list is empty.")
        lines = ["Todos:"]
        for t in todos[:50]:
            lines.append(f"- [{t['status']}] ({t['priority']}) {t['content']}")
        return ToolResult(ok=True, payload="\n".join(lines))
