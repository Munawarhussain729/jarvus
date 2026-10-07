"""Planning tools for complex task management."""
from __future__ import annotations

from typing import Any

from app.security.policy import RiskLevel
from app.security.sanitize import sanitize_tool_argument_string
from app.tools.base import Tool, ToolResult, ToolSpec


class PlanTool(Tool):
    spec = ToolSpec(
        name="plan",
        description="Create/replace a multi-step plan for complex work.",
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

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            goal = sanitize_tool_argument_string(arguments["goal"], field_name="goal")
            steps = arguments["steps"]
            if not isinstance(steps, list):
                raise ValueError("steps must be array")
            lines = [f"Goal: {goal}", ""]
            for i, s in enumerate(steps, 1):
                lines.append(f"{i}. {str(s)[:200]}")
            return ToolResult(ok=True, payload="\n".join(lines))
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))


class TodoWriteTool(Tool):
    spec = ToolSpec(
        name="todo_write",
        description="Update task list (todos).",
        parameters_schema={
            "type": "object",
            "properties": {
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
                }
            },
            "required": ["todos"],
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        todos = arguments.get("todos", [])
        lines = ["Todos:"]
        for t in todos[:50]:
            lines.append(f"- [{t['status']}] ({t['priority']}) {t['content']}")
        return ToolResult(ok=True, payload="\n".join(lines))
