"""Clock tool - current local time and date."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from app.security.policy import RiskLevel
from app.tools.base import Tool, ToolResult, ToolSpec


class ClockTool(Tool):
    spec = ToolSpec(
        name="clock",
        description="Get the current local time and date.",
        parameters_schema={"type": "object", "properties": {}, "additionalProperties": False},
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        now = datetime.now()
        return ToolResult(
            ok=True,
            payload=f"{now.strftime('%A, %B %d %Y, %I:%M %p').replace(' 0', ' ')}",
        )
