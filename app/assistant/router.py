"""
Routes structured tool calls from the model to the allowlisted registry.
Execution never interprets free-form model text as shell.
"""

from __future__ import annotations

import json
from typing import Any

from typing import Callable

from app.audit.logger import AuditLogger, ToolAuditRecord
from app.security.confirm import confirm_risky_action
from app.security.policy import requires_user_confirmation
from app.tools.base import ToolResult
from app.tools.registry import ToolRegistry


class ToolDispatchError(Exception):
    pass


Confirmer = Callable[[str, str], bool]


def _parse_arguments(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError as e:
        raise ToolDispatchError(f"Invalid tool arguments JSON: {e}") from e
    if not isinstance(data, dict):
        raise ToolDispatchError("Tool arguments must be a JSON object")
    return data


class ToolRouter:
    """The only entry point from the assistant into tool execution."""

    def __init__(
        self,
        registry: ToolRegistry,
        audit: AuditLogger,
        confirmer: Confirmer | None = None,
    ) -> None:
        self._registry = registry
        self._audit = audit
        self._confirmer = confirmer or (
            lambda tool_name, summary: confirm_risky_action(tool_name=tool_name, summary=summary)
        )

    def set_confirmer(self, confirmer: Confirmer) -> None:
        self._confirmer = confirmer

    def handle_tool_call(self, *, tool_name: str, arguments_json: str) -> str:
        """
        Returns a short string suitable for a tool role message back to the model.
        Always writes an audit row when execution is attempted.
        """
        try:
            args = _parse_arguments(arguments_json)
        except ToolDispatchError as e:
            self._audit.log_tool_result(
                ToolAuditRecord(
                    tool_name=tool_name,
                    arguments={"_raw": arguments_json[:500]},
                    ok=False,
                    outcome_summary=str(e),
                    detail={"phase": "parse"},
                )
            )
            return f"[tool error] {e}"

        tool = self._registry.get(tool_name)
        if tool is None:
            msg = f"Unknown or disallowed tool: {tool_name}"
            self._audit.log_tool_result(
                ToolAuditRecord(
                    tool_name=tool_name,
                    arguments=args,
                    ok=False,
                    outcome_summary=msg,
                    detail={"phase": "lookup"},
                )
            )
            return f"[tool error] {msg}"

        summary = f"{tool_name}({json.dumps(args, sort_keys=True)[:500]})"
        if requires_user_confirmation(tool.spec.risk):
            approved = self._confirmer(tool_name, summary)
            if not approved:
                self._audit.log_tool_result(
                    ToolAuditRecord(
                        tool_name=tool_name,
                        arguments=args,
                        ok=False,
                        outcome_summary="User denied confirmation",
                        detail={"phase": "confirm"},
                    )
                )
                return "[tool cancelled] User did not confirm high-risk action."

        result = tool.run(args)
        self._audit.log_tool_result(
            ToolAuditRecord(
                tool_name=tool_name,
                arguments=args,
                ok=result.ok,
                outcome_summary=result.payload[:2000],
                detail=result.detail,
            )
        )
        prefix = "ok" if result.ok else "error"
        return f"[tool {prefix}] {result.payload}"
