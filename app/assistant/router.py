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


def _schema_hint(tool) -> str:
    return f" Expected arguments schema: {json.dumps(tool.spec.parameters_schema)[:600]}"


def _validate_arguments(schema: dict[str, Any], args: dict[str, Any]) -> None:
    """Lightweight check of the JSON-schema subset our tools use. Raises
    ToolDispatchError with an actionable message so the model can self-repair
    on the next round instead of the tool failing on garbage input."""
    props = schema.get("properties") or {}
    required = schema.get("required") or []
    for name in required:
        if name not in args:
            raise ToolDispatchError(
                f"Missing required argument '{name}' (expected one of: {', '.join(props) or 'none'})")
    for name, spec in props.items():
        if name not in args:
            continue
        val = args[name]
        t = spec.get("type")
        if t == "integer" and (not isinstance(val, int) or isinstance(val, bool)):
            raise ToolDispatchError(f"Argument '{name}' must be an integer, got {val!r}")
        if t == "number" and (not isinstance(val, (int, float)) or isinstance(val, bool)):
            raise ToolDispatchError(f"Argument '{name}' must be a number, got {val!r}")
        if t == "string" and not isinstance(val, str):
            raise ToolDispatchError(f"Argument '{name}' must be a string, got {val!r}")
        if t == "boolean" and not isinstance(val, bool):
            raise ToolDispatchError(f"Argument '{name}' must be a boolean, got {val!r}")
        enum = spec.get("enum")
        if enum and val not in enum:
            raise ToolDispatchError(f"Argument '{name}' must be one of {enum}, got {val!r}")
        if t in ("number", "integer"):
            lo, hi = spec.get("minimum"), spec.get("maximum")
            if (lo is not None and val < lo) or (hi is not None and val > hi):
                raise ToolDispatchError(f"Argument '{name}'={val!r} out of range [{lo}, {hi}]")


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
            # Echo the expected schema back so the model can self-repair the
            # call on the next round instead of repeating the same bad JSON.
            hint = ""
            tool = self._registry.get(tool_name)
            if tool is not None:
                hint = _schema_hint(tool)
            return f"[tool error] {e}.{hint} Retry the tool call with corrected arguments."

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

        try:
            _validate_arguments(tool.spec.parameters_schema or {}, args)
        except ToolDispatchError as e:
            self._audit.log_tool_result(
                ToolAuditRecord(
                    tool_name=tool_name,
                    arguments=args,
                    ok=False,
                    outcome_summary=str(e),
                    detail={"phase": "validate"},
                )
            )
            return f"[tool error] {e}.{_schema_hint(tool)} Retry the tool call with corrected arguments."

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
