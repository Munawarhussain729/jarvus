"""
Allowlisted fixed commands only. The model never supplies argv or shell strings.

The model may only pass a predefined command_id that maps to a frozen argv list.
"""

from __future__ import annotations

import subprocess
from typing import Any, ClassVar

from app.security.policy import RiskLevel
from app.tools.base import Tool, ToolResult, ToolSpec

# Frozen allowlist: extend by adding entries here, not from runtime/config for MVP.
ALLOWED: dict[str, list[str]] = {
    "pwd": ["pwd"],
    "date_utc": ["date", "-u"],
}


class SafeShellTool(Tool):
    """
    Run a tiny allowlisted command set with shell=False.
    Marked HIGH risk so the user must confirm each invocation (MVP: fail-closed UX).
    """

    spec = ToolSpec(
        name="safe_shell",
        description=(
            "Run one predefined read-only system command by id. "
            "Allowed ids: pwd, date_utc. No arbitrary commands."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "command_id": {
                    "type": "string",
                    "enum": list(ALLOWED.keys()),
                    "description": "Allowlisted command identifier only.",
                }
            },
            "required": ["command_id"],
            "additionalProperties": False,
        },
        risk=RiskLevel.HIGH,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        command_id = arguments.get("command_id")
        if not isinstance(command_id, str) or command_id not in ALLOWED:
            return ToolResult(
                ok=False,
                payload="Invalid or unknown command_id",
                detail={"command_id": command_id},
            )
        argv = ALLOWED[command_id]
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=10,
                shell=False,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, payload="Command timed out", detail={"argv": argv})
        except OSError as e:
            return ToolResult(ok=False, payload=str(e), detail={"argv": argv})

        out = (proc.stdout or "") + (proc.stderr or "")
        ok = proc.returncode == 0
        return ToolResult(
            ok=ok,
            payload=out.strip() or "(no output)",
            detail={"returncode": proc.returncode, "argv": argv},
        )
