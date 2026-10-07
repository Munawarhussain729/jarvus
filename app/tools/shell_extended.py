"""Extended shell tool with safer allowlist."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from app.security.policy import RiskLevel
from app.security.sanitize import sanitize_tool_argument_string
from app.tools.base import Tool, ToolResult, ToolSpec


# Very restricted - only read-only/git inspection etc
SAFE_READ_COMMANDS = {
    "ls": set(["-la", "-l", "-a", "-lh", "--color=never"]),
    "pwd": set(),
    "date": set(["-u"]),
    "git": set(["status", "--short", "--branch", "log", "--oneline", "-5", "-10", "--no-pager", "diff", "--name-only"]),
    "cat": set(),
    "head": set(),
    "tail": set(),
    "wc": set(),
    "python3": set(["--version"]),
    "node": set(["--version"]),
    "npm": set(["--version"]),
    "which": set(),
    "find": set(),
    "open": set(["-a"]),  # allow open for launching apps
}


class ShellReadOnlyTool(Tool):
    spec = ToolSpec(
        name="shell_read",
        description="Run safe read-only shell commands. HIGH risk (still system access).",
        parameters_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "e.g. 'ls -la' (restricted set)"},
                "workdir": {"type": "string", "description": "Working directory (sandboxed)"},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        risk=RiskLevel.HIGH,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            cmd = sanitize_tool_argument_string(arguments["command"], field_name="command")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        parts = cmd.split()
        if not parts:
            return ToolResult(ok=False, payload="empty command")
        base = parts[0]
        if base not in SAFE_READ_COMMANDS:
            return ToolResult(ok=False, payload=f"command not allowed: {base}")
        workdir = None
        if arguments.get("workdir"):
            try:
                from app.tools.filesystem import _validate_path
                workdir = _validate_path(arguments["workdir"])
            except Exception as e:
                return ToolResult(ok=False, payload=str(e))
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(workdir) if workdir else None,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, payload="timeout")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        out = (proc.stdout or "") + (proc.stderr or "")
        return ToolResult(ok=proc.returncode == 0, payload=out[:8000], detail={"code": proc.returncode})
