"""Advanced terminal tool - with very strict controls."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from app.security.policy import RiskLevel
from app.security.sanitize import sanitize_tool_argument_string
from app.tools.base import Tool, ToolResult, ToolSpec


class RunCommandTool(Tool):
    spec = ToolSpec(
        name="run_command",
        description="Run a specific command with args. EXTREMELY HIGH risk - full control.",
        parameters_schema={
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "Command binary"},
                "args": {"type": "array", "items": {"type": "string"}, "default": []},
                "cwd": {"type": "string"},
                "timeout": {"type": "integer", "default": 60, "maximum": 300},
            },
            "required": ["cmd"],
            "additionalProperties": False,
        },
        risk=RiskLevel.HIGH,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            cmd = sanitize_tool_argument_string(arguments["cmd"], field_name="cmd")
            args = arguments.get("args") or []
            if not isinstance(args, list):
                return ToolResult(ok=False, payload="args must be list")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        # block obvious dangerous cmds unless explicitly needed? user wants control - but still confirm
        blocked = ["rm -rf /", ":(){ :|:& };:", "mkfs", "dd if=/dev/zero"]
        full = " ".join([cmd] + args)
        for b in blocked:
            if b in full:
                return ToolResult(ok=False, payload=f"blocked dangerous pattern")
        cwd = None
        if arguments.get("cwd"):
            try:
                from app.tools.filesystem import _validate_path
                cwd = _validate_path(arguments["cwd"])
            except Exception as e:
                return ToolResult(ok=False, payload=str(e))
        try:
            proc = subprocess.run(
                [cmd] + [str(a) for a in args],
                capture_output=True,
                text=True,
                timeout=int(arguments.get("timeout") or 60),
                cwd=str(cwd) if cwd else None,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, payload="timeout")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        out = (proc.stdout or "") + (proc.stderr or "")
        return ToolResult(ok=proc.returncode == 0, payload=out[:12000], detail={"code": proc.returncode})
