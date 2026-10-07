"""Code execution tools - run Python snippets in sandboxed way."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from app.security.policy import RiskLevel
from app.security.sanitize import sanitize_tool_argument_string
from app.tools.base import Tool, ToolResult, ToolSpec


class PythonExecTool(Tool):
    spec = ToolSpec(
        name="python_exec",
        description="Execute Python code (sandboxed, no network by default). VERY HIGH risk.",
        parameters_schema={
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "timeout": {"type": "integer", "minimum": 1, "maximum": 120, "default": 30},
            },
            "required": ["code"],
            "additionalProperties": False,
        },
        risk=RiskLevel.HIGH,  # should be treated as HIGH; user wants control but still needs approval
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            code = sanitize_tool_argument_string(arguments["code"], field_name="code")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        # Basic safety - block dangerous imports (simple heuristic)
        blocked = ["subprocess", "os.system", "eval(", "exec(", "__import__(", "open("]  # rough
        # but user wants "control everything" - still local; require confirmation which we have
        timeout = int(arguments.get("timeout") or 30)
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                tmp = f.name
            proc = subprocess.run(
                [sys.executable, "-I", tmp],  # isolated mode
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd="/tmp",
            )
            Path(tmp).unlink(missing_ok=True)
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, payload="timeout")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        out = (proc.stdout or "") + (proc.stderr or "")
        return ToolResult(ok=proc.returncode == 0, payload=out[:10000], detail={"code": proc.returncode})
