"""System tools - process management, environment info."""
from __future__ import annotations

import os
import platform
import subprocess
from typing import Any

from app.security.policy import RiskLevel
from app.tools.base import Tool, ToolResult, ToolSpec


class SystemInfoTool(Tool):
    spec = ToolSpec(
        name="system_info",
        description="Get system information (OS, Python, env).",
        parameters_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        info = {
            "platform": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cwd": os.getcwd(),
            "home": str(Path.home()) if 'Path' in globals() else os.path.expanduser("~"),
        }
        try:
            from pathlib import Path
            info["home"] = str(Path.home())
        except:
            pass
        lines = [f"{k}: {v}" for k, v in info.items()]
        return ToolResult(ok=True, payload="\n".join(lines))


class PsTool(Tool):
    spec = ToolSpec(
        name="ps_list",
        description="List running processes (read-only).",
        parameters_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            proc = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=5)
            lines = proc.stdout.splitlines()[:100]  # limit
            return ToolResult(ok=True, payload="\n".join(lines))
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
