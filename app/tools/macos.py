"""macOS-specific system control tools."""
from __future__ import annotations

import subprocess
from typing import Any

from app.security.policy import RiskLevel
from app.security.sanitize import sanitize_tool_argument_string
from app.tools.base import Tool, ToolResult, ToolSpec


class OpenAppTool(Tool):
    spec = ToolSpec(
        name="open_app",
        description="Open an application on macOS (e.g., 'WhatsApp', 'Safari', 'Terminal'). HIGH risk.",
        parameters_schema={
            "type": "object",
            "properties": {
                "app_name": {"type": "string", "description": "App name like 'WhatsApp.app' or just 'WhatsApp'"},
                "url": {"type": "string", "description": "Optional URL to open with app"},
            },
            "additionalProperties": False,
        },
        risk=RiskLevel.HIGH,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            app_name = sanitize_tool_argument_string(arguments.get("app_name", ""), field_name="app_name") if arguments.get("app_name") else None
            url = sanitize_tool_argument_string(arguments.get("url", ""), field_name="url") if arguments.get("url") else None
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        if not app_name and not url:
            return ToolResult(ok=False, payload="need app_name or url")
        args = ["open"]
        if app_name:
            # add .app if missing and not a path
            if "/" not in app_name and not app_name.endswith(".app"):
                # try both? just pass as-is
                args.extend(["-a", app_name])
            else:
                args.append(app_name)
        if url:
            args.append(url)
        try:
            proc = subprocess.run(args, capture_output=True, text=True, timeout=30)
            out = (proc.stdout or "") + (proc.stderr or "")
            return ToolResult(ok=proc.returncode == 0, payload=out or f"opened", detail={"code": proc.returncode})
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))


class SayTool(Tool):
    spec = ToolSpec(
        name="say",
        description="Speak text using macOS 'say' command.",
        parameters_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "maxLength": 500},
            },
            "required": ["text"],
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            text = sanitize_tool_argument_string(arguments["text"], field_name="text")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        try:
            subprocess.run(["say", text], timeout=60, capture_output=True)
            return ToolResult(ok=True, payload=f"said: {text}")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
