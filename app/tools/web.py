"""Web tools - search and fetch (local-safe, read-only)."""
from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from app.security.policy import RiskLevel
from app.security.sanitize import sanitize_tool_argument_string
from app.tools.base import Tool, ToolResult, ToolSpec

ALLOWED_DOMAINS = []  # keep empty for now - require explicit URLs? Or block external? Local-first means cautious
# But user wants full capability - add with caution; mark HIGH? Or MEDIUM


class WebFetchTool(Tool):
    spec = ToolSpec(
        name="web_fetch",
        description="Fetch web content (markdown). HIGH risk - network access.",
        parameters_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "format": "uri"},
                "format": {"type": "string", "enum": ["text", "markdown", "html"], "default": "markdown"},
            },
            "required": ["url"],
            "additionalProperties": False,
        },
        risk=RiskLevel.HIGH,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            url = sanitize_tool_argument_string(arguments["url"], field_name="url")
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return ToolResult(ok=False, payload="only http/https")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        try:
            import httpx
        except Exception:
            return ToolResult(ok=False, payload="httpx not installed; pip install httpx to use web tools")
        try:
            with httpx.Client(timeout=60, follow_redirects=True) as c:
                r = c.get(url)
                r.raise_for_status()
                text = r.text
            fmt = arguments.get("format", "markdown")
            # basic truncation
            if len(text) > 50000:
                text = text[:50000] + "\n... (truncated)"
            return ToolResult(ok=True, payload=text)
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
