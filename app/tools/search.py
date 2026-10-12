"""Web search tool using DuckDuckGo."""
from __future__ import annotations

import subprocess
from typing import Any

from app.security.policy import RiskLevel
from app.security.sanitize import sanitize_tool_argument_string
from app.tools.base import Tool, ToolResult, ToolSpec


class WebSearchTool(Tool):
    spec = ToolSpec(
        name="web_search",
        description="Search the web using DuckDuckGo. Returns top results.",
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            query = sanitize_tool_argument_string(arguments["query"], field_name="query")
            max_results = int(arguments.get("max_results", 5))
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))

        try:
            import urllib.parse
            import urllib.request
            import re

            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode("utf-8", errors="replace")

            results = []
            pattern = r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>'
            for match in re.finditer(pattern, html):
                href = match.group(1)
                title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
                if title and href:
                    results.append(f"{title}: {href}")
                if len(results) >= max_results:
                    break

            if not results:
                return ToolResult(ok=True, payload=f"No results found for: {query}")

            return ToolResult(ok=True, payload="\n".join(results))
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
