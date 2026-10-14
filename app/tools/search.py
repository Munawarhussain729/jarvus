"""Web search tool using DuckDuckGo lite (POST required - GET is bot-blocked)."""
from __future__ import annotations

import re
import urllib.parse
import urllib.request
from typing import Any

from app.security.policy import RiskLevel
from app.security.sanitize import sanitize_tool_argument_string
from app.tools.base import Tool, ToolResult, ToolSpec

_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
_LINK_RE = re.compile(r"<a[^>]*href=\"([^\"]*)\"[^>]*class='result-link'[^>]*>(.*?)</a>", re.S)
_SNIPPET_RE = re.compile(r"<td class='result-snippet'>(.*?)</td>", re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def _clean(html: str) -> str:
    return _TAG_RE.sub("", html).strip()


class WebSearchTool(Tool):
    spec = ToolSpec(
        name="web_search",
        description="Search the web using DuckDuckGo. Returns top results with snippets.",
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
            max_results = min(int(arguments.get("max_results", 5)), 10)
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))

        try:
            req = urllib.request.Request(
                "https://lite.duckduckgo.com/lite/",
                data=urllib.parse.urlencode({"q": query}).encode(),
                headers={"User-Agent": _UA},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8", errors="replace")

            links = _LINK_RE.findall(html)
            snippets = _SNIPPET_RE.findall(html)
            results = []
            for i, (href, title) in enumerate(links[:max_results]):
                entry = f"{_clean(title)}\n  {href}"
                if i < len(snippets):
                    snippet = _clean(snippets[i])
                    if snippet:
                        entry += f"\n  {snippet}"
                results.append(entry)

            if not results:
                return ToolResult(ok=True, payload=f"No results found for: {query}")
            return ToolResult(ok=True, payload="\n".join(results))
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
