"""Personal notes tool - append/read/search a persistent local notes file."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from app.security.policy import RiskLevel
from app.security.sanitize import sanitize_tool_argument_string
from app.tools.base import Tool, ToolResult, ToolSpec


def _notes_path() -> Path:
    data_dir = Path(os.environ.get("JARVUS_DATA_DIR", "~/.jarvus")).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "notes.md"


class NoteTool(Tool):
    spec = ToolSpec(
        name="scratchpad",
        description="Manage the user's personal scratchpad notes: add, list (recent), or search. "
                    "NOT related to system notifications.",
        parameters_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "list", "search"], "default": "list"},
                "text": {"type": "string", "description": "Note text (add) or search query (search)"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        action = arguments.get("action", "list")
        path = _notes_path()
        if action == "add":
            try:
                text = sanitize_tool_argument_string(str(arguments.get("text") or ""), field_name="text").strip()
            except Exception as e:
                return ToolResult(ok=False, payload=str(e))
            if not text:
                return ToolResult(ok=False, payload="text required for add")
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"- [{stamp}] {text}\n")
            return ToolResult(ok=True, payload=f"Note saved: {text[:80]}")

        if not path.exists():
            return ToolResult(ok=True, payload="No notes yet.")
        lines = [l.rstrip("\n") for l in path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]

        if action == "search":
            query = (arguments.get("text") or "").strip().lower()
            if not query:
                return ToolResult(ok=False, payload="text required for search")
            hits = [l for l in lines if query in l.lower()]
            return ToolResult(ok=True, payload="\n".join(hits[-20:]) or f"No notes matching '{query}'.")

        limit = int(arguments.get("limit") or 10)
        return ToolResult(ok=True, payload="\n".join(lines[-limit:]) or "No notes yet.")
