"""Filesystem tools with path sandboxing — read-only and write operations."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from app.security.policy import RiskLevel
from app.security.sanitize import InputRejected, sanitize_tool_argument_string
from app.tools.base import Tool, ToolResult, ToolSpec


def _allowed_roots() -> list[Path]:
    roots = [Path.home().resolve(), Path("/tmp").resolve()]
    extra = os.environ.get("JARVUS_FS_ROOTS", "")
    for raw in extra.split(os.pathsep):
        if raw.strip():
            roots.append(Path(raw).expanduser().resolve())
    return roots


# Credential stores and secrets that must never be reachable through the file
# tools, even though they live under the (allowed) home directory.
_DENIED_HOME_SUBPATHS = (
    ".ssh",
    ".aws",
    ".gnupg",
    ".kube",
    ".azure",
    ".config/gh",
    ".config/gcloud",
    ".docker/config.json",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".git-credentials",
    "Library/Keychains",
    "Library/Cookies",
    "Library/Application Support/Google/Chrome",
    "Library/Safari",
)


def _is_denied(path: Path) -> bool:
    if path.name == ".env" or path.name.startswith(".env."):
        return True
    home = Path.home().resolve()
    for sub in _DENIED_HOME_SUBPATHS:
        denied = (home / sub).resolve()
        if path == denied or path.is_relative_to(denied):
            return True
    return False


def _validate_path(p: str, *, field: str = "path") -> Path:
    try:
        # sanitize
        p = sanitize_tool_argument_string(p, field_name=field)
    except InputRejected as e:
        raise ValueError(str(e)) from e
    path = Path(p).expanduser().resolve()
    if not any(path == r or path.is_relative_to(r) for r in _allowed_roots()):
        raise ValueError(f"Path outside allowed sandbox: {path}")
    if _is_denied(path):
        raise ValueError(f"Path is protected (credentials/secrets): {path}")
    return path


class ReadFileTool(Tool):
    spec = ToolSpec(
        name="read_file",
        description="Read text from a file (sandboxed to user dirs).",
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or ~ path"},
                "offset": {"type": "integer", "minimum": 1, "description": "1-based line start"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10000},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            path = _validate_path(arguments["path"])
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        offset = arguments.get("offset") or 1
        limit = arguments.get("limit") or 2000
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        start = max(0, int(offset) - 1)
        end = start + int(limit)
        chunk = lines[start:end]
        text = "".join(f"{i+start+1}: {l}" for i, l in enumerate(chunk))
        if end < len(lines):
            text += f"\n... ({len(lines)-end} more lines)"
        return ToolResult(ok=True, payload=text or "(empty file)")


class WriteFileTool(Tool):
    spec = ToolSpec(
        name="write_file",
        description="Write text to a file (creates dirs, overwrites if exists). HIGH risk.",
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        risk=RiskLevel.HIGH,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            path = _validate_path(arguments["path"])
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(arguments["content"])
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        return ToolResult(ok=True, payload=f"wrote {len(arguments['content'])} bytes to {path}")


class ListDirTool(Tool):
    spec = ToolSpec(
        name="list_dir",
        description="List directory contents.",
        parameters_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            path = _validate_path(arguments["path"])
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        try:
            items = sorted(os.listdir(path))
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        lines = []
        for it in items:
            p = path / it
            if p.is_dir():
                lines.append(f"dir:  {it}/")
            else:
                lines.append(f"file: {it}")
        return ToolResult(ok=True, payload="\n".join(lines) or "(empty)")


class GlobTool(Tool):
    spec = ToolSpec(
        name="glob_files",
        description="Find files by glob pattern under a root.",
        parameters_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "root": {"type": "string", "default": "~/Projects"},
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        import glob as g
        try:
            root = _validate_path(arguments.get("root") or "~/Projects")
            pattern = sanitize_tool_argument_string(arguments["pattern"], field_name="pattern")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        try:
            matches = g.glob(str(root / pattern), recursive=True)
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        # limit to 200
        matches = matches[:200]
        return ToolResult(ok=True, payload="\n".join(matches) or "(no matches)")
