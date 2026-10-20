"""Devin orchestration - Jarvus dispatches prompts to Devin agent sessions.

Projects are registered in memory.meta["devin_projects"]:
    {name: {"path": str|None, "cloud": bool, "session": str|None,
            "permission_mode": str|None}}

Two dispatch modes per send:
- headless: `devin -p "..."` runs one prompt in the project dir and returns
  Devin's reply text (what Jarvus speaks back to the user).
- interactive: opens a Terminal window running `devin --prompt-file ...` so the
  session stays open and steerable in front of the user.
"""
from __future__ import annotations

import json
import logging
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from app.security.policy import RiskLevel
from app.security.sanitize import sanitize_tool_argument_string
from app.tools.base import Tool, ToolResult, ToolSpec

log = logging.getLogger("jarvus.tools.devin")

_REGISTRY_KEY = "devin_projects"
_DEVIN_BIN = "devin"


def _load_registry(memory) -> dict[str, dict[str, Any]]:
    raw = memory.get_meta(_REGISTRY_KEY)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return {str(k): dict(v) for k, v in data.items() if isinstance(v, dict)}
    except Exception:
        return {}


def _save_registry(memory, reg: dict[str, dict[str, Any]]) -> None:
    memory.set_meta(_REGISTRY_KEY, json.dumps(reg))


def _resolve_project(reg: dict[str, dict[str, Any]], name: str) -> tuple[str, dict[str, Any]] | tuple[None, None]:
    """Match a project by exact name, case-insensitive name, or substring."""
    if name in reg:
        return name, reg[name]
    low = name.lower()
    for k, v in reg.items():
        if k.lower() == low:
            return k, v
    hits = [(k, v) for k, v in reg.items() if low in k.lower()]
    if len(hits) == 1:
        return hits[0]
    return None, None


def _devin_cmd(*, cloud: bool, session: str | None, continue_last: bool,
               permission_mode: str | None, prompt: str | None,
               prompt_file: str | None, headless: bool) -> list[str]:
    cmd = [_DEVIN_BIN]
    if cloud:
        cmd.append("--cloud")
    if session:
        cmd += ["-r", session]
    elif continue_last:
        cmd.append("-c")
    if permission_mode:
        cmd += ["--permission-mode", permission_mode]
    if headless:
        cmd += ["-p", prompt or "", "--respect-workspace-trust", "false"]
    else:
        cmd += ["--prompt-file", prompt_file or ""]
    return cmd


class DevinProjectTool(Tool):
    """Register / update / remove a Devin project target."""

    spec = ToolSpec(
        name="devin_project",
        description="Register or remove a Devin project (name -> local directory or cloud session). "
                    "Use when the user wants to add/remove/rename a project Jarvus can drive.",
        parameters_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Project short name, e.g. 'jarvus'"},
                "path": {"type": "string", "description": "Absolute path to the project directory (local sessions)"},
                "cloud": {"type": "boolean", "default": False,
                          "description": "True if sends should use 'devin --cloud'"},
                "session": {"type": "string",
                            "description": "Devin session id or URL to resume (optional)"},
                "permission_mode": {"type": "string",
                                    "enum": ["auto", "accept-edits", "smart", "dangerous"],
                                    "description": "Devin permission mode for dispatched prompts"},
                "remove": {"type": "boolean", "default": False},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def __init__(self, memory) -> None:
        self._memory = memory

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        name = sanitize_tool_argument_string(str(arguments.get("name") or "").strip(), field_name="name")
        if not name:
            return ToolResult(ok=False, payload="name is required")
        key = name.lower().replace(" ", "-")
        reg = _load_registry(self._memory)
        if arguments.get("remove"):
            if reg.pop(key, None) is None:
                return ToolResult(ok=False, payload=f"no project named '{key}'")
            _save_registry(self._memory, reg)
            return ToolResult(ok=True, payload=f"removed project '{key}'")
        entry = dict(reg.get(key) or {})
        if arguments.get("path"):
            p = Path(str(arguments["path"])).expanduser()
            if not p.is_dir():
                return ToolResult(ok=False, payload=f"not a directory: {p}")
            entry["path"] = str(p)
        if "cloud" in arguments:
            entry["cloud"] = bool(arguments["cloud"])
        if arguments.get("session"):
            entry["session"] = str(arguments["session"]).strip()
        if arguments.get("permission_mode"):
            entry["permission_mode"] = str(arguments["permission_mode"])
        if not entry.get("path") and not entry.get("cloud"):
            return ToolResult(ok=False, payload="need 'path' (local project) or cloud=true")
        reg[key] = entry
        _save_registry(self._memory, reg)
        return ToolResult(ok=True, payload=f"registered project '{key}': {json.dumps(entry)}")


class DevinProjectsTool(Tool):
    """List registered projects and their recent Devin sessions."""

    spec = ToolSpec(
        name="devin_projects",
        description="List registered Devin projects and each project's recent Devin sessions "
                    "(titles + ids). Use when the user asks what projects/sessions are available.",
        parameters_schema={"type": "object", "properties": {}, "additionalProperties": False},
        risk=RiskLevel.LOW,
    )

    def __init__(self, memory) -> None:
        self._memory = memory

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        reg = _load_registry(self._memory)
        if not reg:
            return ToolResult(
                ok=True,
                payload="No Devin projects registered. Register one with devin_project "
                        "(name + path) or set JARVUS_DEVIN_PROJECTS=name=/path,...")
        lines = []
        for name, e in reg.items():
            kind = "cloud" if e.get("cloud") else "local"
            lines.append(f"{name} ({kind}): {e.get('path') or e.get('session') or '?'}")
            if e.get("session"):
                lines.append(f"  saved session: {e['session']}")
            path = e.get("path")
            if path and Path(path).is_dir() and not e.get("cloud"):
                lines += self._list_sessions(path)
        return ToolResult(ok=True, payload="\n".join(lines))

    @staticmethod
    def _list_sessions(path: str) -> list[str]:
        try:
            proc = subprocess.run(
                [_DEVIN_BIN, "list", "--format", "json"],
                capture_output=True, text=True, timeout=15, cwd=path,
                stdin=subprocess.DEVNULL,
            )
            sessions = json.loads(proc.stdout or "[]")
        except Exception as ex:
            return [f"  sessions: unavailable ({ex})"]
        if not sessions:
            return ["  sessions: none"]
        return [
            f"  - {s.get('title') or '(untitled)'} [{s.get('id')}] - {s.get('last_activity_ago', '?')}"
            for s in sessions[:5]
        ]


class DevinSendTool(Tool):
    """Send a prompt to a Devin session - headless reply or interactive window."""

    spec = ToolSpec(
        name="devin_send",
        description="Send a prompt to Devin on a registered project. Headless mode waits for "
                    "Devin's reply; interactive mode opens a Terminal window with the session. "
                    "HIGH risk - Devin will act autonomously on the project.",
        parameters_schema={
            "type": "object",
            "properties": {
                "project": {"type": "string",
                            "description": "Registered project name (or absolute path)"},
                "prompt": {"type": "string", "description": "Instruction for Devin"},
                "session": {"type": "string",
                            "description": "Session id/URL to resume (default: project's saved session)"},
                "continue_last": {"type": "boolean", "default": False,
                                  "description": "Continue the most recent session in the project dir"},
                "interactive": {"type": "boolean", "default": False,
                                "description": "Open a Terminal window instead of waiting for a reply"},
                "timeout": {"type": "integer", "default": 300, "maximum": 600},
            },
            "required": ["project", "prompt"],
            "additionalProperties": False,
        },
        risk=RiskLevel.HIGH,
    )

    def __init__(self, memory) -> None:
        self._memory = memory

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        project = sanitize_tool_argument_string(str(arguments.get("project") or "").strip(),
                                                field_name="project")
        prompt = sanitize_tool_argument_string(str(arguments.get("prompt") or "").strip(),
                                               field_name="prompt")
        if not project or not prompt:
            return ToolResult(ok=False, payload="project and prompt are required")

        reg = _load_registry(self._memory)
        name, entry = _resolve_project(reg, project)
        if entry is None:
            # ad-hoc: treat `project` as a path
            p = Path(project).expanduser()
            if p.is_dir():
                name, entry = p.name, {"path": str(p)}
            else:
                known = ", ".join(reg) or "none"
                return ToolResult(ok=False,
                                  payload=f"unknown project '{project}' (registered: {known})")

        cloud = bool(entry.get("cloud"))
        session = str(arguments.get("session") or entry.get("session") or "").strip() or None
        continue_last = bool(arguments.get("continue_last"))
        permission_mode = entry.get("permission_mode")
        path = entry.get("path")

        if arguments.get("interactive"):
            return self._open_window(path, cloud=cloud, session=session,
                                     continue_last=continue_last,
                                     permission_mode=permission_mode, prompt=prompt)
        return self._headless(name, path, cloud=cloud, session=session,
                              continue_last=continue_last,
                              permission_mode=permission_mode, prompt=prompt,
                              timeout=int(arguments.get("timeout") or 300))

    def _headless(self, name, path, *, cloud, session, continue_last,
                  permission_mode, prompt, timeout) -> ToolResult:
        cmd = _devin_cmd(cloud=cloud, session=session, continue_last=continue_last,
                         permission_mode=permission_mode, prompt=prompt,
                         prompt_file=None, headless=True)
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=min(timeout, 600), cwd=path or None,
                stdin=subprocess.DEVNULL,  # never fall into an interactive REPL
            )
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False,
                              payload=f"Devin on '{name}' exceeded {timeout}s - the session keeps "
                                      "running; resume it in the project dir or ask for interactive mode")
        except FileNotFoundError:
            return ToolResult(ok=False, payload="'devin' CLI not found on PATH")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        out = ((proc.stdout or "") + (proc.stderr or "")).strip()
        return ToolResult(ok=proc.returncode == 0,
                          payload=out[-6000:] or f"(exit {proc.returncode}, no output)",
                          detail={"code": proc.returncode, "cmd": " ".join(cmd[:3]) + " ..."})

    def _open_window(self, path, *, cloud, session, continue_last,
                     permission_mode, prompt) -> ToolResult:
        """Open Terminal.app running `devin --prompt-file` - session stays open."""
        try:
            f = tempfile.NamedTemporaryFile("w", prefix="jarvus_devin_",
                                            suffix=".txt", delete=False)
            f.write(prompt)
            f.close()
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        inner = _devin_cmd(cloud=cloud, session=session, continue_last=continue_last,
                           permission_mode=permission_mode, prompt=None,
                           prompt_file=f.name, headless=False)
        shell_cmd = " ".join(shlex.quote(a) for a in inner)
        if path:
            shell_cmd = f"cd {shlex.quote(str(path))} && {shell_cmd}"
        # Escape for embedding inside an AppleScript double-quoted string.
        script = 'tell application "Terminal" to do script "{}"'.format(
            shell_cmd.replace("\\", "\\\\").replace('"', '\\"'))
        try:
            subprocess.run(["osascript", "-e", script], capture_output=True,
                           text=True, timeout=15)
        except Exception as e:
            return ToolResult(ok=False, payload=f"couldn't open Terminal: {e}")
        return ToolResult(ok=True,
                          payload="Opened a Terminal window running Devin with your prompt.")
