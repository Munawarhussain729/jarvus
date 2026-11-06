"""Extended shell tool with safer allowlist."""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any

from app.security.policy import RiskLevel
from app.security.sanitize import sanitize_tool_argument_string
from app.tools.base import Tool, ToolResult, ToolSpec


# Very restricted - only read-only/git inspection etc
SAFE_READ_COMMANDS = {
    "ls": set(["-la", "-l", "-a", "-lh", "--color=never"]),
    "pwd": set(),
    "date": set(["-u"]),
    "git": set(["status", "--short", "--branch", "log", "--oneline", "-5", "-10", "--no-pager", "diff", "--name-only"]),
    "cat": set(),
    "head": set(),
    "tail": set(),
    "wc": set(),
    "python3": set(["--version"]),
    "node": set(["--version"]),
    "npm": set(["--version"]),
    "which": set(),
    "find": set(),
    "open": set(["-a"]),  # allow open for launching apps
}


# Commands whose non-flag arguments are file paths; these go through the same
# sandbox as the file tools so e.g. `cat ~/.ssh/id_rsa` is refused.
_PATH_ARG_COMMANDS = {"ls", "cat", "head", "tail", "wc", "find"}
_GIT_SUBCOMMANDS = {"status", "log", "diff"}
_FIND_FORBIDDEN = {"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprint0", "-fprintf", "-fls"}


def _validate_command(parts: list[str], workdir: Path | None) -> str | None:
    """Return an error string if the argv is not allowed, else None."""
    from app.tools.filesystem import _validate_path

    base, args = parts[0], parts[1:]
    allowed_flags = SAFE_READ_COMMANDS[base]
    if base == "git":
        if not args or args[0] not in _GIT_SUBCOMMANDS:
            return "git: only status, log and diff are allowed"
    if base in {"python3", "node", "npm", "date", "pwd"}:
        # Interpreters/utilities may only be asked for exactly their allowed flags
        # (e.g. `python3 --version`); never a script, module or positional argument.
        for a in args:
            if a not in allowed_flags:
                return f"{base}: argument not allowed: {a}"
    if base == "find":
        bad = _FIND_FORBIDDEN.intersection(args)
        if bad:
            return f"find: option not allowed: {sorted(bad)[0]}"
    for i, a in enumerate(args):
        if base == "git" and i == 0:
            continue
        if a.startswith("-"):
            if base in {"cat", "head", "tail", "wc", "find", "which"}:
                continue  # numeric/predicate flags; no execution or write capability
            if a not in allowed_flags:
                return f"{base}: option not allowed: {a}"
        elif base in _PATH_ARG_COMMANDS:
            target = Path(a).expanduser()
            if not target.is_absolute() and workdir is not None:
                target = workdir / target
            try:
                _validate_path(str(target))
            except ValueError as e:
                return str(e)
    return None


class ShellReadOnlyTool(Tool):
    spec = ToolSpec(
        name="shell_read",
        description="Run safe read-only shell commands. HIGH risk (still system access).",
        parameters_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "e.g. 'ls -la' (restricted set)"},
                "workdir": {"type": "string", "description": "Working directory (sandboxed)"},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        risk=RiskLevel.HIGH,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            cmd = sanitize_tool_argument_string(arguments["command"], field_name="command")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        try:
            parts = shlex.split(cmd)
        except ValueError as e:
            return ToolResult(ok=False, payload=f"could not parse command: {e}")
        if not parts:
            return ToolResult(ok=False, payload="empty command")
        base = parts[0]
        if base not in SAFE_READ_COMMANDS:
            return ToolResult(ok=False, payload=f"command not allowed: {base}")
        workdir = None
        if arguments.get("workdir"):
            try:
                from app.tools.filesystem import _validate_path
                workdir = _validate_path(arguments["workdir"])
            except Exception as e:
                return ToolResult(ok=False, payload=str(e))
        problem = _validate_command(parts, workdir)
        if problem:
            return ToolResult(ok=False, payload=problem)
        try:
            proc = subprocess.run(
                parts,
                shell=False,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(workdir) if workdir else None,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, payload="timeout")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        out = (proc.stdout or "") + (proc.stderr or "")
        return ToolResult(ok=proc.returncode == 0, payload=out[:8000], detail={"code": proc.returncode})
