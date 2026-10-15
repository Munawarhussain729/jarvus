"""Additional macOS skills: Spotlight file search, URL opening, appearance,
screen lock/sleep, Shortcuts, trash."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.security.policy import RiskLevel
from app.security.sanitize import sanitize_tool_argument_string
from app.tools.applescript import run
from app.tools.base import Tool, ToolResult, ToolSpec


class FindFilesTool(Tool):
    spec = ToolSpec(
        name="find_files",
        description="Search for files by name (or contents) using Spotlight. Much faster "
                    "than walking the disk; results are confined to the sandbox roots.",
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "File name fragment or content keyword"},
                "scope": {"type": "string", "description": "Directory to search under", "default": "~"},
                "content": {"type": "boolean", "description": "true = search file contents too", "default": False},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 25},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            query = sanitize_tool_argument_string(str(arguments["query"]), field_name="query").strip()
            if not query or '"' in query:
                return ToolResult(ok=False, payload="query must be non-empty without quotes")
            from app.tools.filesystem import _validate_path
            scope = _validate_path(str(arguments.get("scope") or "~"))
            limit = max(1, min(100, int(arguments.get("limit") or 25)))
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        cmd = ["mdfind", "-onlyin", str(scope)]
        if not arguments.get("content"):
            cmd += ["-name"]
        cmd.append(query)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, payload="Spotlight search timed out")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        if proc.returncode != 0:
            return ToolResult(ok=False, payload=proc.stderr.strip() or "mdfind failed")
        hits = [l for l in proc.stdout.splitlines() if l.strip()][:limit]
        if not hits and not arguments.get("content"):
            # Spotlight skips hidden dirs (~/.jarvus) - fall back to find(1).
            try:
                proc2 = subprocess.run(
                    ["find", str(scope), "-iname", f"*{query}*", "-not", "-path", "*/node_modules/*"],
                    capture_output=True, text=True, timeout=30)
                hits = [l for l in proc2.stdout.splitlines() if l.strip()][:limit]
            except Exception:
                pass
        if not hits:
            return ToolResult(ok=True, payload=f"No files matching '{query}' under {scope}")
        shown = "\n".join(hits)
        if len(hits) == limit:
            shown += "\n… (more results; narrow the query)"
        return ToolResult(ok=True, payload=shown)


_TLD_ALLOWLIST = {
    "com", "org", "net", "io", "dev", "ai", "co", "app", "edu", "gov", "me",
    "xyz", "tv", "info", "site", "cloud", "gg", "sh", "fm", "us", "uk", "ca",
    "de", "fr", "jp", "au", "in", "br", "it", "es", "nl", "se", "ch", "local",
}


class OpenUrlTool(Tool):
    spec = ToolSpec(
        name="open_url",
        description="Open a website in the default browser (e.g. 'github.com', "
                    "'https://news.ycombinator.com'). For apps use open_app instead.",
        parameters_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL or bare domain"},
            },
            "required": ["url"],
            "additionalProperties": False,
        },
        risk=RiskLevel.MEDIUM,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            url = sanitize_tool_argument_string(str(arguments["url"]), field_name="url").strip()
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        if not url or any(c in url for c in " \t\n\"'"):
            return ToolResult(ok=False, payload="invalid url")
        if "://" not in url:
            host = url.split("/")[0].split("?")[0]
            tld = host.rsplit(".", 1)[-1].lower() if "." in host else ""
            if tld not in _TLD_ALLOWLIST:
                return ToolResult(ok=False, payload=f"'{url}' doesn't look like a web address")
            url = "https://" + url
        if urlparse(url).scheme not in ("http", "https"):
            return ToolResult(ok=False, payload="only http/https URLs")
        try:
            proc = subprocess.run(["open", url], capture_output=True, text=True, timeout=15)
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        if proc.returncode == 0:
            return ToolResult(ok=True, payload=f"Opened {url}")
        return ToolResult(ok=False, payload=proc.stderr.strip() or "open failed")


class DarkModeTool(Tool):
    spec = ToolSpec(
        name="dark_mode",
        description="Get, enable, disable or toggle macOS dark mode (system appearance).",
        parameters_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["get", "on", "off", "toggle"], "default": "get"},
            },
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    _BASE = 'tell application "System Events" to tell appearance preferences to '

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        action = arguments.get("action") or "get"
        if action == "get":
            res = run(self._BASE + "get dark mode")
            if res.ok:
                res.payload = f"Dark mode is {'on' if res.payload.strip().lower() == 'true' else 'off'}"
            return res
        if action == "toggle":
            res = run(self._BASE + "set dark mode to not dark mode")
        else:
            res = run(self._BASE + f"set dark mode to {'true' if action == 'on' else 'false'}")
        if res.ok:
            res.payload = f"Dark mode {action} applied"
        return res


class LockScreenTool(Tool):
    spec = ToolSpec(
        name="lock_screen",
        description="Lock the Mac screen immediately (shows the login window).",
        parameters_schema={"type": "object", "properties": {}, "additionalProperties": False},
        risk=RiskLevel.LOW,
    )

    _CGSESSION = "/System/Library/CoreServices/Menu Extras/User.menu/Contents/Resources/CGSession"

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        if not Path(self._CGSESSION).exists():
            return ToolResult(ok=False, payload="CGSession binary not found")
        try:
            subprocess.run([self._CGSESSION, "-suspend"], timeout=10, capture_output=True)
            return ToolResult(ok=True, payload="Screen locked")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))


class SleepDisplayTool(Tool):
    spec = ToolSpec(
        name="sleep_display",
        description="Turn off the display / put the screen to sleep (Mac keeps running).",
        parameters_schema={"type": "object", "properties": {}, "additionalProperties": False},
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            proc = subprocess.run(["pmset", "displaysleepnow"],
                                  capture_output=True, text=True, timeout=10)
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        if proc.returncode == 0:
            return ToolResult(ok=True, payload="Display sleeping")
        return ToolResult(ok=False, payload=proc.stderr.strip() or "pmset failed")


class ShortcutTool(Tool):
    spec = ToolSpec(
        name="run_shortcut",
        description="Run an Apple Shortcuts shortcut by name, or list available shortcuts.",
        parameters_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["run", "list"], "default": "run"},
                "name": {"type": "string", "description": "Shortcut name (run only)"},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        risk=RiskLevel.MEDIUM,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        action = arguments.get("action") or "run"
        if action == "list":
            try:
                proc = subprocess.run(["shortcuts", "list"], capture_output=True, text=True, timeout=15)
            except Exception as e:
                return ToolResult(ok=False, payload=str(e))
            if proc.returncode != 0:
                return ToolResult(ok=False, payload=proc.stderr.strip() or "shortcuts list failed")
            names = [l for l in proc.stdout.splitlines() if l.strip()][:100]
            return ToolResult(ok=True, payload="\n".join(names) or "No shortcuts installed.")
        try:
            name = sanitize_tool_argument_string(str(arguments.get("name") or ""), field_name="name").strip()
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        if not name:
            return ToolResult(ok=False, payload="name required")
        try:
            proc = subprocess.run(["shortcuts", "run", name],
                                  capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, payload="shortcut timed out")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        if proc.returncode == 0:
            out = proc.stdout.strip()
            return ToolResult(ok=True, payload=f"Shortcut '{name}' finished" + (f": {out[:500]}" if out else ""))
        return ToolResult(ok=False, payload=proc.stderr.strip() or f"shortcut '{name}' failed")


class EmptyTrashTool(Tool):
    spec = ToolSpec(
        name="empty_trash",
        description="Empty the macOS Trash. HIGH risk - permanently deletes trashed files.",
        parameters_schema={"type": "object", "properties": {}, "additionalProperties": False},
        risk=RiskLevel.HIGH,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        res = run('tell application "Finder" to empty trash')
        if res.ok:
            res.payload = "Trash emptied"
        elif "-1743" in res.payload:
            res.payload += " (grant Automation permission for Finder in Privacy & Security)"
        return res
