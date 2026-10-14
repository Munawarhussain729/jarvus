"""Read recent macOS Notification Center notifications (usernoted db).

Requires Full Disk Access for the terminal/Python process
(System Settings -> Privacy & Security -> Full Disk Access).
"""
from __future__ import annotations

import plistlib
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from app.security.policy import RiskLevel
from app.security.sanitize import sanitize_tool_argument_string
from app.tools.base import Tool, ToolResult, ToolSpec

DB_PATH = Path.home() / "Library/Group Containers/group.com.apple.usernoted/db2/db"
# Core Data timestamps are seconds since 2001-01-01 UTC.
COCOA_EPOCH = datetime(2001, 1, 1)

NO_ACCESS_MSG = (
    "Cannot read notifications: Full Disk Access is required. "
    "Grant it in System Settings > Privacy & Security > Full Disk Access "
    "to the terminal app running Jarvus, then restart Jarvus."
)


def _plist_get(data: bytes, *keys: str) -> str:
    """Dig a string value out of a notification's binary plist, tolerant of schema drift."""
    try:
        obj = plistlib.loads(data)
    except Exception:
        return ""
    stack = [obj]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for k in keys:
                v = node.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            stack.extend(node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend(node)
    return ""


def _app_names(cur: sqlite3.Cursor) -> dict[int, str]:
    try:
        return {r[0]: r[1] for r in cur.execute("SELECT identifier, bundleid FROM app")}
    except sqlite3.Error:
        try:
            return {r[0]: r[1] for r in cur.execute("SELECT id, bundleid FROM app")}
        except sqlite3.Error:
            return {}


class NotificationsTool(Tool):
    spec = ToolSpec(
        name="read_notifications",
        description="Read recent macOS Notification Center notifications (app name, title, body, time). "
                    "Needs Full Disk Access granted to the terminal.",
        parameters_schema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                "app": {"type": "string", "description": "Optional filter: app name contains this text"},
            },
            "additionalProperties": False,
        },
        risk=RiskLevel.MEDIUM,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        if not DB_PATH.exists():
            return ToolResult(ok=False, payload=NO_ACCESS_MSG)
        try:
            limit = int(arguments.get("limit") or 10)
            app_filter = sanitize_tool_argument_string(
                str(arguments.get("app") or ""), field_name="app"
            ).strip().lower()
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
        try:
            conn = sqlite3.connect(
                f"file:{DB_PATH}?mode=ro&immutable=1", uri=True, timeout=3
            )
        except sqlite3.Error as e:
            return ToolResult(ok=False, payload=f"{NO_ACCESS_MSG} ({e})")
        try:
            cur = conn.cursor()
            apps = _app_names(cur)
            try:
                rows = cur.execute(
                    "SELECT data, delivered_date, app FROM record "
                    "WHERE presented = 1 ORDER BY delivered_date DESC LIMIT ?",
                    (limit * 4,),  # over-fetch: some rows may be filtered/undecodable
                ).fetchall()
            except sqlite3.Error:
                # schema drift fallback: no presented flag
                rows = cur.execute(
                    "SELECT data, delivered_date, app FROM record "
                    "ORDER BY delivered_date DESC LIMIT ?",
                    (limit * 4,),
                ).fetchall()
        except sqlite3.Error as e:
            conn.close()
            return ToolResult(ok=False, payload=f"{NO_ACCESS_MSG} ({e})")
        conn.close()

        lines = []
        for data, delivered, app_id in rows:
            app = apps.get(app_id, "") or _plist_get(data, "app", "bundleid", "bundle_id")
            title = _plist_get(data, "titl", "title", "header")
            body = _plist_get(data, "body", "subt", "subtitle", "message", "infor")
            if app_filter and app_filter not in app.lower() and app_filter not in (title + body).lower():
                continue
            when = ""
            try:
                when = (COCOA_EPOCH + timedelta(seconds=float(delivered))).strftime("%H:%M")
            except Exception:
                pass
            text = " - ".join(p for p in (title, body) if p)
            if not text:
                continue
            lines.append(f"[{when}] {app or 'unknown'}: {text}"[:300])
            if len(lines) >= limit:
                break
        if not lines:
            if app_filter:
                return ToolResult(ok=True, payload=f"No recent notifications matching '{app_filter}'.")
            return ToolResult(ok=True, payload="No recent notifications found (or access is restricted).")
        return ToolResult(ok=True, payload="\n".join(lines))
