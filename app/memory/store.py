"""SQLite-backed local memory (chat turns). Audit rows live here too — single file DB."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


@dataclass
class ChatTurn:
    role: str
    content: str
    created_at: str


class MemoryStore:
    """
    Local-first persistence. For production hardening, swap to SQLCipher
    or file-level encryption (see README).
    """

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                arguments_json TEXT NOT NULL,
                ok INTEGER NOT NULL,
                outcome_summary TEXT NOT NULL,
                detail_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);
            CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log(created_at);
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def append_message(self, role: str, content: str) -> None:
        self._conn.execute(
            "INSERT INTO messages(role, content, created_at) VALUES (?, ?, ?)",
            (role, content, _utc_now_iso()),
        )
        self._conn.commit()

    def recent_messages(self, limit: int = 50) -> list[ChatTurn]:
        cur = self._conn.execute(
            "SELECT role, content, created_at FROM messages "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = [ChatTurn(r["role"], r["content"], r["created_at"]) for r in cur.fetchall()]
        return list(reversed(rows))

    def append_audit_row(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        ok: bool,
        outcome_summary: str,
        detail: dict[str, Any] | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO audit_log(created_at, tool_name, arguments_json, ok, outcome_summary, detail_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                _utc_now_iso(),
                tool_name,
                json.dumps(arguments, default=str),
                1 if ok else 0,
                outcome_summary[:10_000],
                json.dumps(detail or {}, default=str),
            ),
        )
        self._conn.commit()
