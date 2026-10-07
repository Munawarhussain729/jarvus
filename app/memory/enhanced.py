"""Enhanced memory - long-term memory, facts, context."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.memory.store import MemoryStore, _utc_now_iso


class EnhancedMemory(MemoryStore):
    def _init_schema(self) -> None:
        super()._init_schema()
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                content TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    def set_fact(self, key: str, value: Any) -> None:
        self._conn.execute(
            "INSERT INTO facts(key,value,updated_at) VALUES (?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, json.dumps(value, default=str), _utc_now_iso()),
        )
        self._conn.commit()

    def get_fact(self, key: str) -> Any | None:
        cur = self._conn.execute("SELECT value FROM facts WHERE key=?", (key,))
        r = cur.fetchone()
        if not r:
            return None
        try:
            return json.loads(r["value"])
        except Exception:
            return r["value"]

    def list_facts(self) -> list[tuple[str, Any]]:
        cur = self._conn.execute("SELECT key,value FROM facts ORDER BY key")
        out = []
        for r in cur.fetchall():
            try:
                out.append((r["key"], json.loads(r["value"])))
            except:
                out.append((r["key"], r["value"]))
        return out
