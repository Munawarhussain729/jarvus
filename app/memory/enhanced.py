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
            CREATE TABLE IF NOT EXISTS todos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT UNIQUE NOT NULL,
                status TEXT NOT NULL,
                priority TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
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

    def delete_fact(self, key: str) -> bool:
        cur = self._conn.execute("DELETE FROM facts WHERE key=?", (key,))
        self._conn.commit()
        return cur.rowcount > 0

    def facts_prompt_block(self, max_facts: int = 50) -> str:
        """Render stored facts as a system-prompt section."""
        facts = self.list_facts()[:max_facts]
        if not facts:
            return ""
        lines = ["KNOWN FACTS about the user (use them naturally):"]
        lines += [f"- {k}: {v}" for k, v in facts]
        return "\n".join(lines)

    # ---- Internal meta key/value (never injected into the prompt) ----

    def get_meta(self, key: str) -> str | None:
        cur = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,))
        r = cur.fetchone()
        return r["value"] if r else None

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self._conn.commit()

    # ---- Persistent todo list ----

    def set_todos(self, todos: list[dict[str, Any]]) -> None:
        """Replace the whole list (mirrors the model's whole-array semantics)."""
        now = _utc_now_iso()
        self._conn.execute("DELETE FROM todos")
        self._conn.executemany(
            "INSERT INTO todos(content,status,priority,updated_at) VALUES (?,?,?,?)",
            [(str(t["content"]), str(t["status"]), str(t["priority"]), now) for t in todos],
        )
        self._conn.commit()

    def add_todo(self, content: str, status: str = "pending", priority: str = "medium") -> None:
        self._conn.execute(
            "INSERT INTO todos(content,status,priority,updated_at) VALUES (?,?,?,?)",
            (content, status, priority, _utc_now_iso()),
        )
        self._conn.commit()

    def complete_todo(self, content_contains: str) -> int:
        """Mark matching pending/in_progress todos completed. Returns rows touched."""
        cur = self._conn.execute(
            "UPDATE todos SET status='completed', updated_at=? "
            "WHERE status IN ('pending','in_progress') AND lower(content) LIKE ?",
            (_utc_now_iso(), f"%{content_contains.lower()}%"),
        )
        self._conn.commit()
        return cur.rowcount

    def list_todos(self, include_done: bool = False) -> list[dict[str, Any]]:
        q = "SELECT content,status,priority FROM todos"
        if not include_done:
            q += " WHERE status != 'completed'"
        q += " ORDER BY CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, id"
        cur = self._conn.execute(q)
        return [dict(r) for r in cur.fetchall()]

    # ---- Persistent plans ----

    def save_plan(self, name: str, content: str, status: str = "active") -> None:
        now = _utc_now_iso()
        self._conn.execute(
            "INSERT INTO plans(name,content,status,created_at,updated_at) VALUES (?,?,?,?,?)",
            (name, content, status, now, now),
        )
        self._conn.commit()
