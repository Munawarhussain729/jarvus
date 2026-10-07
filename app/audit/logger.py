"""Structured audit trail for every tool execution (reasoning vs execution boundary)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.memory.store import MemoryStore


@dataclass
class ToolAuditRecord:
    tool_name: str
    arguments: dict[str, Any]
    ok: bool
    outcome_summary: str
    detail: dict[str, Any]


class AuditLogger:
    """Writes one row per tool invocation; no silent failures that skip logging."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    def log_tool_result(self, record: ToolAuditRecord) -> None:
        self._store.append_audit_row(
            tool_name=record.tool_name,
            arguments=record.arguments,
            ok=record.ok,
            outcome_summary=record.outcome_summary,
            detail=record.detail,
        )
