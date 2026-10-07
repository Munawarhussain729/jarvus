"""Critical safety boundaries — fail closed, no shell from model text."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from app.assistant.router import ToolRouter
from app.audit.logger import AuditLogger
from app.memory.store import MemoryStore
from app.security.policy import RiskLevel, requires_user_confirmation
from app.security.sanitize import InputRejected, sanitize_user_message
from app.tools.registry import ToolRegistry
from app.tools.safe_shell import ALLOWED, SafeShellTool


def test_sanitize_rejects_nul() -> None:
    with pytest.raises(InputRejected):
        sanitize_user_message("hello\x00world")


def test_sanitize_rejects_oversized() -> None:
    with pytest.raises(InputRejected):
        sanitize_user_message("x" * 50_000)


def test_safe_shell_uses_allowlist_only() -> None:
    tool = SafeShellTool()
    r = tool.run({"command_id": "pwd"})
    assert r.ok is True
    bad = tool.run({"command_id": "rm -rf /"})
    assert bad.ok is False


def test_safe_shell_never_passes_string_to_shell() -> None:
    # argv must come from ALLOWED dict keys only
    assert "pwd" in ALLOWED
    argv = ALLOWED["pwd"]
    assert isinstance(argv, list)
    assert all(isinstance(x, str) for x in argv)


def test_requires_confirmation_for_high_risk() -> None:
    assert requires_user_confirmation(RiskLevel.HIGH) is True
    assert requires_user_confirmation(RiskLevel.LOW) is False


def test_router_unknown_tool_audited(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    store = MemoryStore(db)
    audit = AuditLogger(store)
    router = ToolRouter(ToolRegistry.default(), audit)
    out = router.handle_tool_call(tool_name="not_a_real_tool", arguments_json="{}")
    assert "Unknown" in out or "disallowed" in out
    store.close()
    con = sqlite3.connect(db)
    try:
        cur = con.execute("SELECT COUNT(*) FROM audit_log")
        assert cur.fetchone()[0] >= 1
    finally:
        con.close()


def test_router_high_risk_denied_without_yes(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    store = MemoryStore(db)
    audit = AuditLogger(store)
    router = ToolRouter(ToolRegistry.default(), audit)
    args = json.dumps({"command_id": "pwd"})
    with patch("app.assistant.router.confirm_risky_action", return_value=False):
        out = router.handle_tool_call(tool_name="safe_shell", arguments_json=args)
    assert "cancelled" in out.lower() or "denied" in out.lower()
    store.close()
