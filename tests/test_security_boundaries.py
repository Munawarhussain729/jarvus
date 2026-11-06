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


# ---- release-hardening regressions -------------------------------------------------
# subprocess.run is replaced by a recorder in every shell test, so none of these
# commands can ever actually execute, whatever the code under test does.

from app.tools import shell_extended  # noqa: E402
from app.tools.filesystem import ReadFileTool, _validate_path  # noqa: E402
from app.tools.shell_extended import ShellReadOnlyTool  # noqa: E402


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))

        class _P:
            stdout = "ok"
            stderr = ""
            returncode = 0

        return _P()


@pytest.fixture
def recorder(monkeypatch) -> _Recorder:
    rec = _Recorder()
    monkeypatch.setattr(shell_extended.subprocess, "run", rec)
    return rec


@pytest.mark.parametrize(
    "cmd",
    [
        "ls; rm -rf x",
        "echo pwned",
        "cat $(echo /etc/passwd)",
        "python3 -c print(1)",
        "python3 script.py",
        "node -e 1",
        "git config --get user.email",
        "git -c core.pager=sh log",
        "find . -exec echo hi ;",
        "find . -delete",
        "cat ~/.ssh/id_rsa",
        "head -n 5 ~/.aws/credentials",
    ],
)
def test_shell_read_rejects_unsafe_commands(cmd: str, recorder: _Recorder) -> None:
    r = ShellReadOnlyTool().run({"command": cmd})
    assert r.ok is False, cmd
    assert recorder.calls == [], f"command reached subprocess: {cmd}"


@pytest.mark.parametrize("cmd", ["ls && echo pwned", "ls | cat", "ls ; touch x"])
def test_shell_operators_are_literal_arguments_not_chained(cmd: str, recorder: _Recorder) -> None:
    ShellReadOnlyTool().run({"command": cmd})
    for args, kwargs in recorder.calls:
        assert kwargs.get("shell") is False
        assert args[0][0] == "ls"  # a single ls process; operators are inert arguments


def test_shell_read_never_uses_a_shell(recorder: _Recorder) -> None:
    r = ShellReadOnlyTool().run({"command": "pwd"})
    assert r.ok is True
    (args, kwargs), = recorder.calls
    assert kwargs.get("shell") is False
    assert args[0] == ["pwd"]


def test_shell_read_allows_version_checks(recorder: _Recorder) -> None:
    assert ShellReadOnlyTool().run({"command": "python3 --version"}).ok is True


@pytest.mark.parametrize(
    "rel",
    [".ssh/id_rsa", ".aws/credentials", ".gnupg/pubring.kbx", ".netrc", "project/.env", "project/.env.local",
     "Library/Keychains/login.keychain-db"],
)
def test_file_sandbox_blocks_credential_paths(rel: str) -> None:
    with pytest.raises(ValueError, match="protected"):
        _validate_path(str(Path.home() / rel))


def test_file_sandbox_still_allows_normal_home_files() -> None:
    assert _validate_path(str(Path.home() / "Documents" / "notes.txt")).name == "notes.txt"


def test_read_file_tool_refuses_env_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("JARVUS_FS_ROOTS", str(tmp_path))
    env = tmp_path / ".env"
    env.write_text("SECRET=1")
    assert ReadFileTool().run({"path": str(env)}).ok is False
