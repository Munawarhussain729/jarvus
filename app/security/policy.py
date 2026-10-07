"""Risk classification and allowlisting for tools (execution layer)."""

from __future__ import annotations

from enum import Enum


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def requires_user_confirmation(risk: RiskLevel) -> bool:
    """High-risk tools always require explicit confirmation before execution."""
    return risk is RiskLevel.HIGH


def is_tool_name_allowed(name: str, allowlist: frozenset[str]) -> bool:
    if not name or not isinstance(name, str):
        return False
    # Strict identifier-style names only
    if not name.replace("_", "").isalnum():
        return False
    return name in allowlist
