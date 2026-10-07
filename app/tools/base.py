"""Tool abstraction: structured args only — no raw model text to shell."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.security.policy import RiskLevel


@dataclass(frozen=True)
class ToolSpec:
    """Metadata exposed to the model (OpenAI function schema)."""

    name: str
    description: str
    parameters_schema: dict[str, Any]
    risk: RiskLevel = RiskLevel.LOW


@dataclass
class ToolResult:
    ok: bool
    payload: str
    detail: dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    """A callable capability registered in the allowlisted registry."""

    spec: ToolSpec

    @abstractmethod
    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute after router validation and policy checks."""
