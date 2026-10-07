"""Controlled shell execution - allow specific safe commands with args restrictions."""
from __future__ import annotations

import shlex
import subprocess
from typing import Any

from app.security.policy import RiskLevel
from app.security.sanitize import InputRejected, sanitize_tool_argument_string
from app.tools.base import Tool, ToolResult, ToolSpec

# Allow specific command patterns - very restrictive
ALLOWED_COMMANDS = {
    "ls": ["-la", "-l", "-a", "-lh", "-R", "--color=never"],
    "pwd": [],
    "date": ["-u", "+%Y-%m-%d", "+%H:%M:%S"],
    "echo": [],  # only echo what we pass, but validated
    "git": ["status", "log", "branch", "diff", "show", "--name-only", "--oneline", "-1", "-5", "-10", "--no-pager"],
    "cat": [],
    "head": ["-5", "-10", "-20", "-50"],
    "tail": ["-5", "-10", "-20", "-50"],
    "wc": ["-l", "-w", "-c"],
    "find": [],  # but we'll validate paths
    "python3": ["-m", "--version", "-c"],
    "pip": ["list", "--version"],
    "node": ["--version"],
    "npm": ["--version", "list"],
}
