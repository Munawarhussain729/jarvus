"""Input validation and sanitization (fail closed)."""

from __future__ import annotations

import re

# Reasonable REPL bounds; adjust in one place.
MAX_USER_MESSAGE_CHARS = 32_000
MAX_TOOL_ARG_CHARS = 16_000


class InputRejected(ValueError):
    """Raised when user or tool input fails validation."""


def sanitize_user_message(text: str) -> str:
    if not isinstance(text, str):
        raise InputRejected("Message must be a string")
    if len(text) > MAX_USER_MESSAGE_CHARS:
        raise InputRejected("Message exceeds maximum length")
    if "\x00" in text:
        raise InputRejected("Message contains NUL")
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text):
        raise InputRejected("Message contains disallowed control characters")
    return text.strip()


def sanitize_tool_argument_string(value: str, *, field_name: str) -> str:
    if len(value) > MAX_TOOL_ARG_CHARS:
        raise InputRejected(f"Tool argument {field_name} too long")
    if "\x00" in value:
        raise InputRejected(f"Tool argument {field_name} contains NUL")
    return value
