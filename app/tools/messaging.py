"""Messaging tool using macOS Messages app."""
from __future__ import annotations

import subprocess
from typing import Any

from app.security.policy import RiskLevel
from app.security.sanitize import sanitize_tool_argument_string
from app.tools.base import Tool, ToolResult, ToolSpec


class MessageTool(Tool):
    spec = ToolSpec(
        name="message",
        description="Send a message via macOS Messages app (iMessage/SMS).",
        parameters_schema={
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "description": "Phone number, email, or contact name"},
                "text": {"type": "string", "description": "Message text to send"},
            },
            "required": ["recipient", "text"],
            "additionalProperties": False,
        },
        risk=RiskLevel.HIGH,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            recipient = sanitize_tool_argument_string(arguments["recipient"], field_name="recipient")
            text = sanitize_tool_argument_string(arguments["text"], field_name="text")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))

        script = f'''
        tell application "Messages"
            set targetService to 1st account whose service type = iMessage
            set targetBuddy to participant "{recipient}" of targetService
            send "{text}" to targetBuddy
        end tell'''

        try:
            proc = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if proc.returncode == 0:
                return ToolResult(ok=True, payload=f"Message sent to {recipient}")
            return ToolResult(ok=False, payload=proc.stderr.strip() or "Failed to send message")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))


class EmailTool(Tool):
    spec = ToolSpec(
        name="email",
        description="Send an email via macOS Mail app.",
        parameters_schema={
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject"},
                "body": {"type": "string", "description": "Email body text"},
            },
            "required": ["to", "subject", "body"],
            "additionalProperties": False,
        },
        risk=RiskLevel.HIGH,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            to_addr = sanitize_tool_argument_string(arguments["to"], field_name="to")
            subject = sanitize_tool_argument_string(arguments["subject"], field_name="subject")
            body = sanitize_tool_argument_string(arguments["body"], field_name="body")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))

        script = f'''
        tell application "Mail"
            set newMessage to make new outgoing message with properties {{subject:"{subject}", content:"{body}"}}
            tell newMessage
                make new to recipient at end of to recipients with properties {{address:"{to_addr}"}}
                send
            end tell
        end tell'''

        try:
            proc = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if proc.returncode == 0:
                return ToolResult(ok=True, payload=f"Email sent to {to_addr}")
            return ToolResult(ok=False, payload=proc.stderr.strip() or "Failed to send email")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
