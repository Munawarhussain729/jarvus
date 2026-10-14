"""Messaging tool using macOS Messages app."""
from __future__ import annotations

from typing import Any

from app.security.policy import RiskLevel
from app.security.sanitize import sanitize_tool_argument_string
from app.tools.applescript import escape, run
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
            recipient = escape(sanitize_tool_argument_string(arguments["recipient"], field_name="recipient"))
            text = escape(sanitize_tool_argument_string(arguments["text"], field_name="text"))
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))

        script = f'''
        tell application "Messages"
            set targetService to 1st account whose service type = iMessage
            set targetBuddy to participant "{recipient}" of targetService
            send "{text}" to targetBuddy
        end tell'''
        result = run(script, ok_payload=f"Message sent to {arguments['recipient']}")
        return result


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
            to_addr = escape(sanitize_tool_argument_string(arguments["to"], field_name="to"))
            subject = escape(sanitize_tool_argument_string(arguments["subject"], field_name="subject"))
            body = escape(sanitize_tool_argument_string(arguments["body"], field_name="body"))
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
        return run(script, ok_payload=f"Email sent to {arguments['to']}")
