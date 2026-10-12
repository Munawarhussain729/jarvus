"""Music control tool using macOS AppleScript."""
from __future__ import annotations

import subprocess
from typing import Any

from app.security.policy import RiskLevel
from app.security.sanitize import sanitize_tool_argument_string
from app.tools.base import Tool, ToolResult, ToolSpec


class MusicTool(Tool):
    spec = ToolSpec(
        name="music",
        description="Control Apple Music: play, pause, next, previous, volume, now playing.",
        parameters_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["play", "pause", "next", "previous", "volume", "now_playing", "search_play"],
                    "description": "Music action to perform",
                },
                "value": {"type": "string", "description": "Volume (0-100) or search query"},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        action = arguments.get("action", "")
        value = arguments.get("value", "")

        scripts = {
            "play": 'tell application "Music" to play',
            "pause": 'tell application "Music" to pause',
            "next": 'tell application "Music" to next track',
            "previous": 'tell application "Music" to previous track',
            "now_playing": '''
                tell application "Music"
                    if player state is playing then
                        return (name of current track) & " by " & (artist of current track)
                    else
                        return "Nothing is playing"
                    end if
                end tell
            ''',
        }

        if action == "volume":
            try:
                vol = int(value) if value else 50
                vol = max(0, min(100, vol))
            except ValueError:
                return ToolResult(ok=False, payload="Volume must be 0-100")
            scripts["volume"] = f'set volume output volume {vol}'

        if action == "search_play":
            try:
                query = sanitize_tool_argument_string(value, field_name="value")
            except Exception as e:
                return ToolResult(ok=False, payload=str(e))
            scripts["search_play"] = f'''
                tell application "Music"
                    set results to (every track whose name contains "{query}")
                    if (count of results) > 0 then
                        play item 1 of results
                        return "Playing: " & (name of item 1 of results)
                    else
                        return "No tracks found for: {query}"
                    end if
                end tell
            '''

        if action not in scripts:
            return ToolResult(ok=False, payload=f"Unknown action: {action}")

        try:
            proc = subprocess.run(
                ["osascript", "-e", scripts[action]],
                capture_output=True,
                text=True,
                timeout=15,
            )
            output = proc.stdout.strip()
            if proc.returncode == 0:
                return ToolResult(ok=True, payload=output or f"Music {action} done")
            return ToolResult(ok=False, payload=proc.stderr.strip() or "Music control failed")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
