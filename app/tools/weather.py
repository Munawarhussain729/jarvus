"""Weather tool - get current weather and forecasts."""
from __future__ import annotations

import json
import urllib.request
from typing import Any

from app.security.policy import RiskLevel
from app.security.sanitize import sanitize_tool_argument_string
from app.tools.base import Tool, ToolResult, ToolSpec


class WeatherTool(Tool):
    spec = ToolSpec(
        name="weather",
        description="Get current weather for a city (empty = auto-detect location). Uses wttr.in, no key.",
        parameters_schema={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name (e.g., 'London', 'Tokyo'); "
                                                          "empty = auto-detect from IP", "default": ""},
                "format": {"type": "string", "enum": ["short", "full"], "default": "short"},
            },
            "additionalProperties": False,
        },
        risk=RiskLevel.LOW,
    )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            city = sanitize_tool_argument_string(str(arguments.get("city") or ""), field_name="city").strip()
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))

        fmt = arguments.get("format", "short")
        from app.tools._cache import get as cache_get, put as cache_put
        key = f"weather:{city.lower()}:{fmt}"
        cached = cache_get(key)
        if cached is not None:
            return ToolResult(ok=True, payload=cached)
        try:
            if fmt == "full":
                url = f"https://wttr.in/{city}?format=j1"
            else:
                url = f"https://wttr.in/{city}?format=3"

            req = urllib.request.Request(url, headers={"User-Agent": "curl/8 (weather)"})
            with urllib.request.urlopen(req, timeout=12) as resp:
                output = resp.read().decode("utf-8", "replace").strip()
            if not output:
                return ToolResult(ok=False, payload="No weather data received")

            if fmt == "full" and output.startswith("{"):
                data = json.loads(output)
                current = data.get("current_condition", [{}])[0]
                temp_c = current.get("temp_C", "?")
                temp_f = current.get("temp_F", "?")
                desc = current.get("weatherDesc", [{}])[0].get("value", "?")
                humidity = current.get("humidity", "?")
                wind = current.get("windspeedKmph", "?")
                city_name = data.get("nearest_area", [{}])[0].get("areaName", [{}])[0].get("value", city)
                output = f"Weather in {city_name}: {desc}, {temp_c}°C ({temp_f}°F), humidity {humidity}%, wind {wind} km/h"

            cache_put(key, output, ttl_s=300)
            return ToolResult(ok=True, payload=output)
        except TimeoutError:
            return ToolResult(ok=False, payload="Weather request timed out")
        except Exception as e:
            return ToolResult(ok=False, payload=str(e))
