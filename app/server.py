"""Remote-access HTTP server - lets other devices talk to this Jarvus instance.

Endpoints (JSON):
  GET  /health            -> {"ok": true}
  POST /chat {"text": "…"} -> {"reply": "…"}

Auth: if JARVUS_SERVER_TOKEN is set, requests need `Authorization: Bearer <token>`.
Bind defaults to localhost; use --host 0.0.0.0 only behind a firewall/VPN (Tailscale).
"""
from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from app.bootstrap import build_agent
from app.config import Settings

log = logging.getLogger("jarvus.server")

# Per-request flag: POST /chat {"text": ..., "confirm": true} pre-approves
# HIGH-risk tools for that single request only.
_confirm_ctx = threading.local()


def _request_confirmer(tool_name: str, summary: str) -> bool:
    return bool(getattr(_confirm_ctx, "allow", False))


def make_handler(agent, token: str | None):
    class JarvusHandler(BaseHTTPRequestHandler):
        def _auth_ok(self) -> bool:
            if not token:
                return True
            return self.headers.get("Authorization", "") == f"Bearer {token}"

        def _send(self, code: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if not self._auth_ok():
                return self._send(401, {"error": "unauthorized"})
            if self.path == "/health":
                return self._send(200, {"ok": True})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._auth_ok():
                return self._send(401, {"error": "unauthorized"})
            if self.path != "/chat":
                return self._send(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                text = str(body.get("text", "")).strip()
            except Exception:
                return self._send(400, {"error": "invalid JSON body"})
            if not text:
                return self._send(400, {"error": "missing 'text'"})
            _confirm_ctx.allow = bool(body.get("confirm"))
            try:
                reply = agent.process_user_text(text)
            except Exception as e:
                log.exception("chat failed")
                return self._send(500, {"error": str(e)})
            finally:
                _confirm_ctx.allow = False
            self._send(200, {"reply": reply})

        def log_message(self, fmt, *args):  # route through logging, not stderr
            log.debug("%s - %s", self.address_string(), fmt % args)

    return JarvusHandler


def run_server(settings: Settings, *, host: str, port: int, use_openai: bool = False, model: str | None = None, base_url: str | None = None) -> None:
    agent, _memory = build_agent(
        settings, use_openai=use_openai, model=model, base_url=base_url,
        confirmer=_request_confirmer,
    )
    token = settings.server_token
    if not token and host not in ("127.0.0.1", "localhost", "::1"):
        log.warning("JARVUS_SERVER_TOKEN is not set; anyone who can reach %s:%d can chat", host, port)
    server = ThreadingHTTPServer((host, port), make_handler(agent, token))
    print(f"Jarvus server listening on http://{host}:{port} (POST /chat, GET /health)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


class RemoteJarvus:
    """Client-side stand-in for the agent when the brain runs on a server."""

    def __init__(self, url: str, token: str | None = None) -> None:
        self._url = url.rstrip("/")
        self._token = token

    def process_user_text(self, text: str) -> str:
        import httpx
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        r = httpx.post(f"{self._url}/chat", json={"text": text}, headers=headers, timeout=300.0)
        r.raise_for_status()
        return str(r.json().get("reply", ""))
