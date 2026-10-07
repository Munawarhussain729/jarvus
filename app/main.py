"""CLI entrypoint — text-first REPL, no network listeners."""

from __future__ import annotations

import argparse
import logging
import sys

from app.assistant.core import AssistantSession
from app.assistant.router import ToolRouter
from app.audit.logger import AuditLogger
from app.config import load_settings
from app.memory.store import MemoryStore
from app.models.openai_provider import OpenAIProvider
from app.tools.registry import ToolRegistry


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Jarvus — Typhoon assistant CLI (local-first)")
    parser.add_argument(
        "--log-level",
        default=None,
        help="Override JARVUS_LOG_LEVEL (DEBUG, INFO, WARNING, ...)",
    )
    args = parser.parse_args()

    settings = load_settings()
    log_level = args.log_level or settings.log_level
    _setup_logging(log_level)
    log = logging.getLogger("jarvus.cli")

    registry = ToolRegistry.default()
    memory = MemoryStore(settings.resolved_db_path())
    audit = AuditLogger(memory)
    router = ToolRouter(registry, audit)

    try:
        api_key = settings.get_openai_api_key()
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    provider = OpenAIProvider(api_key=api_key, model=settings.model)
    session = AssistantSession(
        provider=provider,
        registry=registry,
        router=router,
        memory=memory,
    )

    print("Typhoon — type 'exit' or Ctrl-D to quit.")
    print(
        "API model:",
        settings.model,
        "(assistant name is Typhoon; set JARVUS_MODEL to change the backend)",
        "| DB:",
        settings.resolved_db_path(),
    )
    # TODO: Menu bar / SwiftUI or Tauri UI — keep CLI as the thin shell.
    # TODO: Local model provider (llama.cpp, MLX) implementing ModelProvider.

    try:
        while True:
            try:
                line = input("\nYou> ").strip()
            except EOFError:
                print()
                break
            if not line:
                continue
            if line.lower() in {"exit", "quit"}:
                break
            try:
                reply = session.process_user_text(line)
            except Exception:
                log.exception("Unhandled error during turn")
                print("[error] Something went wrong; see logs.", file=sys.stderr)
                continue
            print("\nTyphoon>", reply)
    finally:
        memory.close()


if __name__ == "__main__":
    main()
