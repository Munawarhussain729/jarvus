"""Full Jarvus agent CLI."""
from __future__ import annotations

import argparse
import logging
import sys

from app.assistant.agent import JarvusAgent
from app.assistant.router import ToolRouter
from app.audit.logger import AuditLogger
from app.config import load_settings
from app.memory.store import MemoryStore
from app.models.openai_provider import OpenAIProvider
from app.tools.registry import ToolRegistry, EchoTool
from app.tools.filesystem import ReadFileTool, WriteFileTool, ListDirTool, GlobTool
from app.tools.shell_extended import ShellReadOnlyTool
from app.tools.system import SystemInfoTool, PsTool
from app.tools.plan import PlanTool, TodoWriteTool
from app.tools.code import PythonExecTool
from app.tools.terminal import RunCommandTool
from app.tools.safe_shell import SafeShellTool
from app.tools.macos import OpenAppTool, SayTool
from app.tools.docx import CreateDocxTool


def build_full_registry():
    return ToolRegistry([
        EchoTool(),
        SafeShellTool(),
        ReadFileTool(),
        WriteFileTool(),
        ListDirTool(),
        GlobTool(),
        ShellReadOnlyTool(),
        SystemInfoTool(),
        PsTool(),
        PlanTool(),
        TodoWriteTool(),
        PythonExecTool(),
        RunCommandTool(),
        OpenAppTool(),
        SayTool(),
        CreateDocxTool(),
    ])


def main():
    parser = argparse.ArgumentParser(description="Jarvus - Full Agent")
    parser.add_argument("--local", action="store_true", help="Use local Ollama model")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--fast", action="store_true", help="Use faster smaller model")
    args = parser.parse_args()

    settings = load_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    registry = build_full_registry()
    memory = MemoryStore(settings.resolved_db_path())
    audit = AuditLogger(memory)
    router = ToolRouter(registry, audit)

    if args.local:
        try:
            from app.models.local_provider import LocalProvider
            if args.model:
                model = args.model
            elif args.fast:
                model = "qwen2.5:3b"
            else:
                model = "qwen3:4b"
            provider = LocalProvider(
                base_url=args.base_url,
                model=model,
                temperature=0.1,
                max_tokens=160 if args.fast else 384,
            )
            print(f"Using local model: {model}")
        except Exception as e:
            print(f"[warn] local provider failed: {e}, falling back to OpenAI", file=sys.stderr)
            api_key = settings.get_openai_api_key()
            provider = OpenAIProvider(api_key=api_key, model=args.model or settings.model)
    else:
        api_key = settings.get_openai_api_key()
        provider = OpenAIProvider(api_key=api_key, model=args.model or settings.model)

    agent = JarvusAgent(provider=provider, registry=registry, router=router, memory=memory)

    print("\n🤖 Jarvus (Typhoon) - Full Agent Ready")
    print("Type 'exit' to quit.")
    print(f"DB: {settings.resolved_db_path()}\n")

    try:
        while True:
            try:
                line = input("You> ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line.lower() in ("exit", "quit"):
                break
            try:
                res = agent.process_user_text(line)
                print(f"\nJarvus> {res}\n")
            except Exception as e:
                logging.exception("error")
                print(f"[error] {e}", file=sys.stderr)
    finally:
        memory.close()


if __name__ == "__main__":
    main()
