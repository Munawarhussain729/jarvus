"""Jarvus - unified entry point. Local-first by default (Ollama + local STT/TTS).

Modes:
  python -m app.main                          text REPL, local model
  python -m app.main --voice                  continuous voice control
  python -m app.main --voice --wake-word      wake-word mode ("hey jarvis")
  python -m app.main --serve                  remote-access HTTP server
  python -m app.main --remote-url URL         voice/text client of a remote Jarvus server
  python -m app.main --openai                 use OpenAI instead of local model
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from app.bootstrap import build_agent
from app.config import load_settings
from app.security.confirm import confirm_risky_action


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _text_repl(agent, memory=None, *, auto_brief: bool = False) -> None:
    print("\nJarvus ready. Type 'exit' to quit.\n")
    if auto_brief:
        from app.tools.briefing import maybe_auto_brief
        digest = maybe_auto_brief(memory, enabled=True)
        if digest:
            print(f"Jarvus> {digest}\n")
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
            print(f"\nJarvus> {agent.process_user_text(line)}\n")
        except Exception as e:
            logging.exception("error")
            print(f"[error] {e}", file=sys.stderr)


def _run_voice(settings, agent, *, wake_word: bool, speak: bool, auto_brief: bool = False) -> None:
    from app.voice.listener import VoiceListener
    from app.voice.session import JarvusVoice
    from app.voice.tts import get_tts

    tts = get_tts(engine=settings.tts_engine, voice=settings.tts_voice, rate=settings.tts_rate)
    listener = VoiceListener(
        wake_words=settings.wake_word_list,
        use_wake_word=wake_word,
        stt_engine=settings.stt_engine,
        stt_model=settings.stt_model,
    )
    voice = JarvusVoice(agent, listener=listener, tts=tts, always_speak=speak, auto_brief=auto_brief)
    # Route HIGH-risk confirmations through voice instead of stdin (local runs only).
    router = getattr(agent, "_router", None)
    if router is not None:
        router.set_confirmer(voice.voice_confirmer)

    print("\nVoice mode" + (" (wake word: " + ", ".join(settings.wake_word_list) + ")" if wake_word else " (continuous)"))
    print("Say 'go to sleep' to stop. Ctrl-C to exit.\n")
    voice.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        listener.stop()


def main() -> None:
    p = argparse.ArgumentParser(description="Jarvus - local-first AI assistant")
    p.add_argument("--voice", action="store_true", help="enable voice input/output")
    p.add_argument("--wake-word", action="store_true", help="require wake word before commands")
    p.add_argument("--no-speak", action="store_true", help="voice input only, no spoken replies")
    p.add_argument("--openai", action="store_true", help="use OpenAI instead of the local model")
    p.add_argument("--model", default=None, help="model id (Ollama name or OpenAI id)")
    p.add_argument("--base-url", default=None, help="LLM server URL (default http://localhost:11434)")
    p.add_argument("--serve", action="store_true", help="run remote-access HTTP server")
    p.add_argument("--host", default=None, help="server bind host")
    p.add_argument("--port", type=int, default=None, help="server bind port")
    p.add_argument("--remote-url", default=None, help="talk to a remote Jarvus server instead of running locally")
    args = p.parse_args()

    settings = load_settings()
    _setup_logging(settings.log_level)

    if args.serve:
        from app.server import run_server
        run_server(
            settings,
            host=args.host or settings.server_host,
            port=args.port or settings.server_port,
            use_openai=args.openai,
            model=args.model,
            base_url=args.base_url,
        )
        return

    if args.remote_url:
        from app.server import RemoteJarvus
        agent = RemoteJarvus(args.remote_url, token=settings.server_token)
        memory = None
    else:
        agent, memory = build_agent(
            settings,
            use_openai=args.openai,
            model=args.model,
            base_url=args.base_url,
        )
        print(f"Model: {args.model or (settings.model if args.openai else settings.llm_model)}"
              f"{' (openai)' if args.openai else ' (local)'} | DB: {settings.resolved_db_path()}")

    auto_brief = settings.auto_brief.lower() == "on" or (
        settings.auto_brief.lower() == "auto" and args.voice
    )
    try:
        if args.voice:
            _run_voice(settings, agent, wake_word=args.wake_word, speak=not args.no_speak,
                       auto_brief=auto_brief)
        else:
            _text_repl(agent, memory, auto_brief=auto_brief)
    finally:
        if memory is not None:
            memory.close()


if __name__ == "__main__":
    main()
