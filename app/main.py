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
import threading
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


def _doctor(settings) -> None:
    """Print effective (non-secret) config, backend reachability, and recent
    turn timings - the answer to 'why is it slow?' without guessing."""
    import importlib.util

    print("== effective settings ==")
    for f in ("profile", "llm_model", "llm_num_ctx", "llm_think", "llm_keep_alive",
              "max_tokens", "temperature", "stt_engine", "stt_model",
              "stt_no_speech_threshold", "stt_min_logprob", "stt_silence_rms",
              "tts_engine", "tts_voice", "tool_prefilter", "auto_brief"):
        print(f"  {f}: {getattr(settings, f)}")
    print(f"  data_dir: {settings.data_dir}")

    print("== ollama ==")
    try:
        import httpx
        r = httpx.get(f"{settings.llm_base_url}/api/tags", timeout=5)
        names = [m["name"] for m in r.json().get("models", [])]
        installed = settings.llm_model in names
        print(f"  reachable: yes | models: {len(names)} | configured model installed: {installed}")
        if not installed:
            print(f"  -> pull it: ollama pull {settings.llm_model}")
    except Exception as e:
        print(f"  reachable: no ({e})")

    print("== local backends ==")
    for mod, label in (("mlx_whisper", "MLX Whisper"), ("faster_whisper", "faster-whisper"),
                       ("mlx_audio", "Kokoro MLX TTS"), ("speech_recognition", "speech_recognition"),
                       ("pyaudio", "PyAudio")):
        print(f"  {label}: {'ok' if importlib.util.find_spec(mod) else 'missing'}")

    print("== recent turns (metrics.jsonl) ==")
    try:
        from app.telemetry import tail
        rows = [r for r in tail(200) if r.get("event") in ("turn", "turn_done")]
        if not rows:
            print("  (no recorded turns yet)")
        else:
            walls = [r["wall_ms"] for r in rows if r.get("event") == "turn_done" and "wall_ms" in r]
            stts = [r["stt_ms"] for r in rows if r.get("event") == "turn" and "stt_ms" in r]
            ttss = [r["tts_ms"] for r in rows if r.get("event") == "turn" and "tts_ms" in r]
            def _p(v): return f"p50={sorted(v)[len(v)//2]}ms max={max(v)}ms" if v else "n/a"
            print(f"  turns={len(walls)} llm {_p(walls)} | stt {_p(stts)} | tts {_p(ttss)}")
            outcomes = {}
            for r in rows:
                if r.get("event") == "turn_done":
                    outcomes[r.get("outcome", "?")] = outcomes.get(r.get("outcome", "?"), 0) + 1
            print(f"  outcomes: {outcomes}")
    except Exception as e:
        print(f"  metrics unavailable: {e}")


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
        stt_no_speech_threshold=settings.stt_no_speech_threshold,
        stt_min_logprob=settings.stt_min_logprob,
        silence_rms=settings.stt_silence_rms,
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
        voice.stop()
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
    p.add_argument("--doctor", action="store_true", help="print effective config + backend health, then exit")
    args = p.parse_args()

    settings = load_settings()
    _setup_logging(settings.log_level)

    if args.doctor:
        _doctor(settings)
        return

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
        # Warm the model on a background thread while STT/TTS load - the first
        # command shouldn't pay a 30-60s model-load penalty. Voice sessions pin
        # the model in RAM longer between commands.
        prov = getattr(agent, "_provider", None)
        warmup = getattr(prov, "warmup", None)
        if callable(warmup):
            threading.Thread(
                target=warmup,
                kwargs={"keep_alive": "60m"} if args.voice else {},
                daemon=True,
            ).start()

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
