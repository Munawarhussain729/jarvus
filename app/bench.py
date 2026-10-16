"""Repeatable benchmark so 'is it faster?' has numbers, not vibes.

  python -m app.bench           deterministic paths only (no model needed)
  python -m app.bench --live    also times real LLM turns against Ollama

Measures: intent fast-path dispatch, STT hallucination filtering on synthetic
audio, and (with --live) warm/cold LLM latency + turn wall time. Results are
appended to metrics.jsonl so --doctor can show trends.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def _bench_intents() -> list[tuple[str, float, bool]]:
    from app.assistant.intents import match
    cases = [
        ("set a timer for 3 minutes", True),
        ("what's 15 percent of 80", True),
        ("convert 5 miles to kilometers", True),
        ("good morning", True),
        ("volume up", True),
        ("open safari", True),
        ("dark mode", True),
        ("remind me to call mom at 5pm", True),
        ("tell me a story about dragons", False),   # must NOT fast-path
        ("explain how transformers work", False),
    ]
    rows = []
    for text, expect_hit in cases:
        t0 = time.perf_counter()
        hit = match(text)
        ms = (time.perf_counter() - t0) * 1000
        ok = (hit is not None) == expect_hit
        rows.append((text, ms, ok))
    return rows


def _bench_stt() -> list[tuple[str, float, bool]]:
    from app.voice.stt import _is_hallucination, has_speech, SAMPLE_RATE
    rng = np.random.default_rng(0)
    silence = rng.normal(0, 0.0005, SAMPLE_RATE * 2).astype(np.float32)
    click = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
    click[SAMPLE_RATE // 2: SAMPLE_RATE // 2 + 800] = 0.3  # one loud transient
    voiced = np.tile(rng.normal(0, 0.02, int(SAMPLE_RATE * 0.3)),
                     7)[: SAMPLE_RATE * 2].astype(np.float32)
    rows = []
    for name, arr, expect in (
        ("silence 2s", silence, False),
        ("loud click", click, False),
        ("voiced noise 2s", voiced, True),
    ):
        t0 = time.perf_counter()
        got = has_speech(arr, rms_gate=0.003)
        rows.append((f"vad:{name}", (time.perf_counter() - t0) * 1000, got == expect))
    for text, expect in (
        ("Here we go.", True),
        ("find all of this link " * 40, True),
        ("Like that.", True),
        ("make a snake game in python", False),
        ("set a timer for 5 minutes", False),
    ):
        t0 = time.perf_counter()
        got = _is_hallucination(text)
        rows.append((f"filter:{text[:28]!r}", (time.perf_counter() - t0) * 1000, got == expect))
    return rows


def _bench_live(model: str | None) -> list[tuple[str, float, bool]]:
    from app.bootstrap import build_agent
    from app.config import load_settings
    agent, _memory = build_agent(load_settings(), model=model)
    rows = []
    prompts = [
        ("cold/warm llm", "Say hi in one word."),
        ("tool call", "what's 12 * 7"),  # calculate tool
        ("code gen", "write a python script that prints a multiplication table"),
    ]
    for name, prompt in prompts:
        t0 = time.monotonic()
        try:
            reply = agent.process_user_text(prompt)
            ms = (time.monotonic() - t0) * 1000
            rows.append((f"live:{name}", ms, bool(reply)))
        except Exception as e:
            rows.append((f"live:{name}", (time.monotonic() - t0) * 1000, False))
            print(f"    error: {e}", file=sys.stderr)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Jarvus benchmark")
    p.add_argument("--live", action="store_true", help="also run real LLM turns")
    p.add_argument("--model", default=None)
    args = p.parse_args()

    print(f"{'case':44s} {'ms':>10s}  ok")
    print("-" * 62)
    for section, rows in (("intents", _bench_intents()), ("stt", _bench_stt())):
        print(f"[{section}]")
        for name, ms, ok in rows:
            print(f"  {name:42s} {ms:10.1f}  {'PASS' if ok else 'FAIL'}")
    if args.live:
        print("[live]")
        for name, ms, ok in _bench_live(args.model):
            print(f"  {name:42s} {ms:10.1f}  {'PASS' if ok else 'FAIL'}")
    try:
        from app.telemetry import record
        record("bench", live=args.live)
    except Exception:
        pass


if __name__ == "__main__":
    main()
