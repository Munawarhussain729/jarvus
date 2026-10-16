"""Append-only timing metrics -> <data_dir>/metrics.jsonl.

Every call is best-effort: telemetry must never break a voice turn.
`--doctor` reads the tail of this file to show recent latency stats.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

_PATH = Path(os.environ.get("JARVUS_DATA_DIR", "~/.jarvus")).expanduser() / "metrics.jsonl"


def record(event: str, **fields) -> None:
    try:
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        with _PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": round(time.time(), 3), "event": event, **fields}) + "\n")
    except OSError:
        pass


def tail(n: int = 50) -> list[dict]:
    try:
        lines = _PATH.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
        return [json.loads(l) for l in lines if l.strip()]
    except (OSError, json.JSONDecodeError):
        return []
