"""Backwards-compatible alias - voice mode now lives in python -m app.main --voice."""
from __future__ import annotations

import sys

from app.main import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--voice", *[a for a in sys.argv[1:] if a != "--voice"]]
    main()
