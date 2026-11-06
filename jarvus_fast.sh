#!/bin/bash
cd "$(dirname "$0")" || exit 1
source .venv/bin/activate
python -m app.main --model qwen3:4b-instruct-2507-q4_K_M "$@"
