#!/bin/bash
cd /Users/mac/Projects/Jarvus
source .venv/bin/activate
python -m app.main --model qwen3:4b-instruct-2507-q4_K_M "$@"
