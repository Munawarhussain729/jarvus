#!/bin/bash
cd /Users/mac/Projects/Jarvus
source .venv/bin/activate
python -m app.main_voice_full --voice --no-speak --fast "$@"
