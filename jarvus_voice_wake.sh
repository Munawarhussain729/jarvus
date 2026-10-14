#!/bin/bash
cd /Users/mac/Projects/Jarvus
source .venv/bin/activate
python -m app.main --voice --wake-word "$@"
