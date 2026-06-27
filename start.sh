#!/bin/bash
cd ~/voice-input-tool
source .venv/bin/activate
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi
python3 voice_input.py "$@"
