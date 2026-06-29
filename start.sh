#!/bin/bash
cd ~/voice-input-tool

APP_LAUNCHER="$HOME/Applications/VoiceInputTool.app/Contents/MacOS/VoiceInputTool"
if [ -x "$APP_LAUNCHER" ] && [ $# -eq 0 ]; then
    exec "$APP_LAUNCHER"
fi

source .venv-framework/bin/activate
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi
python3 voice_input.py "$@"
