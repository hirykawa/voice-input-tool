"""Shared filesystem paths for the voice input app."""

from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent

LOG_DIR = APP_DIR / "logs"
MODEL_DIR = str(
    APP_DIR
    / "models"
    / "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01"
)
VAD_MODEL_PATH = str(APP_DIR / "models" / "silero_vad.onnx")

RUNTIME_DIR = APP_DIR / "logs"
COMMAND_FILE_PATH = str(RUNTIME_DIR / "voice-input-command.txt")
OUTPUT_FILE_PATH = str(RUNTIME_DIR / "voice-input-output.jsonl")
STATUS_FILE_PATH = str(RUNTIME_DIR / "voice-input-status.json")
NATIVE_PASTE_READY_PATH = str(RUNTIME_DIR / "native-paste-ready.txt")

TYPING_INDICATOR_ICON_FRAMES = tuple(
    str(APP_DIR / "assets" / f"typing-indicator-{index}.svg")
    for index in range(6)
)
