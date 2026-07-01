"""ASR engine setup."""

import os
import sys

from voice_input_tool.app_paths import MODEL_DIR, VAD_MODEL_PATH
from voice_input_tool.audio_constants import SAMPLE_RATE

# 発話区切り検出（VAD）のチューニング
VAD_THRESHOLD = 0.5
VAD_MIN_SILENCE_SECONDS = 0.5
VAD_MIN_SPEECH_SECONDS = 0.25
# 無音が来ないまま話し続けた場合でも、この秒数を超えたら区切りを確定する
VAD_MAX_SPEECH_SECONDS = 8.0


def create_recognizer(sample_rate=SAMPLE_RATE):
    """Create the ReazonSpeech Zipformer recognizer."""
    import sherpa_onnx

    int8_encoder = os.path.join(MODEL_DIR, "encoder-epoch-99-avg-1.int8.onnx")
    int8_decoder = os.path.join(MODEL_DIR, "decoder-epoch-99-avg-1.int8.onnx")
    int8_joiner = os.path.join(MODEL_DIR, "joiner-epoch-99-avg-1.int8.onnx")
    tokens = os.path.join(MODEL_DIR, "tokens.txt")

    missing = [p for p in [int8_encoder, int8_decoder, int8_joiner, tokens] if not os.path.exists(p)]
    if missing:
        print("[ERROR] 以下のモデルファイルが見つかりません:\n" + "\n".join(f" - {m}" for m in missing))
        print("README の 'モデルダウンロード' 手順を実行してください。")
        sys.exit(1)

    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=int8_encoder,
        decoder=int8_decoder,
        joiner=int8_joiner,
        tokens=tokens,
        num_threads=4,
        sample_rate=sample_rate,
        decoding_method="greedy_search",
        provider="cpu",
    )

def create_vad(sample_rate=SAMPLE_RATE):
    """Create the Silero VAD used to detect natural speech boundaries."""
    import sherpa_onnx

    if not os.path.exists(VAD_MODEL_PATH):
        print(f"[ERROR] VADモデルが見つかりません: {VAD_MODEL_PATH}")
        print("README の 'モデルダウンロード' 手順を実行してください。")
        sys.exit(1)

    config = sherpa_onnx.VadModelConfig(
        silero_vad=sherpa_onnx.SileroVadModelConfig(
            model=VAD_MODEL_PATH,
            threshold=VAD_THRESHOLD,
            min_silence_duration=VAD_MIN_SILENCE_SECONDS,
            min_speech_duration=VAD_MIN_SPEECH_SECONDS,
            max_speech_duration=VAD_MAX_SPEECH_SECONDS,
        ),
        sample_rate=sample_rate,
    )
    return sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=100)


def recognize_speech(recognizer, audio_samples, sample_rate=SAMPLE_RATE):
    """Convert float32 audio samples to text."""
    if len(audio_samples) < sample_rate * 0.1:
        return ""

    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, audio_samples)
    recognizer.decode_stream(stream)
    return stream.result.text.strip()
