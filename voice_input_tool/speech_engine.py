"""ASR and VAD engine setup."""

import os
import sys

import sherpa_onnx

from voice_input_tool.app_paths import MODEL_DIR, VAD_MODEL
from voice_input_tool.audio_constants import SAMPLE_RATE


def create_recognizer(sample_rate=SAMPLE_RATE):
    """Create the ReazonSpeech Zipformer recognizer."""
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


def create_vad(threshold, silence_duration, min_speech, sample_rate=SAMPLE_RATE):
    """Create the Silero VAD engine."""
    if not os.path.exists(VAD_MODEL):
        print(f"[ERROR] VADモデルが見つかりません: {VAD_MODEL}")
        sys.exit(1)

    vad_config = sherpa_onnx.VadModelConfig()
    vad_config.silero_vad.model = VAD_MODEL
    vad_config.silero_vad.threshold = threshold
    vad_config.silero_vad.min_silence_duration = silence_duration
    vad_config.silero_vad.min_speech_duration = min_speech
    vad_config.sample_rate = sample_rate
    vad_config.provider = "cpu"

    return sherpa_onnx.VoiceActivityDetector(vad_config)


def recognize_speech(recognizer, audio_samples, sample_rate=SAMPLE_RATE):
    """Convert float32 audio samples to text."""
    if len(audio_samples) < sample_rate * 0.1:
        return ""

    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, audio_samples)
    recognizer.decode_stream(stream)
    return stream.result.text.strip()
