#!/usr/bin/env python3
"""
Voice Input Tool - ReazonSpeech ASR + Silero VAD + Optional LLM Correction
Mac Mini (Apple Silicon) 向け ローカル音声入力ツール

使い方:
  python3 voice_input.py              # デフォルト: LLM補正なし
  python3 voice_input.py --llm        # LLM句読点補正あり (OpenRouter GLM 5.2)
  python3 voice_input.py --test       # テストWAVで動作確認

ホットキー:
  Ctrl+Shift+Space : 録音開始/停止トグル
  Ctrl+Shift+Q     : 終了
"""

import argparse
import os
import sys
import time
import wave
import numpy as np
import threading
import queue
import pyperclip
from pathlib import Path

# sherpa-onnx
import sherpa_onnx

# 音声入力
import sounddevice as sd

# オプション: キーボードホットキー
try:
    from pynput import keyboard
    HAS_HOTKEY = True
except ImportError:
    HAS_HOTKEY = False

# オプション: LLM補正
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# ============================================================
# 設定
# ============================================================

MODEL_DIR = os.path.expanduser(
    "~/voice-input-tool/models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01"
)
VAD_MODEL = os.path.expanduser("~/voice-input-tool/models/silero_vad.onnx")

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 512  # 32ms at 16kHz

# VAD パラメータ
VAD_THRESHOLD = 0.5      # 発話検出閾値
VAD_SILENCE_DURATION = 0.8  # 無音で発話終了と判断する秒数
VAD_MIN_SPEECH = 0.3       # 最小発話長（秒）

# LLM補正プロンプト
LLM_PROMPT = """以下の音声認識結果に、句読点のみを挿入してください。
文章の変更・言い換え・要約は一切しないでください。
フィラー（あー、えー、まー）の削除のみ許可します。
話し言葉を書き言葉に変換しないでください。
出力は補正後のテキストのみとし、説明やコメントは一切付けないでください。"""


# ============================================================
# ASRエンジン初期化
# ============================================================

def create_recognizer():
    """ReazonSpeech Zipformer モデルでASRエンジンを作成"""
    int8_encoder = os.path.join(MODEL_DIR, "encoder-epoch-99-avg-1.int8.onnx")
    int8_decoder = os.path.join(MODEL_DIR, "decoder-epoch-99-avg-1.int8.onnx")
    int8_joiner = os.path.join(MODEL_DIR, "joiner-epoch-99-avg-1.int8.onnx")
    tokens = os.path.join(MODEL_DIR, "tokens.txt")

    if not os.path.exists(int8_encoder):
        print(f"[ERROR] モデルファイルが見つかりません: {int8_encoder}")
        print("モデルをダウンロードしてください:")
        print("  cd ~/voice-input-tool/models")
        print("  curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2")
        print("  tar xjf sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2")
        sys.exit(1)

    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=int8_encoder,
        decoder=int8_decoder,
        joiner=int8_joiner,
        tokens=tokens,
        num_threads=4,
        sample_rate=SAMPLE_RATE,
        decoding_method="greedy_search",
        provider="cpu",
    )
    return recognizer


def create_vad():
    """Silero VADエンジンを作成"""
    if not os.path.exists(VAD_MODEL):
        print(f"[ERROR] VADモデルが見つかりません: {VAD_MODEL}")
        print("モデルをダウンロードしてください:")
        print("  curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx")
        sys.exit(1)

    vad_config = sherpa_onnx.VadModelConfig()
    vad_config.silero_vad.model = VAD_MODEL
    vad_config.silero_vad.threshold = VAD_THRESHOLD
    vad_config.silero_vad.min_silence_duration = VAD_SILENCE_DURATION
    vad_config.silero_vad.min_speech_duration = VAD_MIN_SPEECH
    vad_config.sample_rate = SAMPLE_RATE
    vad_config.provider = "cpu"

    vad = sherpa_onnx.VoiceActivityDetector(vad_config)
    return vad


# ============================================================
# 音声認識
# ============================================================

def recognize_speech(recognizer, audio_samples):
    """音声サンプルをテキストに変換"""
    if len(audio_samples) < SAMPLE_RATE * 0.1:  # 0.1秒未満は無視
        return ""

    stream = recognizer.create_stream()
    stream.accept_waveform(SAMPLE_RATE, audio_samples)
    recognizer.decode_stream(stream)
    return stream.result.text.strip()


# ============================================================
# LLM補正
# ============================================================

def llm_correct(text, api_key=None):
    """OpenRouter GLM 5.2 で句読点補正"""
    if not HAS_OPENAI or not text:
        return text

    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("[WARN] OPENROUTER_API_KEY が設定されていません。LLM補正をスキップします。")
        return text

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
    )

    try:
        response = client.chat.completions.create(
            model="z-ai/glm-5.2",
            messages=[
                {"role": "system", "content": LLM_PROMPT},
                {"role": "user", "content": text},
            ],
            max_tokens=2048,
            temperature=0.1,
            extra_body={
                "data_collection": "deny",
                "zdr": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        corrected = response.choices[0].message.content
        if corrected is None:
            print("[WARN] LLMが空レスポンスを返しました。元テキストを使用します。")
            return text
        corrected = corrected.strip()
        return corrected if corrected else text
    except Exception as e:
        print(f"[WARN] LLM補正エラー: {e}")
        return text


# ============================================================
# 出力
# ============================================================

def output_text(text, mode="clipboard"):
    """テキストをクリップボードにコピー＆ペースト"""
    if not text:
        return

    print(f"\n[認識結果] {text}")
    
    if mode == "clipboard":
        pyperclip.copy(text)
        print("[クリップボードにコピーしました]")
    elif mode == "stdout":
        print(text)


# ============================================================
# リアルタイム録音ループ
# ============================================================

class VoiceInputApp:
    def __init__(self, recognizer, vad, use_llm=False):
        self.recognizer = recognizer
        self.vad = vad
        self.use_llm = use_llm
        self.is_recording = False
        self.should_exit = False
        self.audio_queue = queue.Queue()
        self.speech_buffer = []
        self.in_speech = False

    def audio_callback(self, indata, frames, time_info, status):
        """音声入力コールバック"""
        if self.is_recording:
            samples = indata[:, 0].astype(np.float32)
            self.audio_queue.put(samples.copy())

    def process_audio(self):
        """音声処理スレッド: VADで発話区間を検出→ASR"""
        audio_buffer = np.array([], dtype=np.float32)

        while not self.should_exit:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            audio_buffer = np.concatenate([audio_buffer, chunk])

            # VADに1ブロックずつ流す
            # sherpa-onnx VADは512サンプル単位で処理
            while len(audio_buffer) >= BLOCK_SIZE:
                block = audio_buffer[:BLOCK_SIZE]
                audio_buffer = audio_buffer[BLOCK_SIZE:]

                self.vad.accept_waveform(block)

                while not self.vad.empty():
                    segment = self.vad.front
                    self.vad.pop()

                    speech_samples = segment.samples
                    if len(speech_samples) < SAMPLE_RATE * VAD_MIN_SPEECH:
                        continue

                    # ASR実行
                    start = time.time()
                    text = recognize_speech(self.recognizer, speech_samples)
                    elapsed = time.time() - start

                    if text:
                        print(f"\n[ASR {elapsed:.2f}s] {text}")
                        # LLM補正
                        if self.use_llm:
                            text = llm_correct(text)
                            print(f"[LLM補正] {text}")
                        
                        output_text(text)
                    else:
                        print(f"\n[ASR {elapsed:.2f}s] (認識結果なし)")

    def start_recording(self):
        """録音開始"""
        if self.is_recording:
            return
        self.is_recording = True
        print("\n[録音開始] 話してください... (Ctrl+Shift+Space で停止)")
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            blocksize=BLOCK_SIZE,
            dtype="float32",
            callback=self.audio_callback,
        )
        self.stream.start()

        # 音声処理スレッド開始
        self.process_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.process_thread.start()

    def stop_recording(self):
        """録音停止"""
        if not self.is_recording:
            return
        self.is_recording = False
        self.stream.stop()
        self.stream.close()
        print("\n[録音停止]")

    def run(self):
        """メインループ"""
        if HAS_HOTKEY:
            print("=== Voice Input Tool ===")
            print(f"LLM補正: {'ON (GLM 5.2 via OpenRouter)' if self.use_llm else 'OFF'}")
            print("ホットキー:")
            print("  Ctrl+Shift+Space : 録音開始/停止")
            print("  Ctrl+Shift+Q     : 終了")
            print()

            def on_activate_record():
                if self.is_recording:
                    self.stop_recording()
                else:
                    self.start_recording()

            def on_activate_exit():
                self.should_exit = True
                self.stop_recording()
                return False  # listener停止

            with keyboard.GlobalHotKeys({
                '<ctrl>+<shift>+<space>': on_activate_record,
                '<ctrl>+<shift>+q': on_activate_exit,
            }) as listener:
                listener.join()
        else:
            # ホットキーなし: Enterで録音トグル
            print("=== Voice Input Tool (ホットキーなし) ===")
            print("Enterキーで録音開始/停止。Ctrl+Cで終了。")
            self.start_recording()
            try:
                while not self.should_exit:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                self.stop_recording()


# ============================================================
# テストモード
# ============================================================

def run_test(recognizer, use_llm=False):
    """テストWAVファイルでASRを検証"""
    test_dir = os.path.join(MODEL_DIR, "test_wavs")
    transcript_path = os.path.join(test_dir, "transcript.txt")

    # 正解テキスト読み込み
    transcripts = {}
    if os.path.exists(transcript_path):
        with open(transcript_path, "r") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    transcripts[parts[0]] = parts[1]

    print("=== テストモード ===\n")

    for i in range(1, 6):
        wav_path = os.path.join(test_dir, f"{i}.wav")
        if not os.path.exists(wav_path):
            continue

        # WAV読み込み
        with wave.open(wav_path, "rb") as wf:
            assert wf.getframerate() == SAMPLE_RATE, f"サンプリングレート不一致: {wf.getframerate()}"
            assert wf.getnchannels() == 1
            raw = wf.readframes(wf.getnframes())
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        # ASR
        start = time.time()
        text = recognize_speech(recognizer, samples)
        elapsed = time.time() - start

        expected = transcripts.get(f"{i}.wav", "")

        print(f"--- テスト {i} ({len(samples)/SAMPLE_RATE:.1f}s) ---")
        print(f"正解: {expected}")
        print(f"認識: {text}")
        print(f"時間: {elapsed:.2f}s")

        if use_llm:
            corrected = llm_correct(text)
            print(f"補正: {corrected}")
        print()

    print("=== テスト完了 ===")


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Voice Input Tool - ReazonSpeech ASR")
    parser.add_argument("--llm", action="store_true", help="LLM句読点補正を有効化 (OpenRouter GLM 5.2)")
    parser.add_argument("--test", action="store_true", help="テストWAVファイルで動作確認")
    args = parser.parse_args()

    print("モデル読み込み中...", end=" ", flush=True)
    start = time.time()
    recognizer = create_recognizer()
    vad = create_vad()
    print(f"完了 ({time.time()-start:.1f}s)")

    if args.test:
        run_test(recognizer, use_llm=args.llm)
    else:
        app = VoiceInputApp(recognizer, vad, use_llm=args.llm)
        app.run()


if __name__ == "__main__":
    main()
