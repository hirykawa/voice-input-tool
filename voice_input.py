#!/usr/bin/env python3
"""
Voice Input Tool - ReazonSpeech ASR + Silero VAD + Optional LLM Correction
Mac Mini (Apple Silicon) 向け ローカル音声入力ツール

メニューバーアイコンから操作:
  録音開始/停止、設定画面、終了
"""

import argparse
import os
import sys
import time
import wave
import numpy as np
import threading
import queue
import logging
import pyperclip
from pathlib import Path
from config import load_config, save_config

# ファイルログ設定
_log_dir = os.path.expanduser("~/voice-input-tool/logs")
os.makedirs(_log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(_log_dir, "voice-input.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("voice_input")

# sherpa-onnx
import sherpa_onnx

# 音声入力
import sounddevice as sd

# メニューバーアプリ
import rumps

# キーボードホットキー
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

# 設定ファイルから読み込み
APP_CONFIG = load_config()

VAD_THRESHOLD = APP_CONFIG["vad_threshold"]
VAD_SILENCE_DURATION = APP_CONFIG["vad_silence_duration"]
VAD_MIN_SPEECH = APP_CONFIG["vad_min_speech"]
LLM_PROMPT = APP_CONFIG["llm_prompt"]


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
    if len(audio_samples) < SAMPLE_RATE * 0.1:
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

    key = api_key or APP_CONFIG.get("openrouter_api_key") or os.environ.get("OPENROUTER_API_KEY")
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
            return text
        corrected = corrected.strip()
        return corrected if corrected else text
    except Exception as e:
        print(f"[WARN] LLM補正エラー: {e}")
        return text


# ============================================================
# メニューバーアプリ
# ============================================================

class VoiceInputApp(rumps.App):
    def __init__(self, recognizer, vad, use_llm=False):
        super().__init__(
            "Voice Input",
            icon=None,
            title="🎙",
            quit_button="終了",
        )
        self.recognizer = recognizer
        self.vad = vad
        self.use_llm = use_llm
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self._stream = None
        self._process_thread = None
        self._settings_ctrl = None

        self._hotkey_listener = None

        # メニュー構成
        hotkey_display = self._get_hotkey_display()
        self.record_button = rumps.MenuItem(
            f"録音開始 ({hotkey_display})", callback=self.toggle_recording
        )
        llm_label = "LLM補正: ON" if self.use_llm else "LLM補正: OFF"
        self.llm_status = rumps.MenuItem(llm_label)
        self.settings_button = rumps.MenuItem("設定...", callback=self.open_settings)

        self.menu = [
            self.record_button,
            None,
            self.llm_status,
            self.settings_button,
        ]

        # ホットキー登録
        self._register_hotkey()

    def _get_hotkey_display(self):
        """設定からホットキーの表示文字列を取得"""
        from settings_ui import hotkey_to_display
        hotkey = APP_CONFIG.get("hotkey_record", "<ctrl>+<shift>+<space>")
        return hotkey_to_display(hotkey)

    def _register_hotkey(self):
        """グローバルホットキーを登録"""
        if not HAS_HOTKEY:
            log.warning("pynput未インストール: ホットキー無効")
            return

        # 既存のリスナーを停止
        if self._hotkey_listener:
            self._hotkey_listener.stop()
            self._hotkey_listener = None

        hotkey = APP_CONFIG.get("hotkey_record", "<ctrl>+<shift>+<space>")
        log.info(f"ホットキー登録: {hotkey}")

        try:
            self._hotkey_listener = keyboard.GlobalHotKeys({
                hotkey: self.toggle_recording,
            })
            self._hotkey_listener.daemon = True
            self._hotkey_listener.start()
        except Exception as e:
            log.error(f"ホットキー登録エラー: {e}")

    def toggle_recording(self, sender=None):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        if self.is_recording:
            return
        self.is_recording = True
        self.title = "🔴"
        hotkey_display = self._get_hotkey_display()
        self.record_button.title = f"録音停止 ({hotkey_display})"
        log.info("録音開始")

        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                blocksize=BLOCK_SIZE,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
            log.info("オーディオストリーム開始")
        except Exception as e:
            log.error(f"オーディオストリーム開始エラー: {e}")
            self.is_recording = False
            self.title = "🎙"
            self.record_button.title = "録音開始"
            return

        self._process_thread = threading.Thread(target=self._process_audio, daemon=True)
        self._process_thread.start()

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        self.title = "🎙"
        hotkey_display = self._get_hotkey_display()
        self.record_button.title = f"録音開始 ({hotkey_display})"
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        log.info("録音停止")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            log.warning(f"オーディオステータス: {status}")
        if self.is_recording:
            samples = indata[:, 0].astype(np.float32)
            self.audio_queue.put(samples.copy())

    def _process_audio(self):
        audio_buffer = np.array([], dtype=np.float32)
        chunk_count = 0

        while self.is_recording:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            chunk_count += 1
            if chunk_count == 1:
                log.info("音声データ受信開始")
            audio_buffer = np.concatenate([audio_buffer, chunk])

            while len(audio_buffer) >= BLOCK_SIZE:
                block = audio_buffer[:BLOCK_SIZE]
                audio_buffer = audio_buffer[BLOCK_SIZE:]

                self.vad.accept_waveform(block)

                while not self.vad.empty():
                    segment = self.vad.front
                    speech_samples = np.array(segment.samples)
                    self.vad.pop()
                    duration = len(speech_samples) / SAMPLE_RATE
                    log.info(f"VAD検出: {duration:.1f}秒")

                    if len(speech_samples) < SAMPLE_RATE * VAD_MIN_SPEECH:
                        log.info("最小発話長未満、スキップ")
                        continue

                    start = time.time()
                    text = recognize_speech(self.recognizer, speech_samples)
                    elapsed = time.time() - start

                    if text:
                        log.info(f"ASR ({elapsed:.2f}s): {text}")
                        if self.use_llm:
                            text = llm_correct(text)
                            log.info(f"LLM補正: {text}")
                        pyperclip.copy(text)
                        rumps.notification(
                            "Voice Input",
                            "クリップボードにコピーしました",
                            text[:100],
                        )
                    else:
                        log.info(f"ASR ({elapsed:.2f}s): 認識結果なし")

    def open_settings(self, sender=None):
        from settings_ui import SettingsWindowController
        from Cocoa import NSApp
        self._settings_ctrl = SettingsWindowController.alloc().initWithCallback_(self._on_settings_saved)
        self._settings_ctrl.show()
        NSApp.activateIgnoringOtherApps_(True)

    def _on_settings_saved(self, new_config):
        global APP_CONFIG, VAD_THRESHOLD, VAD_SILENCE_DURATION, VAD_MIN_SPEECH, LLM_PROMPT
        APP_CONFIG = new_config
        self.use_llm = new_config["use_llm"]
        VAD_THRESHOLD = new_config["vad_threshold"]
        VAD_SILENCE_DURATION = new_config["vad_silence_duration"]
        VAD_MIN_SPEECH = new_config["vad_min_speech"]
        LLM_PROMPT = new_config["llm_prompt"]
        self.llm_status.title = "LLM補正: ON" if self.use_llm else "LLM補正: OFF"
        # ホットキー再登録
        self._register_hotkey()
        hotkey_display = self._get_hotkey_display()
        if self.is_recording:
            self.record_button.title = f"録音停止 ({hotkey_display})"
        else:
            self.record_button.title = f"録音開始 ({hotkey_display})"
        log.info(f"設定更新: LLM={'ON' if self.use_llm else 'OFF'}, "
                 f"ホットキー={new_config.get('hotkey_record')}")

    def quit_app(self):
        self.stop_recording()
        rumps.quit_application()


# ============================================================
# テストモード
# ============================================================

def run_test(recognizer, use_llm=False):
    """テストWAVファイルでASRを検証"""
    test_dir = os.path.join(MODEL_DIR, "test_wavs")
    transcript_path = os.path.join(test_dir, "transcript.txt")

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

        with wave.open(wav_path, "rb") as wf:
            assert wf.getframerate() == SAMPLE_RATE
            assert wf.getnchannels() == 1
            raw = wf.readframes(wf.getnframes())
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

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
    parser.add_argument("--llm", action="store_true", help="LLM句読点補正を有効化")
    parser.add_argument("--test", action="store_true", help="テストWAVファイルで動作確認")
    args = parser.parse_args()

    use_llm = args.llm or APP_CONFIG.get("use_llm", False)

    log.info("モデル読み込み開始")
    start = time.time()
    recognizer = create_recognizer()
    vad = create_vad()
    log.info(f"モデル読み込み完了 ({time.time()-start:.1f}s)")
    log.info(f"LLM補正: {'ON' if use_llm else 'OFF'}")

    if args.test:
        run_test(recognizer, use_llm=use_llm)
    else:
        app = VoiceInputApp(recognizer, vad, use_llm=use_llm)
        app.run()


if __name__ == "__main__":
    main()
