#!/usr/bin/env python3
"""
Voice Input Tool - ReazonSpeech ASR + Silero VAD + Optional LLM Correction
Mac Mini (Apple Silicon) 向け ローカル音声入力ツール

メニューバーアイコンから操作:
  録音開始/停止、設定画面、終了
"""

import argparse
import concurrent.futures
import os
import signal
import sys
import time
import wave
import numpy as np
import threading
import queue
import logging
from app_paths import LOG_DIR, MODEL_DIR
from audio_constants import BLOCK_SIZE, CHANNELS, SAMPLE_RATE
from audio_devices import resolve_input_device
from audio_pipeline import (
    AudioActivityTracker,
    VadAudioHistory,
    drain_vad_segments,
    process_audio_queue,
)
from app_status import AppStatusController
from config import load_config, save_config
from llm_correction import configure_llm, llm_correct
from log_utils import truncate_for_log
from macos_text import (
    copy_to_clipboard,
    get_frontmost_application,
    insert_text_at_cursor,
    request_accessibility_permission,
    target_pid,
)
from native_bridge import (
    NativeCommandReader,
    ensure_bridge_files,
    native_paste_bridge_ready,
    write_output,
)
from notifications import notify_user
from speech_engine import create_recognizer, create_vad, recognize_speech
from status_bar_diagnostics import install_status_bar_diagnostics

# ファイルログ設定
_log_dir = str(LOG_DIR)
os.makedirs(_log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(_log_dir, "voice-input.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("voice_input")
_error_handler = logging.FileHandler(os.path.join(_log_dir, "voice-input-error.log"), encoding="utf-8")
_error_handler.setLevel(logging.ERROR)
_error_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logging.getLogger().addHandler(_error_handler)

# 音声入力
import sounddevice as sd

# メニューバーアプリ
import rumps

try:
    from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
    HAS_APPKIT_APPLICATION = True
except ImportError:
    HAS_APPKIT_APPLICATION = False

# キーボードホットキー
try:
    from pynput import keyboard
    HAS_HOTKEY = True
except ImportError:
    HAS_HOTKEY = False

try:
    from PyObjCTools import AppHelper
    HAS_APP_HELPER = True
except ImportError:
    HAS_APP_HELPER = False

# ============================================================
# 設定
# ============================================================

# 設定ファイルから読み込み
APP_CONFIG = load_config()

VAD_THRESHOLD = APP_CONFIG["vad_threshold"]
VAD_SILENCE_DURATION = APP_CONFIG["vad_silence_duration"]
VAD_MIN_SPEECH = APP_CONFIG["vad_min_speech"]
VAD_PRE_ROLL_DURATION = APP_CONFIG.get("vad_pre_roll_duration", 0.8)
configure_llm(APP_CONFIG)


def build_vad():
    return create_vad(VAD_THRESHOLD, VAD_SILENCE_DURATION, VAD_MIN_SPEECH)


# ============================================================
# メニューバーアプリ
# ============================================================

class _HeadlessMenuItem:
    def __init__(self, title=""):
        self.title = title


class VoiceInputApp(rumps.App):
    def __init__(self, recognizer, vad, use_llm=False, headless=False, native_output=False):
        self._headless = headless
        self._native_output = native_output
        if not headless:
            super().__init__(
                "Voice Input",
                icon=None,
                title="🎙",
                quit_button="終了",
            )
        else:
            self.title = "VI"
            self.icon = None
        self.recognizer = recognizer
        self.vad = vad
        self.use_llm = use_llm
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self._stream = None
        self._process_thread = None
        self._settings_ctrl = None
        self._target_app = None
        self._status_controller = None
        self._has_audio_started = False
        self._start_requested_at = None
        self._audio_activity = AudioActivityTracker()
        self._segment_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._vad_history = VadAudioHistory()

        self._hotkey_listener = None

        # メニュー構成
        hotkey_display = self._get_hotkey_display()
        llm_label = "LLM補正: ON" if self.use_llm else "LLM補正: OFF"
        if headless:
            self.record_button = _HeadlessMenuItem(f"録音開始 ({hotkey_display})")
            self.llm_status = _HeadlessMenuItem(llm_label)
            self.settings_button = _HeadlessMenuItem("設定...")
        else:
            self.record_button = rumps.MenuItem(
                f"録音開始 ({hotkey_display})", callback=self.toggle_recording
            )
            self.llm_status = rumps.MenuItem(llm_label, callback=self.toggle_llm)
            self.settings_button = rumps.MenuItem("設定...", callback=self.open_settings)

            self.menu = [
                self.record_button,
                None,
                self.llm_status,
                self.settings_button,
            ]

        call_after = AppHelper.callAfter if HAS_APP_HELPER else None
        timer_factory = lambda callback, interval: rumps.Timer(callback, interval)
        self._status_controller = AppStatusController(
            app=self,
            record_button=self.record_button,
            get_hotkey_display=self._get_hotkey_display,
            use_llm=lambda: self.use_llm,
            headless=headless,
            call_after=call_after,
            timer_factory=timer_factory,
        )

        # ホットキー登録
        self._register_hotkey()

    def _set_status(self, status, force=False):
        self._status_controller.set(status, force=force)

    def _stop_status_icon_animation(self):
        self._status_controller.stop_icon_animation()

    def _current_status(self):
        return self._status_controller.current()

    def _restore_recording_status(self):
        self._status_controller.restore_recording_status(self.is_recording)

    def _get_hotkey_display(self):
        """設定からホットキーの表示文字列を取得"""
        from settings_ui import hotkey_to_display
        hotkey = APP_CONFIG.get("hotkey_record", "<ctrl>+<shift>+<space>")
        return hotkey_to_display(hotkey)

    def _ensure_audio_stream(self):
        if self._stream is not None:
            return True

        try:
            input_device = resolve_input_device(APP_CONFIG.get("input_device_id", ""))
            if input_device is None:
                log.error("利用可能な入力マイクが見つかりません")
                return False

            self._stream = sd.InputStream(
                device=input_device,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                blocksize=BLOCK_SIZE,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
            device_label = sd.query_devices(input_device).get("name", str(input_device))
            log.info("オーディオストリーム準備完了: %s", device_label)
            return True
        except Exception as e:
            log.error(f"オーディオストリーム準備エラー: {e}")
            self._stream = None
            return False

    def _close_audio_stream(self):
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

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
        if self._process_thread and self._process_thread.is_alive():
            log.info("前回の音声処理中のため録音開始をスキップ")
            return
        self.audio_queue = queue.Queue()
        self.vad.reset()
        self._vad_history.reset()
        self._audio_activity.reset()
        self._target_app = get_frontmost_application()
        self._has_audio_started = False
        self._start_requested_at = time.time()
        self._set_status("starting")
        log.info("録音開始要求")

        self.is_recording = True
        if not self._ensure_audio_stream():
            self.is_recording = False
            self._has_audio_started = False
            self._start_requested_at = None
            self._set_status("idle")
            return

        self._process_thread = threading.Thread(target=self._process_audio, daemon=True)
        self._process_thread.start()

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        self._has_audio_started = False
        self._start_requested_at = None
        self._audio_activity.reset()
        self._set_status("idle")
        log.info("録音停止")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            log.warning(f"オーディオステータス: {status}")
        if self.is_recording:
            samples = indata[:, 0].astype(np.float32)
            if not self._has_audio_started:
                self._has_audio_started = True
                if self._start_requested_at is not None:
                    elapsed = time.time() - self._start_requested_at
                    log.info(f"入力待機開始 ({elapsed:.2f}s)")
                self._set_status("listening")
            self._update_live_audio_status(samples)
            self.audio_queue.put(samples.copy())

    def _update_live_audio_status(self, samples):
        new_status = self._audio_activity.status_for_samples(samples, self._current_status())
        if new_status is not None:
            self._set_status(new_status)

    def _process_audio(self):
        process_audio_queue(
            self.audio_queue,
            lambda: self.is_recording,
            self.vad,
            self._vad_history,
            self._process_vad_segments,
        )

    def _process_vad_segments(self):
        target_app = self._target_app

        def submit_segment(speech_samples):
            self._segment_executor.submit(self._handle_speech_segment, speech_samples, target_app)

        drain_vad_segments(
            self.vad,
            self._vad_history,
            VAD_MIN_SPEECH,
            VAD_PRE_ROLL_DURATION,
            submit_segment,
        )

    def _handle_speech_segment(self, speech_samples, target_app):
        try:
            self._set_status("processing")
            start = time.time()
            text = recognize_speech(self.recognizer, speech_samples)
            elapsed = time.time() - start

            if not text:
                log.info(f"ASR ({elapsed:.2f}s): 認識結果なし")
                return

            log.info(f"ASR ({elapsed:.2f}s): {text}")
            text = self._text_for_insert(text)
            if not text:
                return

            if self._native_output:
                self._set_status("inserting")
                self._send_text_to_native_app(text, target_app)
                if not native_paste_bridge_ready():
                    log.warning("ネイティブ貼り付け受信側が未起動のためPython側で貼り付けます")
                    inserted = insert_text_at_cursor(text, target_app)
                    if inserted:
                        log.info("Python経由でカーソル位置に入力しました")
                    else:
                        notify_user(
                            "Voice Input",
                            "アクセシビリティ許可が必要です",
                            "Python または VoiceInputTool を許可してください",
                        )
                return

            self._set_status("inserting")
            inserted = insert_text_at_cursor(text, target_app)
            if inserted:
                log.info("カーソル位置に入力しました")
            notify_user(
                "Voice Input",
                "カーソル位置に入力しました" if inserted else "貼り付け不可のためコピーしました",
                text[:100],
            )
        except Exception:
            log.exception("発話セグメント処理で予期しないエラーが発生しました")
            notify_user(
                "Voice Input",
                "音声入力処理でエラーが発生しました",
                "詳細はログを確認してください",
            )
        finally:
            self._restore_recording_status()

    def _send_text_to_native_app(self, text, target_app=None):
        pid = target_pid(target_app)

        try:
            write_output(text, pid)
            log.info("ネイティブ貼り付けへ送信: pid=%s text=%r", pid, truncate_for_log(text, 300))
        except Exception:
            log.exception("ネイティブ貼り付けへの送信に失敗しました")
            copy_to_clipboard(text)
            notify_user("Voice Input", "貼り付けに失敗したためコピーしました", text[:100])

    def _text_for_insert(self, text):
        if not self.use_llm:
            return text

        self._set_status("correcting")
        try:
            corrected = llm_correct(text)
        except Exception as e:
            log.exception("LLM補正に失敗しました: %s", e)
            notify_user(
                "Voice Input",
                "LLM補正に失敗しました",
                str(e)[:100],
            )
            return ""

        log.info(f"LLM補正: {corrected}")
        return corrected

    def open_settings(self, sender=None):
        from settings_ui import SettingsWindowController
        from Cocoa import NSApp
        self._settings_ctrl = SettingsWindowController.alloc().initWithCallback_(self._on_settings_saved)
        self._settings_ctrl.show()
        NSApp.activateIgnoringOtherApps_(True)

    def toggle_llm(self, sender=None):
        """メニューから LLM 補正のON/OFFを切り替え、即時保存"""
        global APP_CONFIG
        self.use_llm = not self.use_llm
        self.llm_status.title = "LLM補正: ON" if self.use_llm else "LLM補正: OFF"
        APP_CONFIG["use_llm"] = self.use_llm
        try:
            save_config(APP_CONFIG)
        except Exception as e:
            log.error(f"設定保存に失敗しました: {e}")
        self._set_status(self._current_status(), force=True)

    def _on_settings_saved(self, new_config):
        global APP_CONFIG, VAD_THRESHOLD, VAD_SILENCE_DURATION, VAD_MIN_SPEECH, VAD_PRE_ROLL_DURATION
        APP_CONFIG = new_config
        self.use_llm = new_config["use_llm"]
        VAD_THRESHOLD = new_config["vad_threshold"]
        VAD_SILENCE_DURATION = new_config["vad_silence_duration"]
        VAD_MIN_SPEECH = new_config["vad_min_speech"]
        VAD_PRE_ROLL_DURATION = new_config.get("vad_pre_roll_duration", VAD_PRE_ROLL_DURATION)
        configure_llm(new_config)
        self.llm_status.title = "LLM補正: ON" if self.use_llm else "LLM補正: OFF"
        if not self.is_recording:
            self.vad = build_vad()
            self._close_audio_stream()
            self._ensure_audio_stream()
        # ホットキー再登録
        self._register_hotkey()
        self._set_status(self._current_status(), force=True)
        log.info(f"設定更新: LLM={'ON' if self.use_llm else 'OFF'}, "
                 f"ホットキー={new_config.get('hotkey_record')}, "
                 f"入力マイク={new_config.get('input_device_id', '') or '自動選択'}")

    def quit_app(self):
        self.stop_recording()
        self._stop_status_icon_animation()
        self._close_audio_stream()
        self._segment_executor.shutdown(wait=False, cancel_futures=True)
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
            try:
                corrected = llm_correct(text)
                print(f"補正: {corrected}")
            except Exception as e:
                print(f"補正エラー: {e}")
        print()

    print("=== テスト完了 ===")


def run_settings_window():
    if not HAS_APP_HELPER:
        print("PyObjCTools が利用できないため設定画面を開けません。", file=sys.stderr)
        return 1

    from Cocoa import NSApplication
    from settings_ui import SettingsWindowController

    app = NSApplication.sharedApplication()
    ctrl = SettingsWindowController.alloc().initWithCallback_(lambda _config: AppHelper.stopEventLoop())
    ctrl.show()
    app.activateIgnoringOtherApps_(True)
    AppHelper.runEventLoop()
    return 0


def run_headless_app(recognizer, vad, use_llm=False):
    app = VoiceInputApp(recognizer, vad, use_llm=use_llm, headless=True, native_output=True)
    app._set_status("idle", force=True)

    try:
        ensure_bridge_files()
        command_reader = NativeCommandReader()
    except Exception:
        log.exception("コマンドファイルの準備に失敗しました")
        raise

    should_quit = threading.Event()

    def handle_signal(_signum, _frame):
        should_quit.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    log.info("ヘッドレス音声入力エンジン起動: command_file=%s", command_reader.path)
    if not native_paste_bridge_ready():
        request_accessibility_permission(prompt=True)

    try:
        while not should_quit.is_set():
            time.sleep(0.2)
            try:
                commands = command_reader.read_new_commands()
                if not commands:
                    continue
            except Exception:
                log.exception("コマンドファイルの読み込みに失敗しました")
                continue

            for command in commands:
                command = command.strip()
                if not command:
                    continue
                log.info("コマンド受信: %s", command)
                if command == "toggle":
                    app.toggle_recording()
                elif command == "start":
                    app.start_recording()
                elif command == "stop":
                    app.stop_recording()
                elif command == "toggle_llm":
                    app.toggle_llm()
                elif command == "quit":
                    should_quit.set()
                else:
                    log.warning("未知のコマンド: %s", command)
    finally:
        app.stop_recording()
        app._stop_status_icon_animation()
        app._close_audio_stream()
        app._segment_executor.shutdown(wait=False, cancel_futures=True)
        log.info("ヘッドレス音声入力エンジン終了")


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Voice Input Tool - ReazonSpeech ASR")
    parser.add_argument("--llm", action="store_true", help="LLM句読点補正を有効化")
    parser.add_argument("--no-llm", action="store_true", help="LLM句読点補正を無効化")
    parser.add_argument("--test", action="store_true", help="テストWAVファイルで動作確認")
    parser.add_argument("--headless", action="store_true", help="メニューバーなしで音声入力エンジンのみ起動")
    parser.add_argument("--settings", action="store_true", help="設定画面だけを開く")
    args = parser.parse_args()

    use_llm = (args.llm or APP_CONFIG.get("use_llm", False)) and not args.no_llm

    if args.settings:
        sys.exit(run_settings_window())

    log.info("モデル読み込み開始")
    start = time.time()
    recognizer = create_recognizer()
    vad = build_vad()
    log.info(f"モデル読み込み完了 ({time.time()-start:.1f}s)")
    log.info(f"LLM補正: {'ON' if use_llm else 'OFF'}")

    if args.test:
        run_test(recognizer, use_llm=use_llm)
    elif args.headless:
        run_headless_app(recognizer, vad, use_llm=use_llm)
    else:
        install_status_bar_diagnostics()
        if HAS_APPKIT_APPLICATION:
            try:
                NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
                log.info("アプリ表示モード: accessory")
            except Exception:
                log.exception("アプリ表示モードの設定に失敗しました")
        if HAS_APP_HELPER:
            AppHelper.callLater(2.0, lambda: log.info("メニューバーイベントループ稼働中"))
        app = VoiceInputApp(recognizer, vad, use_llm=use_llm)
        log.info("メニューバーイベントループ開始")
        app.run()
        log.error("メニューバーイベントループが終了しました")


if __name__ == "__main__":
    main()
