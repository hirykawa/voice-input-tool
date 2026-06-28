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
_error_handler = logging.FileHandler(os.path.join(_log_dir, "voice-input-error.log"), encoding="utf-8")
_error_handler.setLevel(logging.ERROR)
_error_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logging.getLogger().addHandler(_error_handler)

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

try:
    from ApplicationServices import (
        AXIsProcessTrusted,
        CGEventCreateKeyboardEvent,
        CGEventPost,
        CGEventSetFlags,
        kCGEventFlagMaskCommand,
        kCGHIDEventTap,
    )
    from AppKit import NSApplicationActivateIgnoringOtherApps, NSWorkspace
    HAS_CURSOR_INSERT = True
except ImportError:
    HAS_CURSOR_INSERT = False

try:
    from PyObjCTools import AppHelper
    HAS_APP_HELPER = True
except ImportError:
    HAS_APP_HELPER = False

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
AUDIO_ACTIVITY_RMS_THRESHOLD = 0.025
AUDIO_ACTIVITY_RELEASE_RMS_THRESHOLD = 0.015
AUDIO_ACTIVITY_HOLD_SECONDS = 0.35
AUDIO_INITIAL_NOISE_FLOOR = 0.003
AUDIO_NOISE_FLOOR_ALPHA_IDLE = 0.08
AUDIO_NOISE_FLOOR_ALPHA_ACTIVE = 0.01
AUDIO_NOISE_ON_MULTIPLIER = 2.2
AUDIO_NOISE_OFF_MULTIPLIER = 1.4
AUDIO_STATUS_UPDATE_INTERVAL = 0.08

# 設定ファイルから読み込み
APP_CONFIG = load_config()

VAD_THRESHOLD = APP_CONFIG["vad_threshold"]
VAD_SILENCE_DURATION = APP_CONFIG["vad_silence_duration"]
VAD_MIN_SPEECH = APP_CONFIG["vad_min_speech"]
LLM_PROMPT = APP_CONFIG["llm_prompt"]
LLM_MODEL = APP_CONFIG.get("llm_model", "openai/gpt-oss-120b")
LLM_PROVIDER_ORDER = APP_CONFIG.get("llm_provider_order", ["Cerebras"])
VAD_PRE_ROLL_DURATION = APP_CONFIG.get("vad_pre_roll_duration", 0.8)

_OPENROUTER_CLIENT = None
_OPENROUTER_CLIENT_KEY = None


# ============================================================
# ASRエンジン初期化
# ============================================================

def create_recognizer():
    """ReazonSpeech Zipformer モデルでASRエンジンを作成"""
    int8_encoder = os.path.join(MODEL_DIR, "encoder-epoch-99-avg-1.int8.onnx")
    int8_decoder = os.path.join(MODEL_DIR, "decoder-epoch-99-avg-1.int8.onnx")
    int8_joiner = os.path.join(MODEL_DIR, "joiner-epoch-99-avg-1.int8.onnx")
    tokens = os.path.join(MODEL_DIR, "tokens.txt")

    missing = [p for p in [int8_encoder, int8_decoder, int8_joiner, tokens] if not os.path.exists(p)]
    if missing:
        print("[ERROR] 以下のモデルファイルが見つかりません:\n" + "\n".join(f" - {m}" for m in missing))
        print("README の 'モデルダウンロード' 手順を実行してください。")
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

def _truncate_for_log(value, limit=2000):
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"...<truncated {len(text) - limit} chars>"


def _mask_secret(value):
    if not value:
        return ""
    value = str(value)
    if len(value) <= 8:
        return "***"
    return f"{value[:6]}...{value[-4:]}"


def _safe_model_dump(obj):
    try:
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="json")
        if hasattr(obj, "dict"):
            return obj.dict()
    except Exception as e:
        return {"dump_error": repr(e), "repr": repr(obj)}
    return repr(obj)


class LLMCorrectionError(RuntimeError):
    pass


def get_openrouter_client(api_key):
    global _OPENROUTER_CLIENT, _OPENROUTER_CLIENT_KEY
    if _OPENROUTER_CLIENT is None or _OPENROUTER_CLIENT_KEY != api_key:
        _OPENROUTER_CLIENT = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=10.0,
            max_retries=0,
        )
        _OPENROUTER_CLIENT_KEY = api_key
    return _OPENROUTER_CLIENT


def llm_correct(text, api_key=None):
    """OpenRouter 経由の Cerebras GPT OSS で句読点補正"""
    if not text:
        return ""
    if not HAS_OPENAI:
        raise LLMCorrectionError("openai パッケージがインストールされていません")

    key = api_key or APP_CONFIG.get("openrouter_api_key") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise LLMCorrectionError("OPENROUTER_API_KEY が設定されていません")

    client = get_openrouter_client(key)
    max_tokens = min(2048, max(1024, len(text) + 256))
    extra_body = {
        "data_collection": "deny",
        "zdr": True,
        "provider": {
            "order": LLM_PROVIDER_ORDER,
            "allow_fallbacks": False,
        },
        "reasoning": {
            "effort": "low",
            "exclude": True,
        },
        "reasoning_effort": "low",
        "chat_template_kwargs": {"enable_thinking": False},
    }

    log.info(
        "LLM補正リクエスト: model=%s providers=%s text_len=%d key=%s prompt_len=%d max_tokens=%d text=%r",
        LLM_MODEL,
        LLM_PROVIDER_ORDER,
        len(text),
        _mask_secret(key),
        len(LLM_PROMPT),
        max_tokens,
        _truncate_for_log(text, 300),
    )

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": LLM_PROMPT},
                {"role": "user", "content": text},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
            extra_body=extra_body,
        )
        choices = getattr(response, "choices", []) or []
        if not choices:
            log.error("LLM応答choicesが空: raw=%s", _truncate_for_log(_safe_model_dump(response)))
            raise LLMCorrectionError("LLM応答choicesが空でした")

        choice = choices[0]
        message = getattr(choice, "message", None)
        corrected = getattr(message, "content", None) if message is not None else None
        finish_reason = getattr(choice, "finish_reason", None)
        response_model = getattr(response, "model", None)
        response_id = getattr(response, "id", None)
        usage = getattr(response, "usage", None)
        content_len = len(corrected) if corrected else 0
        log.info(
            "LLM応答: id=%s response_model=%s finish_reason=%s content_len=%d usage=%s content_preview=%r",
            response_id,
            response_model,
            finish_reason,
            content_len,
            _truncate_for_log(usage, 500),
            _truncate_for_log(corrected or "", 500),
        )

        if corrected is None:
            log.error("LLM応答contentがNone: raw=%s", _truncate_for_log(_safe_model_dump(response)))
            raise LLMCorrectionError("LLM補正結果が空でした")
        corrected = corrected.strip()
        if not corrected:
            log.error("LLM応答contentが空: raw=%s", _truncate_for_log(_safe_model_dump(response)))
            raise LLMCorrectionError("LLM補正結果が空でした")
        return corrected
    except LLMCorrectionError:
        raise
    except Exception as e:
        log.exception("LLM補正API呼び出しエラー: model=%s providers=%s text_len=%d", LLM_MODEL, LLM_PROVIDER_ORDER, len(text))
        raise LLMCorrectionError(str(e)) from e


# ============================================================
# 通知
# ============================================================

def notify_user(title, subtitle, message=""):
    if threading.current_thread() is threading.main_thread() or not HAS_APP_HELPER:
        _notify_user_now(title, subtitle, message)
    else:
        AppHelper.callAfter(_notify_user_now, title, subtitle, message)


def _notify_user_now(title, subtitle, message=""):
    try:
        rumps.notification(title, subtitle, message)
    except Exception as e:
        log.warning("通知表示に失敗しました: %s", e)


# ============================================================
# カーソル位置への入力
# ============================================================

def get_frontmost_application():
    if not HAS_CURSOR_INSERT:
        return None
    try:
        return NSWorkspace.sharedWorkspace().frontmostApplication()
    except Exception as e:
        log.warning(f"前面アプリ取得エラー: {e}")
        return None


def insert_text_at_cursor(text, target_app=None):
    if not text:
        return False

    try:
        pyperclip.copy(text)
    except Exception as e:
        log.error(f"クリップボードへのコピーに失敗しました: {e}")
        return False

    if not HAS_CURSOR_INSERT:
        log.warning("カーソル位置への入力に必要なmacOS APIを読み込めません")
        return False

    try:
        if not AXIsProcessTrusted():
            log.warning("アクセシビリティ権限がないため、カーソル位置へ入力できません")
            return False
    except Exception as e:
        log.warning(f"アクセシビリティ権限確認エラー: {e}")

    try:
        if target_app is not None:
            target_app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
            time.sleep(0.05)
        send_paste_shortcut()
        return True
    except Exception as e:
        log.error(f"カーソル位置への入力に失敗しました: {e}")
        return False


def send_paste_shortcut():
    command_keycode = 55
    v_keycode = 9
    flags = kCGEventFlagMaskCommand

    for keycode, key_down, event_flags in (
        (command_keycode, True, flags),
        (v_keycode, True, flags),
        (v_keycode, False, flags),
        (command_keycode, False, 0),
    ):
        post_key_event(keycode, key_down, event_flags)


def post_key_event(keycode, key_down, flags=0):
    event = CGEventCreateKeyboardEvent(None, keycode, key_down)
    CGEventSetFlags(event, flags)
    CGEventPost(kCGHIDEventTap, event)


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
        self._target_app = None
        self._status = "idle"
        self._status_lock = threading.Lock()
        self._status_version = 0
        self._has_audio_started = False
        self._start_requested_at = None
        self._reset_audio_activity_state()
        self._segment_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._reset_vad_audio_history()

        self._hotkey_listener = None

        # メニュー構成
        hotkey_display = self._get_hotkey_display()
        self.record_button = rumps.MenuItem(
            f"録音開始 ({hotkey_display})", callback=self.toggle_recording
        )
        llm_label = "LLM補正: ON" if self.use_llm else "LLM補正: OFF"
        self.llm_status = rumps.MenuItem(llm_label, callback=self.toggle_llm)
        self.settings_button = rumps.MenuItem("設定...", callback=self.open_settings)

        self.menu = [
            self.record_button,
            None,
            self.llm_status,
            self.settings_button,
        ]

        self._ensure_audio_stream()

        # ホットキー登録
        self._register_hotkey()

    def _set_status(self, status, force=False):
        lock = getattr(self, "_status_lock", None)
        if lock is None:
            if not force and getattr(self, "_status", None) == status:
                return
            self._status = status
            self._apply_status()
            return

        with lock:
            if not force and self._status == status:
                return
            self._status = status
            self._status_version += 1
            version = self._status_version

        if threading.current_thread() is threading.main_thread() or not HAS_APP_HELPER:
            self._apply_status(version)
        else:
            AppHelper.callAfter(self._apply_status, version)

    def _apply_status(self, version=None):
        lock = getattr(self, "_status_lock", None)
        if lock is None:
            status = getattr(self, "_status", "idle")
        else:
            with lock:
                if version is not None and version != self._status_version:
                    return
                status = self._status

        title, record_title = self._status_labels(status)
        self.title = title
        self.record_button.title = record_title

    def _status_labels(self, status):
        hotkey_display = self._get_hotkey_display()
        states = {
            "idle": ("🎙", f"録音開始 ({hotkey_display})"),
            "starting": ("⏳", f"マイク起動中… ({hotkey_display})"),
            "listening": ("🟢", f"録音停止・入力待機中 ({hotkey_display})"),
            "hearing": ("🔴", f"録音停止・発話検出中 ({hotkey_display})"),
            "processing": ("📝", "音声認識中…"),
            "correcting": ("🧠", "LLM補正中…"),
            "inserting": ("⌨️", "カーソル位置へ入力中…"),
        }
        return states.get(status, states["idle"])

    def _current_status(self):
        lock = getattr(self, "_status_lock", None)
        if lock is None:
            return getattr(self, "_status", "idle")
        with lock:
            return self._status

    def _restore_recording_status(self):
        if self.is_recording:
            self._set_status("listening")
        else:
            self._set_status("idle")

    def _get_hotkey_display(self):
        """設定からホットキーの表示文字列を取得"""
        from settings_ui import hotkey_to_display
        hotkey = APP_CONFIG.get("hotkey_record", "<ctrl>+<shift>+<space>")
        return hotkey_to_display(hotkey)

    def _reset_audio_activity_state(self):
        self._last_audio_status_update = 0.0
        self._audio_noise_floor = AUDIO_INITIAL_NOISE_FLOOR
        self._audio_is_speaking = False
        self._audio_last_active_at = 0.0

    def _reset_vad_audio_history(self):
        self._vad_audio_blocks = []
        self._vad_audio_sample_count = 0

    def _accept_vad_block(self, block):
        if not hasattr(self, "_vad_audio_blocks"):
            self._reset_vad_audio_history()
        self._vad_audio_blocks.append((self._vad_audio_sample_count, block.copy()))
        self._vad_audio_sample_count += len(block)
        self.vad.accept_waveform(block)

    def _get_vad_audio_range(self, start, end):
        chunks = []
        for block_start, block in self._vad_audio_blocks:
            block_end = block_start + len(block)
            if block_end <= start:
                continue
            if block_start >= end:
                break
            chunk_start = max(0, start - block_start)
            chunk_end = min(len(block), end - block_start)
            if chunk_start < chunk_end:
                chunks.append(block[chunk_start:chunk_end])
        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32, copy=False)

    def _prune_vad_audio_history(self, keep_from):
        self._vad_audio_blocks = [
            (start, block)
            for start, block in self._vad_audio_blocks
            if start + len(block) > keep_from
        ]

    def _segment_samples_with_preroll(self, segment):
        segment_samples = np.array(segment.samples, dtype=np.float32)
        segment_start = int(getattr(segment, "start", 0))
        segment_end = segment_start + len(segment_samples)
        pre_roll_samples = int(SAMPLE_RATE * VAD_PRE_ROLL_DURATION)
        padded_start = max(0, segment_start - pre_roll_samples)

        speech_samples = self._get_vad_audio_range(padded_start, segment_end)
        if len(speech_samples) == 0:
            speech_samples = segment_samples

        added = max(0, min(segment_start, segment_end) - padded_start)
        if added:
            log.info(f"VAD先頭補完: {added / SAMPLE_RATE:.2f}秒")

        self._prune_vad_audio_history(max(0, segment_end - pre_roll_samples))
        return speech_samples

    def _ensure_audio_stream(self):
        if self._stream is not None:
            return True

        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                blocksize=BLOCK_SIZE,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
            log.info("オーディオストリーム準備完了")
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
        self._reset_vad_audio_history()
        self._reset_audio_activity_state()
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
        self._reset_audio_activity_state()
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
        status = self._current_status()
        if status in {"processing", "correcting", "inserting"}:
            return

        now = time.monotonic()
        if now - self._last_audio_status_update < AUDIO_STATUS_UPDATE_INTERVAL:
            return
        self._last_audio_status_update = now

        rms = float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.0
        noise_floor = getattr(self, "_audio_noise_floor", AUDIO_INITIAL_NOISE_FLOOR)
        is_speaking = getattr(self, "_audio_is_speaking", False)
        alpha = AUDIO_NOISE_FLOOR_ALPHA_ACTIVE if is_speaking else AUDIO_NOISE_FLOOR_ALPHA_IDLE
        noise_floor = max(0.00001, (1.0 - alpha) * noise_floor + alpha * rms)
        self._audio_noise_floor = noise_floor

        on_threshold = max(AUDIO_ACTIVITY_RMS_THRESHOLD, noise_floor * AUDIO_NOISE_ON_MULTIPLIER)
        off_threshold = max(AUDIO_ACTIVITY_RELEASE_RMS_THRESHOLD, noise_floor * AUDIO_NOISE_OFF_MULTIPLIER)

        if rms >= on_threshold:
            is_speaking = True
            self._audio_last_active_at = now
        elif is_speaking and (
            rms <= off_threshold
            or now - getattr(self, "_audio_last_active_at", 0.0) >= AUDIO_ACTIVITY_HOLD_SECONDS
        ):
            is_speaking = False

        self._audio_is_speaking = is_speaking
        new_status = "hearing" if is_speaking else "listening"
        if new_status != status:
            log.info(
                "音声状態: %s rms=%.4f noise=%.4f on=%.4f off=%.4f",
                new_status,
                rms,
                noise_floor,
                on_threshold,
                off_threshold,
            )
        self._set_status(new_status)

    def _process_audio(self):
        audio_buffer = np.array([], dtype=np.float32)
        chunk_count = 0

        while self.is_recording or not self.audio_queue.empty():
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

                self._accept_vad_block(block)
                self._process_vad_segments()

        if len(audio_buffer) > 0:
            padded = np.pad(audio_buffer, (0, BLOCK_SIZE - len(audio_buffer)))
            self._accept_vad_block(padded)
        self.vad.flush()
        self._process_vad_segments()
        self.vad.reset()
        self._reset_vad_audio_history()

    def _process_vad_segments(self):
        while not self.vad.empty():
            segment = self.vad.front
            segment_sample_count = len(segment.samples)
            speech_samples = self._segment_samples_with_preroll(segment)
            self.vad.pop()
            duration = len(speech_samples) / SAMPLE_RATE
            log.info(f"VAD検出: {duration:.1f}秒")

            if segment_sample_count < SAMPLE_RATE * VAD_MIN_SPEECH:
                log.info("最小発話長未満、スキップ")
                continue

            target_app = self._target_app
            self._segment_executor.submit(self._handle_speech_segment, speech_samples, target_app)

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

    def _on_settings_saved(self, new_config):
        global APP_CONFIG, VAD_THRESHOLD, VAD_SILENCE_DURATION, VAD_MIN_SPEECH, LLM_PROMPT, LLM_MODEL, LLM_PROVIDER_ORDER, VAD_PRE_ROLL_DURATION
        APP_CONFIG = new_config
        self.use_llm = new_config["use_llm"]
        VAD_THRESHOLD = new_config["vad_threshold"]
        VAD_SILENCE_DURATION = new_config["vad_silence_duration"]
        VAD_MIN_SPEECH = new_config["vad_min_speech"]
        LLM_PROMPT = new_config["llm_prompt"]
        LLM_MODEL = new_config.get("llm_model", LLM_MODEL)
        LLM_PROVIDER_ORDER = new_config.get("llm_provider_order", LLM_PROVIDER_ORDER)
        VAD_PRE_ROLL_DURATION = new_config.get("vad_pre_roll_duration", VAD_PRE_ROLL_DURATION)
        self.llm_status.title = "LLM補正: ON" if self.use_llm else "LLM補正: OFF"
        if not self.is_recording:
            self.vad = create_vad()
        # ホットキー再登録
        self._register_hotkey()
        self._set_status(self._status, force=True)
        log.info(f"設定更新: LLM={'ON' if self.use_llm else 'OFF'}, "
                 f"ホットキー={new_config.get('hotkey_record')}")

    def quit_app(self):
        self.stop_recording()
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
