"""設定ファイル管理"""
import json
import os

CONFIG_PATH = os.path.expanduser("~/voice-input-tool/config.json")

DEFAULTS = {
    "use_llm": True,
    "openrouter_api_key": "",
    "vad_threshold": 0.5,
    "vad_silence_duration": 0.8,
    "vad_min_speech": 0.3,
    "hotkey_record": "<ctrl>+<shift>+<space>",
    "llm_prompt": "以下の音声認識結果に、句読点のみを挿入してください。\n文章の変更・言い換え・要約は一切しないでください。\nフィラー（あー、えー、まー）の削除のみ許可します。\n話し言葉を書き言葉に変換しないでください。\n出力は補正後のテキストのみとし、説明やコメントは一切付けないでください。",
}


def load_config():
    """設定ファイルを読み込み、デフォルト値とマージして返す"""
    config = dict(DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
        config.update(saved)
    # .env の API キーをフォールバック
    if not config["openrouter_api_key"]:
        config["openrouter_api_key"] = os.environ.get("OPENROUTER_API_KEY", "")
    return config


def save_config(config):
    """設定ファイルに保存"""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
