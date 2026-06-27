# Voice Input Tool

ReazonSpeech (sherpa-onnx) + Silero VAD + オプションLLM補正による
ローカル音声入力ツール。Mac Mini (Apple Silicon) 向け。

## 特徴

- 完全ローカルASR（ReazonSpeech K2 v2 int8）
- Silero VAD による発話区間自動検出
- LLM補正オン/オフ切替可能（デフォルト: ON）
- LLM補正は句読点挿入のみ（文章変更・言い換え禁止）
- OpenRouter GLM 5.2 で学習拒否設定（data_collection: deny / zdr: true）
- macOS メニューバーアプリ（rumps）として常駐
- ネイティブ設定画面（PyObjC）でGUIから設定変更可能
- グローバルホットキー対応（設定画面でカスタマイズ可能）
- 認識結果はクリップボードにコピー＆通知表示

## セットアップ

### 1. 依存パッケージ

```bash
cd ~/voice-input-tool
python3 -m venv .venv
source .venv/bin/activate
pip install sherpa-onnx numpy sounddevice pyperclip pynput openai rumps
```

### 2. モデルダウンロード

```bash
mkdir -p ~/voice-input-tool/models
cd ~/voice-input-tool/models

# ASRモデル (ReazonSpeech K2 v2 int8)
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2
tar xjf sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2

# VADモデル (Silero)
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
```

### 3. 環境変数（LLM補正を使う場合）

```bash
# .env ファイルに設定（.gitignore済み）
echo "OPENROUTER_API_KEY=sk-or-v1-xxxxx" > ~/voice-input-tool/.env
```

または設定画面からAPI Keyを入力可能。

### 4. macOS アプリとして使う

`~/Applications/Voice Input Tool.app` として配置済み。
Dockには表示されず、メニューバーに 🎙 アイコンとして常駐する。

アプリ起動後、以下の権限が必要:
- **マイク**: 音声認識に必要
- **アクセシビリティ**: グローバルホットキーに必要（任意）

## 使い方

### メニューバーから操作

メニューバーの 🎙 アイコンをクリック:

| メニュー項目 | 説明 |
|-------------|------|
| 録音開始/停止 | 録音をトグル（録音中は 🔴 に変化） |
| LLM補正: ON/OFF | 現在のLLM補正状態を表示 |
| 設定... | 設定画面を開く |
| 終了 | アプリを終了 |

### CLIから使う

```bash
cd ~/voice-input-tool
source .venv/bin/activate

# メニューバーアプリとして起動
python3 voice_input.py --llm

# テストモード（WAVファイルで動作確認）
python3 voice_input.py --test
```

### ホットキー

デフォルト: `Ctrl+Shift+Space` で録音開始/停止（設定画面で変更可能）

## 設定画面

メニューバーの 🎙 →「設定...」で開く。

| 設定項目 | 説明 | デフォルト |
|---------|------|-----------|
| LLM 句読点補正 | ON/OFF切替 | ON |
| OpenRouter API Key | LLM補正用APIキー | .envから読み込み |
| 録音 開始/停止 | ホットキーの設定（フィールドをクリックしてキーを押す） | Ctrl+Shift+Space |
| VAD 発話検出閾値 | 発話と判定する閾値 (0.1-0.9) | 0.5 |
| 無音判定時間 | 発話終了と判断する無音の長さ (0.2-3.0秒) | 0.8秒 |
| 最小発話長 | 認識対象とする最短の発話長 (0.1-2.0秒) | 0.3秒 |
| LLM 補正プロンプト | LLM補正時のシステムプロンプト | 句読点挿入のみ |

設定は `~/voice-input-tool/config.json` に保存される（.gitignore済み）。
保存時に即座に反映（再起動不要）。

## ファイル構成

```
voice_input.py   - メインアプリ（ASR/VAD/LLM/メニューバー）
config.py        - 設定ファイル管理（config.json の読み書き）
settings_ui.py   - ネイティブmacOS設定画面（PyObjC）
start.sh         - CLI起動用シェルスクリプト
models/          - ASR・VADモデル（.gitignore済み）
logs/            - ログファイル（.gitignore済み）
config.json      - ユーザー設定（.gitignore済み）
.env             - 環境変数（.gitignore済み）
```

## パフォーマンス

| 項目 | 結果 |
|------|------|
| ASR処理速度 | 0.06〜0.16秒（10秒程度の音声） |
| モデル読み込み | 0.4〜0.5秒 |
| 日本語精度 | テスト5問中ほぼ完璧 |
| LLM補正 | 句読点のみ挿入、文章変更なし |

## モデル

- ASR: sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01 (int8)
- VAD: silero_vad.onnx
- LLM: z-ai/glm-5.2 (OpenRouter経由、学習拒否設定付き)
