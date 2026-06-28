# Voice Input Tool

ReazonSpeech (sherpa-onnx) + Silero VAD + OpenRouter/Cerebras GPT OSS による
macOS 向けローカル音声入力ツールです。

## 特徴

- 完全ローカルASR（ReazonSpeech K2 v2 int8）
- Silero VAD による発話区間自動検出
- マイク入力を事前準備し、録音開始直後の頭欠けを抑制
- VAD検出前の音声も先頭補完し、話し始めの欠落を抑制
- 録音は明示的に停止するまで継続（無音では停止しない）
- 無音で発話区間が確定したら、ASR/LLM補正/入力処理を聞き取りとは別スレッドで実行
- LLM補正オン/オフ切替可能（デフォルト: ON）
- LLM補正ON時は、必ずLLM補正後のテキストだけを出力（未補正フォールバックなし）
- OpenRouter 経由で Cerebras の GPT OSS を使用
  - model: `openai/gpt-oss-120b`
  - provider: `Cerebras`
  - provider fallback: 無効
  - `data_collection: deny` / `zdr: true`
- macOS メニューバーアプリ（rumps）として常駐
- ネイティブ設定画面（PyObjC）でGUIから設定変更可能
- グローバルホットキー対応（設定画面でカスタマイズ可能）
- 認識結果はカーソル位置に入力（貼り付け不可時は補正済みテキストをクリップボードに保持）
- 詳細ログを `logs/` に出力

## セットアップ

### 1. 依存パッケージ

```bash
cd ~/voice-input-tool
python3 -m venv .venv
source .venv/bin/activate
pip install sherpa-onnx numpy sounddevice pyperclip pynput openai rumps pyobjc
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

### 3. OpenRouter API Key

LLM補正を使う場合は `OPENROUTER_API_KEY` が必要です。

```bash
# .env ファイルに設定（.gitignore 済み・自動で読み込みます）
echo "OPENROUTER_API_KEY=sk-or-v1-xxxxx" > ~/voice-input-tool/.env
```

または設定画面からAPI Keyを入力できます。その場合も `.env` に保存され、`config.json` には保存しません。

### 4. macOS アプリとして使う

`~/Applications/Voice Input Tool.app` として配置して利用します。Dockには表示されず、メニューバーに 🎙 アイコンとして常駐します。

必要な権限:

- **マイク**: 音声認識に必要
- **アクセシビリティ**: グローバルホットキーとカーソル位置への入力に必要

## 使い方

### メニューバーから操作

メニューバーの 🎙 アイコンをクリックします。

| メニュー項目 | 説明 |
|-------------|------|
| 録音開始/停止 | 録音をトグル |
| LLM補正: ON/OFF | LLM補正の有効/無効を切替 |
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

デフォルト: `Ctrl+Shift+Space` で録音開始/停止。設定画面で変更できます。

## 入力フロー

1. ホットキー/メニューで録音開始
2. マイク入力をキューに蓄積
3. 無音で発話区間が確定
4. 発話区間を別スレッドへ投入
5. ASRでテキスト化
6. LLM補正ONの場合は Cerebras GPT OSS で整形
7. 補正済みテキストを録音開始時の前面アプリへ入力
8. 録音は停止コマンドまで継続

重要な仕様:

- 無音は「発話区間の区切り」であり、「録音停止」ではありません。
- LLM補正ON時は、LLM補正が成功したテキストだけを出力します。
- LLM補正に失敗した場合、未補正テキストは出力しません。
- LLM補正OFF時は、ASR結果をそのまま出力します。
- 貼り付けに失敗した場合でも、補正済みテキストはクリップボードに残ります。

## 状態表示

メニューバーアイコンは状態に応じて変わります。

| アイコン | 状態 |
|---------|------|
| 🎙 | 停止中 |
| ⏳ | マイク起動中 |
| 🟢 | 入力待機中 |
| 🔴 | 発話検出中 |
| 📝 | 音声認識中 |
| 🧠 | LLM補正中 |
| ⌨️ | カーソル位置へ入力中 |

`🔴` / `🟢` の切り替えはマイク入力の音量とノイズ床をもとに判定します。

## 設定画面

メニューバーの 🎙 →「設定...」で開きます。

| 設定項目 | 説明 | デフォルト |
|---------|------|-----------|
| LLM 句読点補正 | ON/OFF切替 | ON |
| OpenRouter API Key | LLM補正用APIキー | `.env` から読み込み |
| 録音 開始/停止 | ホットキー設定 | Ctrl+Shift+Space |
| VAD 発話検出閾値 | 発話と判定する閾値 (0.1-0.9) | 0.5 |
| 無音判定時間 | 発話終了と判断する無音の長さ (0.2-3.0秒) | 0.8秒 |
| 最小発話長 | 認識対象とする最短の発話長 (0.1-2.0秒) | 0.3秒 |
| LLM 補正プロンプト | LLM補正時のシステムプロンプト | 句読点挿入のみ |

設定は `~/voice-input-tool/config.json` に保存されます。API Key は `.env` に保存され、`config.json` には保存されません。

## LLM設定

既定値は `config.py` の `DEFAULTS` で管理しています。

| キー | 既定値 | 説明 |
|------|--------|------|
| `llm_model` | `openai/gpt-oss-120b` | OpenRouter のモデル名 |
| `llm_provider_order` | `["Cerebras"]` | 使用する provider |
| `vad_pre_roll_duration` | `0.8` | VAD検出前に補完する音声秒数 |

LLMリクエストでは provider fallback を無効にしています。Cerebras が利用できない場合、LLM補正は失敗し、LLM補正ONでは出力されません。

## ログとトラブルシュート

ログは以下に出力します。

```bash
~/voice-input-tool/logs/voice-input.log
~/voice-input-tool/logs/voice-input-error.log
```

リアルタイム確認:

```bash
tail -f ~/voice-input-tool/logs/voice-input.log
```

出力されない場合は、以下を確認します。

```bash
tail -n 120 ~/voice-input-tool/logs/voice-input.log
tail -n 120 ~/voice-input-tool/logs/voice-input-error.log
```

よく見るログ:

| ログ | 意味 |
|------|------|
| `ASR (...s): ...` | ASRは成功 |
| `LLM補正リクエスト` | OpenRouterへ送信開始 |
| `LLM応答` | OpenRouterから応答あり |
| `content_len=0` / `content=None` | LLM本文が空。LLM補正失敗扱い |
| `finish_reason=length` | reasoning等で token を使い切った可能性 |
| `カーソル位置に入力しました` | 入力成功 |
| `貼り付け不可のためコピーしました` | 補正済みテキストをクリップボードに保持 |

ログでは API Key をマスクします。

## ファイル構成

```
voice_input.py   - メインアプリ（ASR/VAD/LLM/メニューバー）
config.py        - 設定ファイル管理（config.json/.env の読み書き）
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
- LLM: openai/gpt-oss-120b (OpenRouter経由、Cerebras固定・fallback無効)
