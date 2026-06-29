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
- macOS メニューバーアプリとして常駐
- ネイティブ設定画面（PyObjC）でGUIから設定変更可能
- グローバルホットキー対応（設定画面でカスタマイズ可能）
- 認識結果はカーソル位置に入力（貼り付け不可時は補正済みテキストをクリップボードに保持）
- 詳細ログを `logs/` に出力

## 社員向けセットアップ（初回）

この手順は、他の社員が自分のMacで最短で使い始めるためのものです。

> 重要: 現在の既定設定では、ツール本体・設定・モデルを `~/voice-input-tool` に置く前提です。別の場所に置く場合は、`voice_input.py` / `voice_input_tool/config.py` のパス設定も変更してください。

### 0. 事前に必要なもの

- macOS
- Python 3.11 推奨（3.10以上でも動作想定）
- ターミナル操作権限
- 空き容量 2GB 以上（ASRモデルが約1.4GB）
- LLM補正を使う場合: OpenRouter API Key

Python が入っているか確認します。`python3.11` があれば優先して使います。

```bash
if command -v python3.11 >/dev/null; then python3.11 --version; else python3 --version; fi
```

Python がない、または古い場合は、社内の標準手順に従って Python を入れてください。Homebrew を使える環境なら次で入れられます。

```bash
brew install python@3.11
```

`git` やビルドツールがない場合は、以下を実行して Command Line Tools を入れます。

```bash
xcode-select --install
```

### 1. ツール本体を配置する

社内Gitから取得する場合:

```bash
cd ~
git clone <社内GitリポジトリURL> voice-input-tool
cd ~/voice-input-tool
```

`<社内GitリポジトリURL>` は、共有時に実際のURLへ置き換えてください。

ZIPや社内ファイル共有で受け取った場合は、展開後のフォルダ名を `voice-input-tool` にして、ホーム直下に置いてください。

```bash
cd ~
# 例: Downloads に展開された場合
mv ~/Downloads/voice-input-tool ~/voice-input-tool
cd ~/voice-input-tool
```

### 2. Python仮想環境を作る

```bash
cd ~/voice-input-tool
PYTHON_BIN="$(command -v python3.11 || command -v python3)"
"$PYTHON_BIN" -m venv .venv-framework
source .venv-framework/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2回目以降に作業する場合も、コマンド実行前は次で仮想環境を有効化します。

```bash
cd ~/voice-input-tool
source .venv-framework/bin/activate
```

### 3. 音声認識モデルをダウンロードする

```bash
mkdir -p ~/voice-input-tool/models
cd ~/voice-input-tool/models

# ASRモデル (ReazonSpeech K2 v2 int8)
curl -L -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2
tar xjf sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2

# VADモデル (Silero)
curl -L -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

cd ~/voice-input-tool
```

モデル配置を確認します。

```bash
test -f models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/encoder-epoch-99-avg-1.int8.onnx && \
test -f models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/decoder-epoch-99-avg-1.int8.onnx && \
test -f models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/joiner-epoch-99-avg-1.int8.onnx && \
test -f models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/tokens.txt && \
test -f models/silero_vad.onnx && \
echo "モデル配置 OK"
```

社内で複数人がセットアップする場合は、上記2ファイル（`.tar.bz2` と `silero_vad.onnx`）を社内ファイル共有に置いてからコピーすると、各自のダウンロード時間を短縮できます。

### 4. OpenRouter API Keyを設定する（LLM補正を使う場合）

LLM補正を使う場合は、`.env` に `OPENROUTER_API_KEY` を保存します。`.env` はGit管理対象外です。

```bash
cd ~/voice-input-tool
cat > .env <<'EOF'
OPENROUTER_API_KEY=sk-or-v1-xxxxx
EOF
chmod 600 .env
```

`sk-or-v1-xxxxx` は各自のAPI Keyに置き換えてください。設定画面からAPI Keyを入力することもできます。その場合も `.env` に保存され、`config.json` には保存されません。

LLM補正を使わない場合は、起動後にメニューバーの `LLM補正: ON/OFF` でOFFにしてください。API Key未設定のままLLM補正ONで使うと、補正に失敗してテキストが出力されません。

### 5. 起動前テストを行う

```bash
cd ~/voice-input-tool
source .venv-framework/bin/activate
python voice_input.py --test --no-llm
```

ASRのテスト結果が表示されれば、Python依存関係とモデル配置は正常です。

### 6. 起動する

ターミナルから起動する場合:

```bash
cd ~/voice-input-tool
chmod +x start.sh
./start.sh
```

起動すると、メニューバーに 🎙 アイコンが表示されます。`~/Applications/VoiceInputTool.app` がある場合は、`./start.sh` からそのアプリを起動します。

設定でLLM補正をOFFにした後、一時的にLLM補正ONで起動したい場合は次を使います。

```bash
./start.sh --llm
```

### 7. macOSの権限を許可する

初回起動時、または初回録音時に権限許可が必要です。

| 権限 | 用途 | 許可する対象 |
|------|------|--------------|
| マイク | 音声入力 | `VoiceInputTool.app` / ターミナル / iTerm |
| アクセシビリティ | ホットキー、カーソル位置への入力 | `VoiceInputTool.app` / ターミナル / iTerm |
| 入力監視 | ホットキーが反応しない場合に必要なことがあります | `VoiceInputTool.app` / ターミナル / iTerm |

設定場所:

1. macOSの「システム設定」を開く
2. 「プライバシーとセキュリティ」を開く
3. 「マイク」「アクセシビリティ」「入力監視」を確認
4. 起動に使っているアプリ（`VoiceInputTool.app`、Terminal、iTermなど）を許可
5. 権限を変更したら、Voice Input Toolを一度終了して再起動

### 8. 動作確認

1. 入力したいアプリ（Slack、Notion、ブラウザ、エディタなど）のテキスト欄にカーソルを置く
2. `Ctrl+Shift+Space` を押す、またはメニューバーの 🎙 →「録音開始」を選ぶ
3. 話す
4. 話している間、メニューバーが三つの白い丸の入力中アイコンに変わることを確認
5. 少し無音にすると、認識・補正後のテキストがカーソル位置に入力される
6. 終了するときは、もう一度 `Ctrl+Shift+Space` を押す、またはメニューから停止する

### 9. ダブルクリックで起動する

同梱のネイティブアプリを `~/Applications/VoiceInputTool.app` に配置すると、Finderからダブルクリックで起動できます。デスクトップにある `VoiceInputTool.app` は、このアプリへのショートカットです。

ログイン時に自動起動したい場合は、macOSの「システム設定」→「一般」→「ログイン項目」に `VoiceInputTool.app` を追加してください。

### 10. 更新手順

社内Gitから取得している場合:

```bash
cd ~/voice-input-tool
git pull
source .venv-framework/bin/activate
python -m pip install -r requirements.txt --upgrade
```

モデル更新が案内された場合だけ、手順3のモデルダウンロードを再実行してください。

## 使い方

### メニューバーから操作

メニューバーの 🎙 アイコンをクリックします。

| メニュー項目 | 説明 |
|-------------|------|
| 録音開始/停止 | 録音をトグル |
| LLM補正: ON/OFF | LLM補正の有効/無効を切替 |
| アクセシビリティ許可を確認 | 自動入力に必要なmacOS権限を確認 |
| 設定... | 設定画面を開く |
| 終了 | アプリを終了 |

### CLIから使う

```bash
cd ~/voice-input-tool
source .venv-framework/bin/activate

# メニューバーアプリとして起動
./start.sh

# テストモード（WAVファイルでASR動作確認）
python voice_input.py --test --no-llm
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
| ![入力中](assets/typing-indicator.svg) | 音声入力中 |
| 📝 | 音声認識中 |
| 🧠 | LLM補正中 |
| ⌨️ | カーソル位置へ入力中 |

入力待機中から音声入力中への切り替えは、マイク入力の音量とノイズ床をもとに判定します。音声入力中は三つの白い丸が順番に跳ねるインジケーターを表示します。

## 設定画面

メニューバーの 🎙 →「設定...」で開きます。

| 設定項目 | 説明 | デフォルト |
|---------|------|-----------|
| LLM 句読点補正 | ON/OFF切替 | ON |
| OpenRouter API Key | LLM補正用APIキー | `.env` から読み込み |
| 録音 開始/停止 | ホットキー設定 | Ctrl+Shift+Space |
| 入力マイク | 使用するマイク。未選択時はシステムの既定入力 | 自動選択 |
| VAD 発話検出閾値 | 発話と判定する閾値 (0.1-0.9) | 0.5 |
| 無音判定時間 | 発話終了と判断する無音の長さ (0.2-3.0秒) | 0.8秒 |
| 最小発話長 | 認識対象とする最短の発話長 (0.1-2.0秒) | 0.3秒 |
| LLM 補正プロンプト | LLM補正時のシステムプロンプト | 句読点挿入のみ |

設定は `~/voice-input-tool/config.json` に保存されます。API Key は `.env` に保存され、`config.json` には保存されません。

## LLM設定

既定値は `voice_input_tool/config.py` の `DEFAULTS` で管理しています。

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

初期セットアップで詰まりやすい点:

| 症状 | 確認すること |
|------|--------------|
| `ModuleNotFoundError` が出る | `cd ~/voice-input-tool` → `source .venv-framework/bin/activate` → `python -m pip install -r requirements.txt` を再実行 |
| `モデルファイルが見つかりません` と出る | `models/` 配下に ASRモデル展開済みフォルダと `silero_vad.onnx` があるか確認 |
| メニューバーに何も出ない | `~/Applications/VoiceInputTool.app` を起動しているか確認。`logs/native-status.log` と `logs/native-engine.err.log` も確認 |
| マイク入力できない | macOSの「マイク」権限で、起動に使っているアプリを許可して再起動 |
| `Ctrl+Shift+Space` が効かない | 「アクセシビリティ」と、必要に応じて「入力監視」を許可して再起動。別アプリのショートカットと衝突していないかも確認 |
| カーソル位置に入力されない | 「アクセシビリティ」権限を確認。貼り付け不可のアプリではクリップボードに残ります |
| LLM補正で出力されない | `.env` の `OPENROUTER_API_KEY`、OpenRouterの残高、Cerebras providerの利用可否を確認。急ぎの場合はメニューでLLM補正をOFF |

## ファイル構成

```
voice_input.py       - CLIエントリーポイント（互換用ラッパー）
voice_input_tool/    - Python実装（engine/ASR/VAD/LLM/設定UI/ネイティブ連携）
native/              - macOSネイティブ常駐アプリ/ランチャーのソース
packaging/           - VoiceInputTool.app 用 Info.plist
launch-agents/       - 自動起動用 LaunchAgent サンプル
assets/              - メニューバー用アイコン素材
start.sh             - CLI起動用シェルスクリプト
requirements.txt     - Python依存パッケージ一覧
models/              - ASR・VADモデル（.gitignore済み）
logs/                - ログファイル（.gitignore済み）
config.json          - ユーザー設定（.gitignore済み）
.env                 - 環境変数（.gitignore済み）
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
