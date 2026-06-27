# Voice Input Tool

ReazonSpeech (sherpa-onnx) + Silero VAD + オプションLLM補正による
ローカル音声入力ツール。Mac Mini (Apple Silicon) 向け。

## 特徴

- 完全ローカルASR（ReazonSpeech K2 v2 int8）
- Silero VAD による発話区間自動検出
- LLM補正オン/オフ切替可能（デフォルト: オフ）
- LLM補正は句読点挿入のみ（文章変更・言い換え禁止）
- OpenRouter GLM 5.2 で学習拒否設定（data_collection: deny / zdr: true）
- グローバルホットキー対応（Ctrl+Shift+Space で録音トグル）

## 使い方

### テスト（WAVファイルで動作確認）
```bash
cd ~/voice-input-tool
python3 voice_input.py --test
```

### リアルタイム音声入力（LLM補正なし）
```bash
python3 voice_input.py
```

### リアルタイム音声入力（LLM句読点補正あり）
```bash
python3 voice_input.py --llm
```

### ホットキー
- Ctrl+Shift+Space : 録音開始/停止
- Ctrl+Shift+Q     : 終了

## パフォーマンス（テスト結果）

| 項目 | 結果 |
|------|------|
| ASR処理速度 | 0.06〜0.16秒（10秒程度の音声） |
| モデル読み込み | 0.5秒 |
| 日本語精度 | テスト5問中ほぼ完璧 |
| LLM補正 | 句読点のみ挿入、文章変更なし |

## モデル

- ASR: sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01 (int8)
- VAD: silero_vad.onnx
- LLM: z-ai/glm-5.2 (OpenRouter経由、学習拒否設定付き)
