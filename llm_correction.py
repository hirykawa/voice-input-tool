"""LLM punctuation correction via OpenRouter."""

import logging
import os

from log_utils import mask_secret, safe_model_dump, truncate_for_log

log = logging.getLogger("voice_input")

try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class LLMCorrectionError(RuntimeError):
    pass


_OPENROUTER_CLIENT = None
_OPENROUTER_CLIENT_KEY = None
_LLM_PROMPT = ""
_LLM_MODEL = "openai/gpt-oss-120b"
_LLM_PROVIDER_ORDER = ["Cerebras"]
_OPENROUTER_API_KEY = ""


def configure_llm(config):
    global _LLM_PROMPT, _LLM_MODEL, _LLM_PROVIDER_ORDER, _OPENROUTER_API_KEY
    _LLM_PROMPT = config["llm_prompt"]
    _LLM_MODEL = config.get("llm_model", _LLM_MODEL)
    _LLM_PROVIDER_ORDER = config.get("llm_provider_order", _LLM_PROVIDER_ORDER)
    _OPENROUTER_API_KEY = config.get("openrouter_api_key", "")


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
    """Insert punctuation through OpenRouter without falling back to raw ASR text."""
    if not text:
        return ""
    if not HAS_OPENAI:
        raise LLMCorrectionError("openai パッケージがインストールされていません")

    key = api_key or _OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise LLMCorrectionError("OPENROUTER_API_KEY が設定されていません")

    client = get_openrouter_client(key)
    max_tokens = min(2048, max(1024, len(text) + 256))
    extra_body = {
        "data_collection": "deny",
        "zdr": True,
        "provider": {
            "order": _LLM_PROVIDER_ORDER,
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
        _LLM_MODEL,
        _LLM_PROVIDER_ORDER,
        len(text),
        mask_secret(key),
        len(_LLM_PROMPT),
        max_tokens,
        truncate_for_log(text, 300),
    )

    try:
        response = client.chat.completions.create(
            model=_LLM_MODEL,
            messages=[
                {"role": "system", "content": _LLM_PROMPT},
                {"role": "user", "content": text},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
            extra_body=extra_body,
        )
        choices = getattr(response, "choices", []) or []
        if not choices:
            log.error("LLM応答choicesが空: raw=%s", truncate_for_log(safe_model_dump(response)))
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
            truncate_for_log(usage, 500),
            truncate_for_log(corrected or "", 500),
        )

        if corrected is None:
            log.error("LLM応答contentがNone: raw=%s", truncate_for_log(safe_model_dump(response)))
            raise LLMCorrectionError("LLM補正結果が空でした")
        corrected = corrected.strip()
        if not corrected:
            log.error("LLM応答contentが空: raw=%s", truncate_for_log(safe_model_dump(response)))
            raise LLMCorrectionError("LLM補正結果が空でした")
        return corrected
    except LLMCorrectionError:
        raise
    except Exception as e:
        log.exception("LLM補正API呼び出しエラー: model=%s providers=%s text_len=%d", _LLM_MODEL, _LLM_PROVIDER_ORDER, len(text))
        raise LLMCorrectionError(str(e)) from e
