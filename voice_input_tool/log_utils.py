"""Small helpers for safe structured logging."""


def truncate_for_log(value, limit=2000):
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"...<truncated {len(text) - limit} chars>"


def mask_secret(value):
    if not value:
        return ""
    value = str(value)
    if len(value) <= 8:
        return "***"
    return f"{value[:6]}...{value[-4:]}"


def safe_model_dump(obj):
    try:
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="json")
        if hasattr(obj, "dict"):
            return obj.dict()
    except Exception as e:
        return {"dump_error": repr(e), "repr": repr(obj)}
    return repr(obj)
