"""Input device helpers backed by sounddevice."""

import logging

import sounddevice as sd

log = logging.getLogger("voice_input")


def list_input_devices():
    devices = []
    try:
        for index, device in enumerate(sd.query_devices()):
            channels = int(device.get("max_input_channels", 0))
            if channels <= 0:
                continue
            name = str(device.get("name", f"Input {index}"))
            devices.append(
                {
                    "id": str(index),
                    "label": f"{name} ({channels}ch)",
                }
            )
    except Exception:
        return []
    return devices


def _first_input_device():
    for index, device in enumerate(sd.query_devices()):
        if int(device.get("max_input_channels", 0)) > 0:
            return index, device
    return None, None


def resolve_input_device(device_id):
    """Convert a saved device id into a sounddevice InputStream value."""
    try:
        if device_id in (None, ""):
            default_input = sd.default.device[0]
            if default_input is not None and int(default_input) >= 0:
                device = sd.query_devices(int(default_input))
                if int(device.get("max_input_channels", 0)) > 0:
                    return int(default_input)

            index, device = _first_input_device()
            if device is not None:
                log.warning(
                    "既定入力マイクが使えないため別の入力を使用します: %s",
                    device.get("name", index),
                )
                return index
            return None

        index = int(device_id)
        if index < 0:
            return resolve_input_device("")

        device = sd.query_devices(index)
        if int(device.get("max_input_channels", 0)) <= 0:
            raise ValueError("入力チャンネルがありません")
        return index
    except Exception as e:
        log.error("入力マイク設定を使用できません: id=%s error=%s", device_id, e)
        index, device = _first_input_device()
        if device is not None:
            log.warning("入力マイク設定の代わりに別の入力を使用します: %s", device.get("name", index))
            return index
        return None
