"""User notification helpers."""

import logging
import threading

import rumps

log = logging.getLogger("voice_input")

try:
    from PyObjCTools import AppHelper

    HAS_APP_HELPER = True
except ImportError:
    HAS_APP_HELPER = False


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
