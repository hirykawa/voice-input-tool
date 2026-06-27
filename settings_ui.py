"""ネイティブ macOS 設定画面 (PyObjC)"""
import threading
import objc
from Cocoa import (
    NSApplication,
    NSWindow,
    NSWindowStyleMaskTitled,
    NSWindowStyleMaskClosable,
    NSBackingStoreBuffered,
    NSTextField,
    NSSecureTextField,
    NSButton,
    NSButtonTypeSwitch,
    NSTextView,
    NSScrollView,
    NSFont,
    NSColor,
    NSBezelStyleRounded,
    NSMakeRect,
    NSObject,
    NSApp,
    NSApplicationActivationPolicyAccessory,
    NSSlider,
    NSControlStateValueOn,
    NSControlStateValueOff,
)
from config import load_config, save_config

# pynput のキー名を表示用に変換
DISPLAY_KEY_MAP = {
    "<ctrl>": "Ctrl",
    "<shift>": "Shift",
    "<alt>": "Alt",
    "<cmd>": "Cmd",
    "<space>": "Space",
    "<tab>": "Tab",
    "<enter>": "Enter",
}


def hotkey_to_display(hotkey_str):
    """pynput形式のホットキー文字列を表示用に変換"""
    parts = hotkey_str.split("+")
    display_parts = []
    for p in parts:
        p = p.strip()
        display_parts.append(DISPLAY_KEY_MAP.get(p, p.strip("<>")))
    return " + ".join(display_parts)


def display_to_hotkey(display_str):
    """表示用文字列をpynput形式に変換"""
    reverse_map = {v.lower(): k for k, v in DISPLAY_KEY_MAP.items()}
    parts = [p.strip() for p in display_str.split("+")]
    hotkey_parts = []
    for p in parts:
        key = reverse_map.get(p.lower(), p.lower())
        hotkey_parts.append(key)
    return "+".join(hotkey_parts)


def _label(text, x, y, width=140):
    label = NSTextField.alloc().initWithFrame_(NSMakeRect(x, y, width, 20))
    label.setStringValue_(text)
    label.setBezeled_(False)
    label.setDrawsBackground_(False)
    label.setEditable_(False)
    label.setSelectable_(False)
    label.setFont_(NSFont.systemFontOfSize_(13))
    return label


def _value_label(text, x, y, width=60):
    label = NSTextField.alloc().initWithFrame_(NSMakeRect(x, y, width, 20))
    label.setStringValue_(text)
    label.setBezeled_(False)
    label.setDrawsBackground_(False)
    label.setEditable_(False)
    label.setSelectable_(False)
    label.setFont_(NSFont.monospacedDigitSystemFontOfSize_weight_(12, 0.0))
    label.setAlignment_(2)  # right
    return label


def _hint_label(text, x, y, width=300):
    label = NSTextField.alloc().initWithFrame_(NSMakeRect(x, y, width, 14))
    label.setStringValue_(text)
    label.setBezeled_(False)
    label.setDrawsBackground_(False)
    label.setEditable_(False)
    label.setSelectable_(False)
    label.setFont_(NSFont.systemFontOfSize_(10))
    label.setTextColor_(NSColor.secondaryLabelColor())
    return label


class HotkeyField(NSTextField):
    """キー入力をキャプチャするカスタムテキストフィールド"""
    _captured_keys = objc.ivar()
    _modifiers = objc.ivar()

    def initWithFrame_(self, frame):
        self = objc.super(HotkeyField, self).initWithFrame_(frame)
        if self is None:
            return None
        self._captured_keys = set()
        self._modifiers = set()
        self.setEditable_(False)
        self.setSelectable_(False)
        self.setFont_(NSFont.systemFontOfSize_(13))
        self.setAlignment_(1)  # center
        return self

    def acceptsFirstResponder(self):
        return True

    def becomeFirstResponder(self):
        self.setStringValue_("キーを押してください...")
        self.setTextColor_(NSColor.systemBlueColor())
        return objc.super(HotkeyField, self).becomeFirstResponder()

    def resignFirstResponder(self):
        self.setTextColor_(NSColor.labelColor())
        return objc.super(HotkeyField, self).resignFirstResponder()

    def keyDown_(self, event):
        mods = event.modifierFlags()
        keycode = event.keyCode()

        parts = []
        if mods & (1 << 18):  # Control
            parts.append("<ctrl>")
        if mods & (1 << 17):  # Shift
            parts.append("<shift>")
        if mods & (1 << 19):  # Alt/Option
            parts.append("<alt>")
        if mods & (1 << 20):  # Command
            parts.append("<cmd>")

        # Map keycode to key name
        key_map = {
            49: "<space>", 48: "<tab>", 36: "<enter>",
            0: "a", 1: "s", 2: "d", 3: "f", 4: "h", 5: "g",
            6: "z", 7: "x", 8: "c", 9: "v", 11: "b",
            12: "q", 13: "w", 14: "e", 15: "r", 16: "y", 17: "t",
            31: "o", 32: "i", 33: "[", 34: "p", 35: "]",
            37: "l", 38: "j", 39: "'", 40: "k", 41: ";",
            45: "n", 46: "m",
        }
        key_name = key_map.get(keycode, f"key{keycode}")

        if not parts:
            return  # 修飾キーなしは無視

        parts.append(key_name)
        hotkey = "+".join(parts)
        self.setStringValue_(hotkey_to_display(hotkey))
        self.setTextColor_(NSColor.labelColor())
        # Store the pynput format
        self._hotkey_value = hotkey

    def flagsChanged_(self, event):
        pass  # 修飾キー単体は無視

    def getHotkeyValue(self):
        return getattr(self, '_hotkey_value', None)

    def setHotkeyValue_(self, value):
        self._hotkey_value = value
        self.setStringValue_(hotkey_to_display(value))


class SettingsWindowController(NSObject):
    """設定ウィンドウ"""

    window = objc.ivar()
    llm_checkbox = objc.ivar()
    api_key_field = objc.ivar()
    vad_threshold_slider = objc.ivar()
    vad_threshold_label = objc.ivar()
    vad_silence_slider = objc.ivar()
    vad_silence_label = objc.ivar()
    vad_min_speech_slider = objc.ivar()
    vad_min_speech_label = objc.ivar()
    hotkey_record_field = objc.ivar()
    prompt_textview = objc.ivar()
    on_save_callback = objc.ivar()

    def initWithCallback_(self, callback):
        self = objc.super(SettingsWindowController, self).init()
        if self is None:
            return None
        self.on_save_callback = callback
        self._build_window()
        return self

    def _build_window(self):
        config = load_config()
        W, H = 500, 600

        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(200, 200, W, H),
            NSWindowStyleMaskTitled | NSWindowStyleMaskClosable,
            NSBackingStoreBuffered,
            False,
        )
        self.window.setTitle_("Voice Input Tool - 設定")
        self.window.setLevel_(3)  # floating
        content = self.window.contentView()
        y = H - 50

        # --- LLM 補正 ---
        content.addSubview_(_label("LLM 句読点補正", 20, y))
        self.llm_checkbox = NSButton.alloc().initWithFrame_(NSMakeRect(170, y - 2, 200, 24))
        self.llm_checkbox.setButtonType_(NSButtonTypeSwitch)
        self.llm_checkbox.setTitle_("有効")
        self.llm_checkbox.setState_(NSControlStateValueOn if config["use_llm"] else NSControlStateValueOff)
        content.addSubview_(self.llm_checkbox)
        y -= 40

        # --- API Key ---
        content.addSubview_(_label("OpenRouter API Key", 20, y))
        self.api_key_field = NSSecureTextField.alloc().initWithFrame_(NSMakeRect(170, y - 2, 300, 24))
        self.api_key_field.setStringValue_(config.get("openrouter_api_key", ""))
        self.api_key_field.setPlaceholderString_("sk-or-v1-...")
        self.api_key_field.setFont_(NSFont.systemFontOfSize_(12))
        content.addSubview_(self.api_key_field)
        y -= 50

        # --- ホットキー: 録音開始/停止 ---
        content.addSubview_(_label("録音 開始/停止", 20, y))
        self.hotkey_record_field = HotkeyField.alloc().initWithFrame_(NSMakeRect(170, y - 2, 200, 24))
        self.hotkey_record_field.setHotkeyValue_(config.get("hotkey_record", "<ctrl>+<shift>+<space>"))
        content.addSubview_(self.hotkey_record_field)
        content.addSubview_(_hint_label("クリックしてキーを押して設定", 380, y - 2))
        y -= 45

        # --- VAD Threshold ---
        content.addSubview_(_label("VAD 発話検出閾値", 20, y))
        self.vad_threshold_label = _value_label(f"{config['vad_threshold']:.2f}", 420, y)
        content.addSubview_(self.vad_threshold_label)
        self.vad_threshold_slider = NSSlider.alloc().initWithFrame_(NSMakeRect(170, y, 240, 24))
        self.vad_threshold_slider.setMinValue_(0.1)
        self.vad_threshold_slider.setMaxValue_(0.9)
        self.vad_threshold_slider.setDoubleValue_(config["vad_threshold"])
        self.vad_threshold_slider.setTarget_(self)
        self.vad_threshold_slider.setAction_(b"sliderChanged:")
        self.vad_threshold_slider.setTag_(1)
        content.addSubview_(self.vad_threshold_slider)
        y -= 40

        # --- VAD Silence Duration ---
        content.addSubview_(_label("無音判定時間 (秒)", 20, y))
        self.vad_silence_label = _value_label(f"{config['vad_silence_duration']:.1f}", 420, y)
        content.addSubview_(self.vad_silence_label)
        self.vad_silence_slider = NSSlider.alloc().initWithFrame_(NSMakeRect(170, y, 240, 24))
        self.vad_silence_slider.setMinValue_(0.2)
        self.vad_silence_slider.setMaxValue_(3.0)
        self.vad_silence_slider.setDoubleValue_(config["vad_silence_duration"])
        self.vad_silence_slider.setTarget_(self)
        self.vad_silence_slider.setAction_(b"sliderChanged:")
        self.vad_silence_slider.setTag_(2)
        content.addSubview_(self.vad_silence_slider)
        y -= 40

        # --- VAD Min Speech ---
        content.addSubview_(_label("最小発話長 (秒)", 20, y))
        self.vad_min_speech_label = _value_label(f"{config['vad_min_speech']:.1f}", 420, y)
        content.addSubview_(self.vad_min_speech_label)
        self.vad_min_speech_slider = NSSlider.alloc().initWithFrame_(NSMakeRect(170, y, 240, 24))
        self.vad_min_speech_slider.setMinValue_(0.1)
        self.vad_min_speech_slider.setMaxValue_(2.0)
        self.vad_min_speech_slider.setDoubleValue_(config["vad_min_speech"])
        self.vad_min_speech_slider.setTarget_(self)
        self.vad_min_speech_slider.setAction_(b"sliderChanged:")
        self.vad_min_speech_slider.setTag_(3)
        content.addSubview_(self.vad_min_speech_slider)
        y -= 45

        # --- LLM Prompt ---
        content.addSubview_(_label("LLM 補正プロンプト", 20, y))
        y -= 10
        scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(20, y - 140, 460, 150))
        scroll.setHasVerticalScroller_(True)
        scroll.setBorderType_(3)  # NSBezelBorder
        self.prompt_textview = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, 440, 150))
        self.prompt_textview.setString_(config.get("llm_prompt", ""))
        self.prompt_textview.setFont_(NSFont.systemFontOfSize_(12))
        self.prompt_textview.setMinSize_((440, 150))
        self.prompt_textview.setMaxSize_((440, 10000))
        self.prompt_textview.setVerticallyResizable_(True)
        self.prompt_textview.textContainer().setWidthTracksTextView_(True)
        scroll.setDocumentView_(self.prompt_textview)
        content.addSubview_(scroll)
        y -= 160

        # --- Buttons ---
        save_btn = NSButton.alloc().initWithFrame_(NSMakeRect(W - 200, 15, 80, 32))
        save_btn.setTitle_("保存")
        save_btn.setBezelStyle_(NSBezelStyleRounded)
        save_btn.setTarget_(self)
        save_btn.setAction_(b"saveClicked:")
        save_btn.setKeyEquivalent_("\r")
        content.addSubview_(save_btn)

        cancel_btn = NSButton.alloc().initWithFrame_(NSMakeRect(W - 110, 15, 80, 32))
        cancel_btn.setTitle_("キャンセル")
        cancel_btn.setBezelStyle_(NSBezelStyleRounded)
        cancel_btn.setTarget_(self)
        cancel_btn.setAction_(b"cancelClicked:")
        cancel_btn.setKeyEquivalent_("\x1b")  # Escape
        content.addSubview_(cancel_btn)

        reset_btn = NSButton.alloc().initWithFrame_(NSMakeRect(20, 15, 120, 32))
        reset_btn.setTitle_("デフォルトに戻す")
        reset_btn.setBezelStyle_(NSBezelStyleRounded)
        reset_btn.setTarget_(self)
        reset_btn.setAction_(b"resetClicked:")
        content.addSubview_(reset_btn)

    @objc.typedSelector(b"v@:@")
    def sliderChanged_(self, sender):
        tag = sender.tag()
        if tag == 1:
            self.vad_threshold_label.setStringValue_(f"{sender.doubleValue():.2f}")
        elif tag == 2:
            self.vad_silence_label.setStringValue_(f"{sender.doubleValue():.1f}")
        elif tag == 3:
            self.vad_min_speech_label.setStringValue_(f"{sender.doubleValue():.1f}")

    @objc.typedSelector(b"v@:@")
    def saveClicked_(self, sender):
        hotkey_val = self.hotkey_record_field.getHotkeyValue()
        config = {
            "use_llm": self.llm_checkbox.state() == NSControlStateValueOn,
            "openrouter_api_key": str(self.api_key_field.stringValue()),
            "hotkey_record": hotkey_val if hotkey_val else "<ctrl>+<shift>+<space>",
            "vad_threshold": round(self.vad_threshold_slider.doubleValue(), 2),
            "vad_silence_duration": round(self.vad_silence_slider.doubleValue(), 1),
            "vad_min_speech": round(self.vad_min_speech_slider.doubleValue(), 1),
            "llm_prompt": str(self.prompt_textview.string()),
        }
        save_config(config)
        if self.on_save_callback:
            self.on_save_callback(config)
        self.window.close()

    @objc.typedSelector(b"v@:@")
    def cancelClicked_(self, sender):
        self.window.close()

    @objc.typedSelector(b"v@:@")
    def resetClicked_(self, sender):
        from config import DEFAULTS
        self.llm_checkbox.setState_(NSControlStateValueOn if DEFAULTS["use_llm"] else NSControlStateValueOff)
        self.hotkey_record_field.setHotkeyValue_(DEFAULTS["hotkey_record"])
        self.vad_threshold_slider.setDoubleValue_(DEFAULTS["vad_threshold"])
        self.vad_threshold_label.setStringValue_(f"{DEFAULTS['vad_threshold']:.2f}")
        self.vad_silence_slider.setDoubleValue_(DEFAULTS["vad_silence_duration"])
        self.vad_silence_label.setStringValue_(f"{DEFAULTS['vad_silence_duration']:.1f}")
        self.vad_min_speech_slider.setDoubleValue_(DEFAULTS["vad_min_speech"])
        self.vad_min_speech_label.setStringValue_(f"{DEFAULTS['vad_min_speech']:.1f}")
        self.prompt_textview.setString_(DEFAULTS["llm_prompt"])

    def show(self):
        self.window.center()
        self.window.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)
