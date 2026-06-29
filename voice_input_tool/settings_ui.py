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
    NSPopUpButton,
    NSTextView,
    NSScrollView,
    NSFont,
    NSColor,
    NSBezelStyleRounded,
    NSMakeRect,
    NSObject,
    NSApp,
    NSApplicationActivationPolicyAccessory,
    NSControlStateValueOn,
    NSControlStateValueOff,
    NSEventModifierFlagControl,
    NSEventModifierFlagShift,
    NSEventModifierFlagOption,
    NSEventModifierFlagCommand,
)
from voice_input_tool.audio_devices import list_input_devices
from voice_input_tool.config import load_config, save_config

# pynput のキー名を表示用に変換
DISPLAY_KEY_MAP = {
    "<ctrl>": "Ctrl",
    "<shift>": "Shift",
    "<alt>": "Alt",
    "<cmd>": "Cmd",
    "<space>": "Space",
    "<tab>": "Tab",
    "<enter>": "Enter",
    "<backspace>": "Backspace",
    "<esc>": "Esc",
    "<left>": "Left",
    "<right>": "Right",
    "<up>": "Up",
    "<down>": "Down",
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
    _hotkey_value = objc.ivar()
    _previous_display = objc.ivar()
    _is_recording = objc.ivar()

    def initWithFrame_(self, frame):
        self = objc.super(HotkeyField, self).initWithFrame_(frame)
        if self is None:
            return None
        self._hotkey_value = None
        self._previous_display = ""
        self._is_recording = False
        self.setEditable_(False)
        self.setSelectable_(False)
        self.setBezeled_(True)
        self.setDrawsBackground_(True)
        self.setBackgroundColor_(NSColor.textBackgroundColor())
        self.setFont_(NSFont.systemFontOfSize_(13))
        self.setAlignment_(1)  # center
        return self

    def acceptsFirstResponder(self):
        return True

    def mouseDown_(self, event):
        window = self.window()
        if window is not None:
            window.makeFirstResponder_(self)

    def becomeFirstResponder(self):
        self._is_recording = True
        self._previous_display = str(self.stringValue())
        self.setStringValue_("キーを押してください...")
        self.setTextColor_(NSColor.systemBlueColor())
        return True

    def resignFirstResponder(self):
        if self._is_recording:
            self.setStringValue_(self._previous_display)
        self._is_recording = False
        self.setTextColor_(NSColor.labelColor())
        return True

    def keyDown_(self, event):
        hotkey = self._hotkey_from_event(event)
        if hotkey is None:
            return
        self.setHotkeyValue_(hotkey)
        self._is_recording = False
        window = self.window()
        if window is not None:
            window.makeFirstResponder_(None)

    def performKeyEquivalent_(self, event):
        if not self._is_recording:
            return False
        self.keyDown_(event)
        return True

    def flagsChanged_(self, event):
        pass  # 修飾キー単体は無視

    def getHotkeyValue(self):
        return self._hotkey_value

    def setHotkeyValue_(self, value):
        self._hotkey_value = value
        display = hotkey_to_display(value)
        self._previous_display = display
        self.setStringValue_(display)
        self.setTextColor_(NSColor.labelColor())

    def _hotkey_from_event(self, event):
        mods = int(event.modifierFlags())
        parts = []
        if mods & NSEventModifierFlagControl:
            parts.append("<ctrl>")
        if mods & NSEventModifierFlagShift:
            parts.append("<shift>")
        if mods & NSEventModifierFlagOption:
            parts.append("<alt>")
        if mods & NSEventModifierFlagCommand:
            parts.append("<cmd>")

        if not parts:
            return None

        key_name = self._key_name_from_event(event)
        if key_name is None:
            return None

        return "+".join(parts + [key_name])

    def _key_name_from_event(self, event):
        key_map = {
            49: "<space>", 48: "<tab>", 36: "<enter>", 51: "<backspace>",
            53: "<esc>", 123: "<left>", 124: "<right>", 125: "<down>", 126: "<up>",
            0: "a", 1: "s", 2: "d", 3: "f", 4: "h", 5: "g",
            6: "z", 7: "x", 8: "c", 9: "v", 11: "b",
            12: "q", 13: "w", 14: "e", 15: "r", 16: "y", 17: "t",
            31: "o", 32: "i", 33: "[", 34: "p", 35: "]",
            37: "l", 38: "j", 39: "'", 40: "k", 41: ";",
            45: "n", 46: "m", 18: "1", 19: "2", 20: "3", 21: "4",
            23: "5", 22: "6", 26: "7", 28: "8", 25: "9", 29: "0",
            27: "-", 24: "=", 42: "\\", 43: ",", 47: ".", 44: "/", 50: "`",
            94: "_",
        }
        keycode = int(event.keyCode())
        if keycode in key_map:
            return key_map[keycode]

        chars = event.charactersIgnoringModifiers()
        if chars and len(chars) == 1 and chars.isprintable():
            return str(chars).lower()
        return f"key{keycode}"


class SettingsWindowController(NSObject):
    """設定ウィンドウ"""

    window = objc.ivar()
    llm_checkbox = objc.ivar()
    api_key_field = objc.ivar()
    hotkey_record_field = objc.ivar()
    input_device_popup = objc.ivar()
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
        W, H = 500, 520

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
        content.addSubview_(_hint_label("クリック後、Ctrl/Shift等とキーを押す", 380, y - 2))
        y -= 45

        # --- 入力マイク ---
        content.addSubview_(_label("入力マイク", 20, y))
        self.input_device_popup = NSPopUpButton.alloc().initWithFrame_pullsDown_(
            NSMakeRect(170, y - 3, 300, 26),
            False,
        )
        self._populate_input_devices(config.get("input_device_id", ""))
        content.addSubview_(self.input_device_popup)
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

    def _populate_input_devices(self, selected_device_id):
        self.input_device_popup.removeAllItems()
        self.input_device_popup.addItemWithTitle_("自動選択")
        self.input_device_popup.itemAtIndex_(0).setRepresentedObject_("")

        selected_index = 0
        for device in list_input_devices():
            self.input_device_popup.addItemWithTitle_(device["label"])
            item_index = self.input_device_popup.numberOfItems() - 1
            item = self.input_device_popup.itemAtIndex_(item_index)
            item.setRepresentedObject_(device["id"])
            if device["id"] == str(selected_device_id):
                selected_index = item_index

        if self.input_device_popup.numberOfItems() == 1:
            self.input_device_popup.addItemWithTitle_("入力デバイスが見つかりません")
            self.input_device_popup.itemAtIndex_(1).setRepresentedObject_("")
            self.input_device_popup.setEnabled_(False)

        self.input_device_popup.selectItemAtIndex_(selected_index)

    @objc.typedSelector(b"v@:@")
    def saveClicked_(self, sender):
        config = load_config()
        hotkey_val = self.hotkey_record_field.getHotkeyValue()
        input_device_id = ""
        selected_device = self.input_device_popup.selectedItem()
        if selected_device is not None and selected_device.representedObject() is not None:
            input_device_id = str(selected_device.representedObject())
        config.update({
            "use_llm": self.llm_checkbox.state() == NSControlStateValueOn,
            "openrouter_api_key": str(self.api_key_field.stringValue()),
            "hotkey_record": hotkey_val if hotkey_val else "<ctrl>+<shift>+<space>",
            "input_device_id": input_device_id,
            "llm_prompt": str(self.prompt_textview.string()),
        })
        save_config(config)
        if self.on_save_callback:
            self.on_save_callback(config)
        self.window.close()

    @objc.typedSelector(b"v@:@")
    def cancelClicked_(self, sender):
        self.window.close()

    @objc.typedSelector(b"v@:@")
    def resetClicked_(self, sender):
        from voice_input_tool.config import DEFAULTS
        self.llm_checkbox.setState_(NSControlStateValueOn if DEFAULTS["use_llm"] else NSControlStateValueOff)
        self.hotkey_record_field.setHotkeyValue_(DEFAULTS["hotkey_record"])
        self._populate_input_devices(DEFAULTS["input_device_id"])
        self.prompt_textview.setString_(DEFAULTS["llm_prompt"])

    def show(self):
        self.window.center()
        self.window.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)
