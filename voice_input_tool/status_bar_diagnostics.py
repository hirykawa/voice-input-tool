"""Diagnostics for rumps status bar initialization."""

import logging

log = logging.getLogger("voice_input")


def install_status_bar_diagnostics():
    """Wrap rumps status-bar creation with detailed logging."""
    try:
        import rumps.rumps as rumps_impl
    except Exception:
        log.exception("ステータスバー診断の初期化に失敗しました")
        return

    original = rumps_impl.NSApp.initializeStatusBar
    if getattr(original, "_voice_input_wrapped", False):
        return

    def wrapped_initialize_status_bar(self):
        log.info("ステータスバー初期化開始")
        try:
            self.nsstatusitem = rumps_impl.NSStatusBar.systemStatusBar().statusItemWithLength_(-1)
            log.info("ステータスバー項目作成完了")
            self.nsstatusitem.setHighlightMode_(True)
            log.info("ステータスバーハイライト設定完了")

            self.setStatusBarIcon()
            log.info("ステータスバーアイコン設定完了")
            self.setStatusBarTitle()
            log.info("ステータスバータイトル設定完了: title=%r", str(self.nsstatusitem.title()))

            mainmenu = self._app["_menu"]
            quit_button = self._app["_quit_button"]
            if quit_button is not None:
                quit_button.set_callback(rumps_impl.quit_application)
                mainmenu.add(quit_button)
                log.info("終了メニュー追加完了")
            else:
                log.info("rumps標準の終了メニューは無効です（アプリ側のメニューを使用）")

            self.nsstatusitem.setMenu_(mainmenu._menu)
            log.info("ステータスバーメニュー設定完了")

            menu = self.nsstatusitem.menu()
            menu_items = menu.numberOfItems() if menu is not None else -1
            is_visible = self.nsstatusitem.isVisible() if hasattr(self.nsstatusitem, "isVisible") else "unknown"
            log.info(
                "ステータスバー初期化完了: title=%r menu_items=%s visible=%s",
                str(self.nsstatusitem.title()),
                menu_items,
                is_visible,
            )
        except BaseException:
            log.exception("ステータスバー初期化に失敗しました")
            raise

    wrapped_initialize_status_bar._voice_input_wrapped = True
    rumps_impl.NSApp.initializeStatusBar = wrapped_initialize_status_bar
