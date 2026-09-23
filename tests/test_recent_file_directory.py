import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from imr_gui.app import MainWindow, QFileDialog, QMessageBox


class RecentFileDirectoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / "experiment"
        self.folder.mkdir()
        self.settings = self.root / "settings.json"
        self.settings.write_text(json.dumps({"physics": {"rho": 998}, "ui": {"active_model": "NHKV"}}))
        self.window = SimpleNamespace(
            _last_file_directory="",
            _settings_path=lambda: self.settings,
        )
        self.window._file_dialog_start = lambda filename="": MainWindow._file_dialog_start(self.window, filename)
        self.window._remember_file_dialog_path = lambda path, **kwargs: MainWindow._remember_file_dialog_path(
            self.window, path, **kwargs
        )

    def test_remember_file_preserves_settings_and_builds_save_path(self):
        self.window._remember_file_dialog_path(str(self.folder / "data.mat"))
        data = json.loads(self.settings.read_text())
        self.assertEqual(data["physics"], {"rho": 998})
        self.assertEqual(data["ui"], {"active_model": "NHKV"})
        self.assertEqual(data["file_dialogs"]["last_directory"], str(self.folder))
        self.assertEqual(self.window._file_dialog_start(), str(self.folder))
        self.assertEqual(self.window._file_dialog_start("result.mat"), str(self.folder / "result.mat"))

    def test_file_menu_dialog_uses_last_selected_directory(self):
        source = str(self.folder / "params.mat")
        loaded = []
        self.window._load_params_from_path = loaded.append
        with patch.object(QFileDialog, "getOpenFileName", return_value=(source, "")) as dialog:
            MainWindow.on_load_params(self.window)
        self.assertEqual(dialog.call_args.args[2], "")
        self.assertEqual(loaded, [source])
        self.assertEqual(self.window._file_dialog_start(), str(self.folder))

        with patch.object(QFileDialog, "getOpenFileName", return_value=("", "")) as dialog:
            MainWindow.on_load_params(self.window)
        self.assertEqual(dialog.call_args.args[2], str(self.folder))

    def test_directory_choice_and_deleted_directory(self):
        self.window._remember_file_dialog_path(str(self.folder), directory=True)
        self.assertEqual(self.window._file_dialog_start(), str(self.folder))
        self.folder.rmdir()
        self.assertEqual(self.window._file_dialog_start(), "")

    def test_view_import_and_export_share_recent_directory(self):
        source = str(self.folder / "saved_view.mat")

        def fail_after_selection(_path):
            raise RuntimeError("stop after recording the selected directory")

        self.window._load_curve_view_mat = fail_after_selection
        with (
            patch.object(QFileDialog, "getOpenFileName", return_value=(source, "")) as dialog,
            patch.object(QMessageBox, "warning"),
        ):
            MainWindow._on_import_view_mat(self.window)

        self.assertEqual(dialog.call_args.args[2], "")
        self.assertEqual(self.window._file_dialog_start(), str(self.folder))

        exported = str(self.folder / "edited_view.svg")
        messages = []
        self.window._ensure_view_curves_for_output = lambda: True
        self.window._view_export_stem = lambda: "saved_view"
        self.window._with_curve_view_render = lambda _callback: None
        self.window.statusBar = lambda: SimpleNamespace(showMessage=messages.append)
        with patch.object(QFileDialog, "getSaveFileName", return_value=(exported, "")) as dialog:
            MainWindow._on_export_view_svg(self.window)

        self.assertEqual(dialog.call_args.args[2], str(self.folder / "saved_view.svg"))
        self.assertEqual(self.window._file_dialog_start(), str(self.folder))
        self.assertEqual(messages, [f"Exported Curve View SVG: {exported}"])


if __name__ == "__main__":
    unittest.main()
