import os
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from imr_gui.app import MainWindow


ROOT = Path(__file__).resolve().parents[1]


class CurveViewRegressionTests(unittest.TestCase):
    def _load_all_data_view(self):
        loader = MainWindow.__new__(MainWindow)
        curves, _settings = MainWindow._load_curve_view_mat(
            loader,
            str(ROOT / "tests" / "AllData.mat")
        )
        return curves

    def test_hidden_panel_does_not_replace_imported_view_with_empty_canvas(self):
        state = SimpleNamespace(
            _force_curve_view_render=False,
            _multi_curve_enabled=True,
        )

        self.assertTrue(MainWindow._curve_view_active(state))

    def test_export_bootstraps_empty_curve_view_from_current_preview(self):
        state = SimpleNamespace(_view_curves=[])

        def seed_current_preview():
            state._view_curves.append({"legend": "exp data"})

        state._seed_view_curves_from_current_canvas = seed_current_preview

        self.assertTrue(MainWindow._ensure_view_curves_for_output(state))
        self.assertEqual(state._view_curves, [{"legend": "exp data"}])

    def test_imported_curve_view_preserves_black_colors(self):
        curves = self._load_all_data_view()
        expected = [str(curve["color"]).lower() for curve in curves]

        self.assertEqual(len(curves), 32)
        self.assertIn("#000000", expected)

    def test_executable_bundle_includes_curve_colors_and_svg_backend(self):
        spec = (ROOT / "IMR_GUI.spec").read_text(encoding="utf-8")

        self.assertIn('("imr_gui/view_colors.json", "imr_gui")', spec)
        self.assertIn('"matplotlib.backends.backend_svg"', spec)

    def test_missing_color_file_falls_back_to_black_for_first_experiment(self):
        loader = SimpleNamespace(
            _view_colors_path=lambda: ROOT / "tests" / "missing-view-colors.json"
        )
        presets = MainWindow._load_view_color_presets(loader)
        experiment_colors = [
            preset["color"]
            for preset in presets
            if preset.get("role") in ("exp", "both", "")
        ]

        self.assertEqual(experiment_colors[0], "#000000")

    def test_same_color_mode_repeats_selected_color_for_all_targets(self):
        state = SimpleNamespace(
            _same_curve_color="#d62728",
            _view_curves=[
                {"type": "experiment"},
                {"type": "simulation"},
                {"type": "experiment"},
            ],
        )

        colors = MainWindow._colors_for_curve_targets(state, [0, 2], "same")

        self.assertEqual(colors, ["#d62728", "#d62728"])


if __name__ == "__main__":
    unittest.main()
