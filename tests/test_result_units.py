import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from imr_gui.app import MainWindow


class ResultUnitTests(unittest.TestCase):
    @staticmethod
    def _loader() -> MainWindow:
        loader = MainWindow.__new__(MainWindow)
        loader._import_wizard_keywords = MainWindow._normalise_import_wizard_keywords({})
        loader._import_wizard_units = MainWindow._normalise_import_wizard_units({
            "t_exp": "s",
            "R_exp": "pixel",
            "Req": "um",
        })
        loader._import_wizard_um_per_pixel = 99.0
        loader._import_wizard_fps = 1_000_000.0
        loader._import_wizard_remove_below = False
        loader._import_wizard_remove_spikes = False
        loader._import_wizard_spike_threshold = 2.0
        return loader

    def test_export_unit_metadata_round_trips_through_mat(self):
        export = {}
        MainWindow._add_unit_system_to_export(export)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "result.mat"
            savemat(path, export)
            loaded = loadmat(path, squeeze_me=True, struct_as_record=False)

        self.assertEqual(int(loaded["unit_metadata_version"]), 1)
        self.assertEqual(str(loaded["unit_system"]), "SI")
        self.assertEqual(str(loaded["time_unit"]), "s")
        self.assertEqual(str(loaded["radius_unit"]), "m")

    def test_file_units_override_saved_import_defaults(self):
        loader = self._loader()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "micro_units.mat"
            savemat(path, {
                "t_exp": np.array([0.0, 1.0, 2.0]),
                "R_exp": np.array([1.0, 3.0, 2.0]),
                "Req": 2.0,
                "time_unit": "us",
                "radius_unit": "um",
            })
            exp = MainWindow._load_experiment_with_import_defaults(loader, str(path))

        np.testing.assert_allclose(np.diff(exp.t), [1e-6, 1e-6])
        self.assertAlmostEqual(float(np.ptp(exp.t)), 2e-6)
        np.testing.assert_allclose(exp.R, [1e-6, 3e-6, 2e-6])
        self.assertAlmostEqual(exp.R_eq, 2e-6)
        self.assertEqual(exp.import_metadata["unit_source"], "file")
        self.assertEqual(exp.import_metadata["t_unit"], "us")
        self.assertEqual(exp.import_metadata["R_unit"], "um")

    def test_exported_si_result_is_not_reconverted_by_pixel_defaults(self):
        loader = self._loader()
        result = {
            "t_exp": np.array([0.0, 1e-6, 2e-6]),
            "R_exp": np.array([1e-6, 3e-6, 2e-6]),
            "Req": 2e-6,
        }
        MainWindow._add_unit_system_to_export(result)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "exported_result.mat"
            savemat(path, result)
            exp = MainWindow._load_experiment_with_import_defaults(loader, str(path))

        np.testing.assert_allclose(exp.R, [1e-6, 3e-6, 2e-6])
        self.assertAlmostEqual(exp.R_eq, 2e-6)
        self.assertEqual(exp.import_metadata["unit_source"], "file")

    def test_legacy_struct_import_units_do_not_reconvert_result_arrays(self):
        loader = self._loader()
        mat = {
            "struct_import": {
                "t_unit": "us",
                "R_unit": "pixel",
            }
        }

        self.assertEqual(MainWindow._declared_mat_units(loader, mat), {})


if __name__ == "__main__":
    unittest.main()
