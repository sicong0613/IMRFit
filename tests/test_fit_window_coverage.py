import pickle
import unittest
from types import SimpleNamespace

import numpy as np

from imr_gui.opt.nhkv_fit import (
    FitConfig,
    OptConfig,
    _DEObjFn,
    _INVALID_FIT_ERROR,
    _eval_and_sim,
    fit_nhkv_to_experiment,
)


PEAK_TIME = 0.6e-6


def delayed_peak_sim(_params, tspan):
    t = np.linspace(0.0, tspan, 101)
    if tspan > PEAK_TIME:
        t = np.unique(np.append(t, PEAK_TIME))
    shifted = t - PEAK_TIME
    return SimpleNamespace(t_sim=shifted, R_sim=5e-6 - np.abs(shifted))


def fixed_endpoint_sim(_params, _tspan):
    t = np.linspace(-0.6e-6, 0.4e-6, 101)
    return SimpleNamespace(t_sim=t, R_sim=5e-6 + t)


class FitWindowCoverageTests(unittest.TestCase):
    def setUp(self):
        self.t_exp = np.array([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8]) * 1e-6
        self.R_exp = 5e-6 - np.abs(self.t_exp)

    def config(self, make_sim, R_exp=None):
        return FitConfig(
            t_exp=self.t_exp,
            R_exp=self.R_exp if R_exp is None else R_exp,
            make_sim=make_sim,
            param_names=["G"],
        )

    def test_extends_past_shifted_peak_and_scores_all_points(self):
        calls = []

        def simulate(params, tspan):
            calls.append(tspan)
            return delayed_peak_sim(params, tspan)

        R_exp = self.R_exp.copy()
        R_exp[-1] += 1e-6
        err, out = _eval_and_sim({"G": 1.0}, self.config(simulate, R_exp))
        self.assertGreater(len(calls), 1)
        self.assertGreaterEqual(out.t_sim[-1], self.t_exp[-1])
        self.assertAlmostEqual(err, 1.0 / self.t_exp.size, places=10)

    def test_left_edge_trims_uncovered_point_without_extending_span(self):
        calls = []

        def simulate(params, tspan):
            calls.append(tspan)
            out = delayed_peak_sim(params, tspan)
            out.t_sim += 0.5e-6
            return out

        err, out = _eval_and_sim({"G": 1.0}, self.config(simulate))
        self.assertIsNotNone(out)
        self.assertLess(err, _INVALID_FIT_ERROR)
        self.assertEqual(len(calls), 1)

    def test_three_left_points_are_allowed_but_not_scored(self):
        t_exp = np.arange(7, dtype=float) * 1e-6
        R_exp = np.array([100, 100, 100, 5, 5, 5, 6], dtype=float) * 1e-6

        def simulate(_params, _tspan):
            return SimpleNamespace(
                t_sim=t_exp[3:].copy(), R_sim=np.full(4, 5e-6)
            )

        cfg = FitConfig(t_exp=t_exp, R_exp=R_exp, make_sim=simulate, param_names=["G"])
        err, out = _eval_and_sim({"G": 1.0}, cfg)
        self.assertIsNotNone(out)
        self.assertAlmostEqual(err, 1.0 / 4.0)

    def test_four_left_points_are_rejected(self):
        t_exp = np.arange(8, dtype=float) * 1e-6

        def simulate(_params, _tspan):
            return SimpleNamespace(
                t_sim=t_exp[4:].copy(), R_sim=np.full(4, 5e-6)
            )

        cfg = FitConfig(t_exp=t_exp, R_exp=np.full(8, 5e-6), make_sim=simulate, param_names=["G"])
        err, out = _eval_and_sim({"G": 1.0}, cfg)
        self.assertEqual(err, _INVALID_FIT_ERROR)
        self.assertIsNone(out)

    def test_trim_must_leave_at_least_three_points(self):
        t_exp = np.arange(5, dtype=float) * 1e-6

        def simulate(_params, _tspan):
            return SimpleNamespace(
                t_sim=np.array([3.0, 3.5, 4.0]) * 1e-6,
                R_sim=np.full(3, 5e-6),
            )

        cfg = FitConfig(t_exp=t_exp, R_exp=np.full(5, 5e-6), make_sim=simulate, param_names=["G"])
        err, out = _eval_and_sim({"G": 1.0}, cfg)
        self.assertEqual(err, _INVALID_FIT_ERROR)
        self.assertIsNone(out)

    def test_unreachable_right_edge_is_penalized_with_bounded_retries(self):
        calls = []

        def simulate(params, tspan):
            calls.append(tspan)
            return fixed_endpoint_sim(params, tspan)

        err, out = _eval_and_sim({"G": 1.0}, self.config(simulate))
        self.assertEqual(err, _INVALID_FIT_ERROR)
        self.assertIsNone(out)
        self.assertLessEqual(len(calls), 5)
        self.assertLessEqual(max(calls), 8 * calls[0])

    def test_extends_when_peak_is_not_reached_initially(self):
        t_exp = np.array([-1.5, -1.3, -1.1, -0.9, -0.7]) * 1e-6
        R_exp = 5e-6 - np.abs(t_exp)
        calls = []

        def simulate(_params, tspan):
            calls.append(tspan)
            t = np.linspace(0.0, tspan, 101)
            if tspan > 2e-6:
                t = np.unique(np.append(t, 2e-6))
            return SimpleNamespace(
                t_sim=t - min(tspan, 2e-6),
                R_sim=5e-6 - np.abs(t - 2e-6),
            )

        cfg = FitConfig(t_exp=t_exp, R_exp=R_exp, make_sim=simulate, param_names=["G"])
        err, out = _eval_and_sim({"G": 1.0}, cfg)
        self.assertGreater(len(calls), 1)
        self.assertLess(err, _INVALID_FIT_ERROR)
        self.assertLessEqual(np.searchsorted(t_exp, out.t_sim[0]), 3)
        self.assertGreaterEqual(out.t_sim[-1], t_exp[-1])

    def test_all_invalid_trials_fail_instead_of_reporting_fit(self):
        with self.assertRaisesRegex(RuntimeError, "right edge of the fitting window"):
            fit_nhkv_to_experiment(
                self.config(fixed_endpoint_sim),
                bounds_si={"G": (0.5, 1.5)},
                initial_values={"G": 1.0},
                opt_config=OptConfig(method="Nelder-Mead", max_fev=5),
            )

    def test_parallel_objective_uses_same_window_rule(self):
        objective = _DEObjFn(
            active=[("G", "lin")], fixed={}, mp_make_sim=delayed_peak_sim,
            t_exp=self.t_exp, R_exp=self.R_exp, tspan_factor=1.2,
        )
        restored = pickle.loads(pickle.dumps(objective))
        serial_err, _ = _eval_and_sim({"G": 1.0}, self.config(delayed_peak_sim))
        self.assertAlmostEqual(restored(np.array([1.0])), serial_err)


if __name__ == "__main__":
    unittest.main()
