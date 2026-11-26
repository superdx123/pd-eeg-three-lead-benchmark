import unittest
from unittest.mock import patch

import pandas as pd

from pd_eeg.analysis.tdc_comparison import STRICT_NAME, compare_tdc


class TdcComparisonTests(unittest.TestCase):
    cohort = "ds002778"
    config = {"cohorts": {cohort: {}}}

    def frame(self, experiment, model, protocol, scores):
        return pd.DataFrame([
            {"cohort": self.cohort, "source": self.cohort, "experiment": experiment, "model": model,
             "protocol": protocol, "variant": "full", "seed": seed, "repeat": int(protocol == "W"),
             "fold": fold if protocol == "W" else label + 1, "window_id": label + 2 * (fold - 1),
             "subject_id": ("a", "b")[label], "true_label": label, "prob_pd": probability}
            for seed in (11, 22, 33)
            for fold, probabilities in enumerate(scores, start=1)
            for label, probability in enumerate(probabilities)
        ])

    def compare(self, left, right, *, protocol="S", reference=None):
        frames = (left, right) if reference is None else (left, right, reference)
        with patch("pd_eeg.analysis.tdc_comparison.collect_predictions",
                   side_effect=[(frame, 6, 6) for frame in frames]) as collect:
            result = compare_tdc("strict", "be", self.config, cohort=self.cohort, protocol=protocol,
                                 reference_runs=None if reference is None else "reference")
        for call in collect.call_args_list:
            self.assertEqual(call.kwargs["seeds"], (11, 22, 33))
            self.assertEqual(call.kwargs["protocol"], protocol)
            self.assertFalse(call.kwargs.get("allow_partial", False))
        return result

    def test_s_uses_strict_minus_comparator_and_preserves_reference_identity(self):
        strict = self.frame("strict-tdc", "TDCNet", "S", [(0.25, 0.75)])
        be_msnet = self.frame("benchmark-s", "BE-MSNet", "S", [(0.25, 0.75)]).iloc[::-1]
        reference = self.frame("tdc-reference", "TDCNet-reference", "S", [(0.75, 0.25)])
        result = self.compare(strict, be_msnet, reference=reference)
        panel = result["panels"][0]
        self.assertEqual(panel["bootstrap_seed"], 20260930)
        self.assertEqual(result["bootstrap"]["replicates"], 2000)
        pairs = {pair["right"]: pair for pair in panel["paired_differences"]}
        self.assertEqual(pairs["BE-MSNet"]["left"], STRICT_NAME)
        self.assertEqual(pairs["BE-MSNet"]["metrics"]["auc"], {"difference": 0, "ci95": [0, 0]})
        self.assertEqual(pairs["TDCNet†"]["metrics"]["auc"], {"difference": 1, "ci95": [1, 1]})

    def test_w_scores_each_fold_before_averaging(self):
        strict = self.frame("strict-tdc", "TDCNet", "W", [(0.49, 0.51), (0.99, 0.01)])
        be_msnet = self.frame("benchmark-w", "BE-MSNet", "W", [(0.2, 0.8), (0.2, 0.8)])
        panel = self.compare(strict, be_msnet, protocol="W")["panels"][0]
        pair = panel["paired_differences"][0]
        for metric in ("auc", "balanced_accuracy"):
            self.assertEqual(panel["models"][STRICT_NAME]["metrics"][metric]["point"], 0.5)
            self.assertEqual(pair["metrics"][metric], {"difference": -0.5, "ci95": [-0.5, -0.5]})

    def test_windows_must_match_across_models(self):
        strict = self.frame("strict-tdc", "TDCNet", "S", [(0.25, 0.75)])
        be_msnet = self.frame("benchmark-s", "BE-MSNet", "S", [(0.25, 0.75)])
        be_msnet.loc[0, "window_id"] = 99
        with self.assertRaisesRegex(ValueError, "do not align"):
            self.compare(strict, be_msnet)

    def test_duplicate_and_missing_seeds_are_rejected(self):
        strict = self.frame("strict-tdc", "TDCNet", "S", [(0.25, 0.75)])
        be_msnet = self.frame("benchmark-s", "BE-MSNet", "S", [(0.25, 0.75)])
        with self.assertRaisesRegex(ValueError, "duplicated test-window"):
            self.compare(strict, pd.concat([be_msnet, be_msnet.iloc[:1]]))
        with self.assertRaisesRegex(ValueError, "exactly seeds"):
            self.compare(strict, be_msnet[be_msnet.seed != 22])
