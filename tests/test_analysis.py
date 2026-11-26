"""Synthetic checks for the benchmark's aggregation and statistical estimands."""

from __future__ import annotations

import unittest

import numpy as np

from pd_eeg.analysis.metrics import (
    aggregate_subjects, analyze_predictions, bootstrap_fold_metrics,
    stratified_counts, subject_metrics,
)


def prediction(subject, label, probability, *, model="BE-MSNet", seed=11, fold=0, protocol="S", window=""):
    return {"cohort": "synthetic", "protocol": protocol, "model": model,
            "seed": seed, "subject_id": subject, "true_label": label,
            "prob_pd": probability, "repeat": 0, "fold": fold, "window_id": window}


class AnalysisTests(unittest.TestCase):
    def test_metric_ties_and_fixed_threshold(self):
        scores = subject_metrics([0, 1], [0.5, 0.5])
        self.assertEqual(scores["auc"], 0.5)
        self.assertEqual(scores["balanced_accuracy"], 0.5)
        self.assertEqual(scores["sensitivity"], 1)
        self.assertEqual(scores["specificity"], 0)

    def test_subject_aggregation_uses_equal_window_mean_and_sorted_ids(self):
        table = aggregate_subjects(["p", "h", "p"], [1, 0, 1], [.6, .2, 1.])
        self.assertEqual(table["subject_id"].tolist(), ["h", "p"])
        np.testing.assert_allclose(table["prob_pd"], [.2, .8])
        self.assertEqual(table["n_windows"].tolist(), [1, 2])

    def test_seed_mean_precedes_auc(self):
        rows = [prediction("h", 0, .1, seed=11), prediction("p", 1, .2, seed=11),
                prediction("p", 1, .6, seed=22), prediction("h", 0, .9, seed=22)]
        panel = analyze_predictions(rows, replicates=20)["panels"][0]
        self.assertEqual(panel["models"]["BE-MSNet"]["metrics"]["auc"]["point"], 0)
        self.assertEqual(panel["models"]["BE-MSNet"]["seeds"], [11, 22])

    def test_oof_subjects_cannot_appear_in_multiple_folds(self):
        rows = [prediction("h", 0, .2), prediction("p", 1, .8), prediction("h", 0, .3, fold=1)]
        with self.assertRaises(ValueError):
            analyze_predictions(rows, replicates=10)

    def test_missing_seed_score_is_rejected(self):
        rows = [prediction("h", 0, .2), prediction("p", 1, .8), prediction("p", 1, .7, seed=22)]
        with self.assertRaises(ValueError):
            analyze_predictions(rows, replicates=10)

    def test_pairing_aligns_subjects_not_row_positions(self):
        rows = [prediction("h", 0, .2), prediction("p", 1, .8),
                prediction("p", 1, .8, model="EEGNet"), prediction("h", 0, .2, model="EEGNet")]
        difference = analyze_predictions(rows, replicates=20)["panels"][0]["paired_differences"][0]
        self.assertEqual(difference["metrics"]["auc"]["difference"], 0)
        self.assertEqual(difference["metrics"]["auc"]["ci95"], [0, 0])

    def test_window_means_are_formed_within_seed(self):
        rows = [prediction("h", 0, .2, window="0"), prediction("h", 0, .4, window="1"),
                prediction("p", 1, .8, window="2"), prediction("p", 1, .6, window="3")]
        auc = analyze_predictions(rows, replicates=10)["panels"][0]["models"]["BE-MSNet"]["metrics"]["auc"]["point"]
        self.assertEqual(auc, 1)

    def test_fold_mean_is_not_probability_pooling(self):
        labels = np.array([0, 1])
        scores = np.array([[.1, .2], [.9, .4]])
        wp, wn = stratified_counts(labels, 15)
        result = bootstrap_fold_metrics(labels, scores, wp, wn)
        np.testing.assert_allclose(result["auc"], .5)
        self.assertEqual(subject_metrics(labels, scores.mean(axis=0))["auc"], 0)

    def test_bootstrap_counts_are_shared_and_reproducible(self):
        labels = [0, 0, 1, 1]
        first = stratified_counts(labels, 20, 3)
        second = stratified_counts(labels, 20, 3)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        np.testing.assert_array_equal(first[0].sum(axis=1), 2)
        np.testing.assert_array_equal(first[1].sum(axis=1), 2)

    def test_invalid_label_or_probability_is_rejected(self):
        for row in (prediction("h", 2, .2), prediction("h", 0, float("nan")), prediction("h", 0, 1.1)):
            with self.assertRaises(ValueError):
                analyze_predictions([row], replicates=10)

    def test_subject_labels_are_consistent(self):
        with self.assertRaises(ValueError):
            aggregate_subjects(["same", "same"], [0, 1], [.2, .8])


if __name__ == "__main__":
    unittest.main()
