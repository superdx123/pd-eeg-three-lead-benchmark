"""Synthetic checks for spectral recipes and representation-fitting boundaries."""

from __future__ import annotations

from unittest import mock
import unittest

import numpy as np
import torch

from pd_eeg.training import representation, spectral


def fixture(windows_per_subject=8):
    groups = np.repeat(np.array([f"s{x}" for x in range(8)]), windows_per_subject)
    labels = np.repeat(np.arange(8) % 2, windows_per_subject)
    X = np.random.default_rng(731).normal(size=(len(groups), 3, 512)).astype(np.float32)
    final = np.arange(len(X))
    left, right = final[:len(final) // 2], final[len(final) // 2:]
    return X, labels, groups, [(1, left, right), (2, right, left)], final


class SpectralTests(unittest.TestCase):
    def test_feature_recipes(self):
        X, *_ = fixture()
        five = spectral.five_band_features(X)
        common = spectral.full_spectrum_features(X)
        self.assertEqual(five.shape, (len(X), 15))
        self.assertEqual(common.shape, five.shape)
        self.assertEqual(five.dtype, np.float32)
        self.assertEqual(common.dtype, np.float32)
        self.assertTrue(np.isfinite(five).all())
        self.assertFalse(np.array_equal(five, common))
        np.testing.assert_allclose((10 ** five.reshape(len(X), 3, 5).astype(float)).sum(-1), 1, atol=1e-6)

    def test_all_spectral_pipelines_refit_and_predict(self):
        X, y, g, splits, final = fixture()
        for name, cohort in (("Spectral-LR", "ds002778"), ("Spectral-LR", "ds003490"),
                             ("Harmonized-LR", "ds002778"), ("RBF-SVM", "ds003490")):
            with self.subTest(model=name, cohort=cohort):
                result = spectral.fit_classifier(name, X, y, g, splits, final, X[:3], 11, cohort)
                self.assertEqual(result["probabilities"].shape, (3,))
                self.assertTrue(np.isfinite(result["probabilities"]).all())
                np.testing.assert_array_equal(result["probabilities"], spectral.predict_classifier(result, X[:3]))
                expected = 25 if name == "RBF-SVM" else 0 if cohort == "ds003490" and name == "Spectral-LR" else 7
                self.assertEqual(len(result["candidates"]), expected)

    def test_source_scalar_is_fitted_only_on_final_train(self):
        X, y, g, _, _ = fixture()
        final = np.arange(len(X) // 2)
        result = spectral.fit_classifier("Spectral-LR", X, y, g, [], final, X[len(X) // 2:],
                                         0, "ds002778", transfer=True)
        self.assertEqual(result["estimator"]["scalar"], spectral.scalar_fit(X[final]))
        self.assertEqual(result["selected"]["C"], 1)
        self.assertEqual(result["candidates"], [])

    def test_overlapping_indices_are_rejected(self):
        X, y, g, _, final = fixture()
        with self.assertRaises(ValueError):
            spectral.fit_classifier("Harmonized-LR", X, y, g, [(1, final, final)], final, X[:1])


class RepresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_psd_pipeline_shortened_synthetic_run(self):
        X, y, g, _, final = fixture(windows_per_subject=30)
        train, validation = final[::2], final[1::2]
        result = representation.fit_classifier("PSD-AE-RBFNN", X, y, g,
            [(1, train, validation)], final, X[:2], seed=11, max_epochs=1, device="cpu")
        self.assertEqual(result["probabilities"].shape, (2,))
        self.assertEqual(result["selected"]["epochs"], 1)
        self.assertEqual(len(result["candidates"]), 3)
        np.testing.assert_array_equal(result["probabilities"], representation.predict_classifier(result, X[:2]))

    def test_tdc_strict_and_reference_fitting_membership(self):
        X, y, g, splits, final = fixture()
        # Encode sample indices into a synthetic signal for boundary observation.
        X[:, 0, 0] = np.arange(len(X))
        for name, expected in (("TDCNet", [splits[0][1], splits[1][1], final]),
                               ("TDCNet-reference", [final])):
            fitted = []

            def train(values, groups, device, epochs, seed):
                fitted.append(values[:, 0, 0].astype(int))
                return torch.nn.Linear(1, 1), [{"loss": 1.0}]

            def encode(network, values, device):
                return np.tile(values[:, :1, 0], (1, 384)).astype(np.float32)

            def infer(classifier, values):
                return (1 / (1 + np.exp(-values[:, 0] / 100))).astype(np.float64)

            with mock.patch.object(representation.tdcnet, "train_tdcnet_representation", side_effect=train), \
                 mock.patch.object(representation.tdcnet, "encode_tdcnet_embeddings", side_effect=encode), \
                 mock.patch.object(representation.tdcnet, "fit_rbf_svm", return_value={}), \
                 mock.patch.object(representation.tdcnet, "positive_class_probabilities", side_effect=infer), \
                 mock.patch.object(representation, "predict_classifier", return_value=np.full(2, .5)):
                result = representation.fit_classifier(name, X, y, g, splits, final, X[:2],
                                                       seed=11, max_epochs=1, device="cpu")
            self.assertEqual(len(fitted), len(expected))
            for actual, allowed in zip(fitted, expected):
                np.testing.assert_array_equal(actual, allowed)
            self.assertEqual(len(result["candidates"]), 10)


if __name__ == "__main__":
    unittest.main()
