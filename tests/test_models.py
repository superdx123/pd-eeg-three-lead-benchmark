"""Small, synthetic CPU checks for model interfaces and representations."""

from __future__ import annotations

import importlib.metadata
import unittest

import numpy as np
import torch

from pd_eeg.models import BEMSNet, build_model, expand_bands, parameter_count, prepare_input
from pd_eeg.models.psd_ae import max_normalized_welch_psd_features


class BEMSNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_manuscript_parameter_counts_and_shapes(self):
        expected = {"full": 67190, "raw_only": 35186, "single_scale": 27198,
                    "mean_only": 57974, "r18": 67190}
        raw = np.random.default_rng(17).normal(size=(2, 3, 512)).astype(np.float32)
        for variant, count in expected.items():
            with self.subTest(variant=variant):
                model = build_model("BE-MSNet", variant=variant).eval()
                self.assertEqual(parameter_count(model), count)
                values = torch.from_numpy(prepare_input(raw, variant))
                with torch.no_grad():
                    logits = model(values)
                self.assertEqual(tuple(logits.shape), (2, 2))
                self.assertTrue(torch.isfinite(logits).all())

    def test_saved_state_names(self):
        keys = set(BEMSNet().state_dict())
        for name in (
            "encoder.branches.0.features.0.weight", "encoder.projection.0.weight",
            "encoder.projection.1.norm.weight", "fusion_proj.0.weight", "head.weight", "head.bias",
        ):
            self.assertIn(name, keys)

    def test_full_and_r18_have_the_same_state(self):
        torch.manual_seed(23)
        full = BEMSNet("full")
        torch.manual_seed(23)
        r18 = BEMSNet("r18")
        self.assertEqual(list(full.state_dict()), list(r18.state_dict()))
        for name, value in full.state_dict().items():
            torch.testing.assert_close(value, r18.state_dict()[name], rtol=0, atol=0)

    def test_gradient_is_finite(self):
        model = BEMSNet()
        values = torch.randn(2, 18, 512, requires_grad=True)
        model(values).square().mean().backward()
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_boundary_bins_are_shared(self):
        time = np.arange(512, dtype=np.float64) / 256.0
        raw = np.broadcast_to(np.sin(2 * np.pi * 4 * time), (1, 3, 512)).astype(np.float32)
        expanded = expand_bands(raw)
        self.assertEqual(expanded.shape, (1, 18, 512))
        np.testing.assert_array_equal(expanded[:, :3], raw)
        np.testing.assert_allclose(expanded[:, 3:6], raw, rtol=0, atol=1e-6)
        np.testing.assert_allclose(expanded[:, 6:9], raw, rtol=0, atol=1e-6)
        np.testing.assert_allclose(expanded[:, 9:], 0, rtol=0, atol=1e-6)

    def test_r18_preserves_triplet_order(self):
        raw = np.random.default_rng(23).normal(size=(2, 3, 512)).astype(np.float32)
        np.testing.assert_array_equal(prepare_input(raw, "r18"), np.tile(raw, (1, 6, 1)))

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            prepare_input(np.zeros((2, 7, 512), dtype=np.float32))
        with self.assertRaises(ValueError):
            BEMSNet("unknown")
        with self.assertRaises(ValueError):
            BEMSNet()(torch.zeros(2, 3, 512))


class BaselineModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_waveform_baselines(self):
        raw = torch.randn(2, 3, 512)
        for name in ("4L-CNN", "EEGNet", "Lee-CRNN"):
            with self.subTest(model=name):
                model = build_model(name).eval()
                with torch.no_grad():
                    logits = model(raw)
                self.assertEqual(tuple(logits.shape), (2, 2))
                self.assertTrue(torch.isfinite(logits).all())

    def test_psd_autoencoder_and_classifier(self):
        raw = np.random.default_rng(11).normal(size=(2, 3, 512)).astype(np.float32)
        features, names, metadata = max_normalized_welch_psd_features(raw, 256.0, ("C3", "Cz", "C4"))
        self.assertEqual(features.shape, (2, 120))
        self.assertEqual(len(names), 120)
        self.assertEqual(metadata["n_features"], 120)
        autoencoder = build_model("PSD-AE-RBFNN", stage="autoencoder").eval()
        with torch.no_grad():
            self.assertEqual(tuple(autoencoder(torch.from_numpy(features)).shape), (2, 120))
        centers = np.random.default_rng(11).normal(size=(100, 50)).astype(np.float32)
        combined = build_model("PSD-AE-RBFNN", initial_centers=centers, initial_sigma=1.0).eval()
        with torch.no_grad():
            self.assertEqual(tuple(combined(torch.from_numpy(features)).shape), (2, 2))
        with self.assertRaises(ValueError):
            build_model("PSD-AE-RBFNN")

    def test_tdc_embedding_shape(self):
        model = build_model("TDCNet").eval()
        with torch.no_grad():
            embedding = model.clean_embedding(torch.randn(2, 3, 512))
        self.assertEqual(tuple(embedding.shape), (2, 384))
        self.assertTrue(torch.isfinite(embedding).all())

    def test_braindecode_models_when_installed(self):
        try:
            installed = importlib.metadata.version("braindecode")
        except importlib.metadata.PackageNotFoundError:
            self.skipTest("Install the pinned Braindecode dependency to test these four models.")
        if installed != "1.2.0":
            self.skipTest("These constructors require braindecode 1.2.0.")
        for name in ("EEGConformer", "ATCNet", "EEGNeX", "FBCNet"):
            with self.subTest(model=name):
                model = build_model(name).eval()
                with torch.no_grad():
                    logits = model(torch.randn(2, 3, 512))
                self.assertEqual(tuple(logits.shape), (2, 2))


if __name__ == "__main__":
    unittest.main()
