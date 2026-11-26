"""Synthetic tests for dataset I/O, EO boundaries and cross-validation isolation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from pd_eeg.data import (
    Dataset, build_nested_loso_manifests, build_protocol_w_manifest,
    build_time_blocks, blocked_roles, representation_folds,
    file_sha256, load_dataset, resolve_output_path, s_inner_splits,
    s_outer_indices, save_dataset, validate_s_manifests,
    validate_w_manifest, w_role_indices,
)
from pd_eeg.data.dataset import validate_dataset
from pd_eeg.data.eo import EO_EVENT, eo_interval, inward_crop_bounds, window_and_standardize


def synthetic_dataset() -> Dataset:
    groups = np.repeat([f"person_{i:02d}" for i in range(12)], 10)
    labels = np.repeat([0] * 6 + [1] * 6, 10).astype(np.int64)
    signal = np.random.default_rng(7).normal(size=(120, 3, 512)).astype(np.float32)
    return Dataset(signal, labels, groups, np.char.add(groups, "_record"), ("C3", "Cz", "C4"), 256.0, 2.0)


class DatasetTests(unittest.TestCase):
    def test_round_trip_and_hash(self):
        original = synthetic_dataset()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "data.npz"
            save_dataset(original, destination)
            loaded = load_dataset(destination, expected_sha256=file_sha256(destination))
            np.testing.assert_array_equal(loaded.X, original.X)
            np.testing.assert_array_equal(loaded.y, original.y)
            self.assertEqual(loaded.channel_names, original.channel_names)
            with self.assertRaises(FileExistsError):
                save_dataset(original, destination)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_dataset(destination, expected_sha256="0" * 64)

    def test_seven_channel_import_is_indexing_only(self):
        original = synthetic_dataset()
        seven = np.random.default_rng(3).normal(2.0, 4.0, (120, 7, 512)).astype(np.float32)
        names = np.asarray(["FC1", "FC2", "C3", "Cz", "C4", "CP1", "CP2"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.npz"
            np.savez_compressed(path, X=seven, y=original.y, groups=original.groups,
                                channel_names=names, sfreq=256.0, window_sec=2.0)
            loaded = load_dataset(path)
            np.testing.assert_array_equal(loaded.X, seven[:, [2, 3, 4], :])
            # Nonzero means are intentional: the loader must not normalize a second time.
            self.assertGreater(abs(float(loaded.X.mean())), 1.0)

    def test_inconsistent_subject_labels_rejected(self):
        original = synthetic_dataset()
        labels = original.y.copy()
        labels[0] = 1
        with self.assertRaisesRegex(ValueError, "one label"):
            validate_dataset(replace(original, y=labels))

    def test_object_arrays_are_not_accepted(self):
        original = synthetic_dataset()
        with self.assertRaisesRegex(ValueError, "string vector"):
            validate_dataset(replace(original, groups=original.groups.astype(object)))

    def test_output_path_cannot_escape_workspace(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            self.assertEqual(resolve_output_path("data/input.npz", base), base.resolve() / "data/input.npz")
            with self.assertRaises(ValueError):
                resolve_output_path("../outside.npz", base)


class SplitTests(unittest.TestCase):
    def test_nested_loso_is_reproducible_and_subject_independent(self):
        data = synthetic_dataset()
        outer, inner = build_nested_loso_manifests(data.y, data.groups)
        outer2, inner2 = build_nested_loso_manifests(data.y, data.groups)
        pd.testing.assert_frame_equal(outer, outer2)
        pd.testing.assert_frame_equal(inner, inner2)
        validate_s_manifests(data.y, data.groups, outer, inner)
        for fold in range(1, 13):
            roles = s_outer_indices(data.groups, outer, fold)
            test = set(data.groups[roles["test"]])
            for _, train, val in s_inner_splits(data.groups, inner, fold):
                self.assertFalse(set(data.groups[train]) & set(data.groups[val]))
                self.assertFalse(test & set(data.groups[np.r_[train, val]]))

    def test_inner_manifest_cannot_include_outer_test_subject(self):
        data = synthetic_dataset()
        outer, inner = build_nested_loso_manifests(data.y, data.groups)
        altered = inner.copy()
        altered.loc[0, "subject_id"] = outer.loc[0, "outer_test_subject_id"]
        with self.assertRaises(ValueError):
            validate_s_manifests(data.y, data.groups, outer, altered)

    def test_window_roles_cover_each_subject_without_window_overlap(self):
        data = synthetic_dataset()
        manifest = build_protocol_w_manifest(data.y, data.groups)
        validate_w_manifest(data.y, data.groups, manifest)
        pd.testing.assert_frame_equal(manifest, build_protocol_w_manifest(data.y, data.groups))
        for repeat in range(1, 6):
            for fold in range(1, 6):
                roles = w_role_indices(manifest, repeat, fold)
                self.assertFalse(set(roles["train"]) & set(roles["test"]))
                self.assertFalse(set(roles["validation"]) & set(roles["test"]))
                for indices in roles.values():
                    self.assertEqual(set(data.groups[indices]), set(data.groups))
                self.assertEqual(len(roles["test"]), 24)

    def test_changed_window_labels_are_rejected(self):
        data = synthetic_dataset()
        manifest = build_protocol_w_manifest(data.y, data.groups)
        manifest.loc[0, "true_label"] = 1
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            validate_w_manifest(data.y, data.groups, manifest)


class EOTests(unittest.TestCase):
    def test_contiguous_markers(self):
        events = pd.DataFrame({"onset": np.arange(60, dtype=float), "trial_type": EO_EVENT})
        self.assertEqual(eo_interval(events), (0.0, 59.0))
        events.loc[30, "onset"] = 31.1
        with self.assertRaisesRegex(ValueError, "continuous"):
            eo_interval(events)

    def test_auditory_events_inside_eo_are_rejected(self):
        events = pd.DataFrame({"onset": np.arange(60, dtype=float), "trial_type": EO_EVENT})
        events = pd.concat([events, pd.DataFrame({"onset": [20.4], "trial_type": ["Standard Tone"]})])
        with self.assertRaisesRegex(ValueError, "auditory"):
            eo_interval(events)

    def test_crop_rounds_inward(self):
        start, stop = inward_crop_bounds(1.0001, 59.9999)
        self.assertEqual(start, 1.002)
        self.assertEqual(stop, 59.998)
        self.assertGreaterEqual(start, 1.0001)
        self.assertLessEqual(stop, 59.9999)

    def test_qc_zscore_and_two_second_edge_guard(self):
        t = np.arange(64 * 256) / 256
        values = np.tile(10.0 * np.sin(2 * np.pi * 10.0 * t), (3, 1))
        windows, metadata = window_and_standardize(values, 4.0)
        self.assertEqual(windows.shape, (30, 3, 512))
        self.assertEqual(metadata["window_start_seconds"][0], 6.0)
        self.assertEqual(metadata["window_start_seconds"][-1] + 2.0, 66.0)
        np.testing.assert_allclose(windows.mean(axis=2), 0.0, atol=1e-6)
        np.testing.assert_allclose(windows.std(axis=2), 1.0, atol=1e-6)
        with self.assertRaisesRegex(ValueError, "QC"):
            window_and_standardize(np.zeros_like(values), 0.0)


class AdditionalSplitTests(unittest.TestCase):
    def time_frame(self):
        starts = np.tile(np.arange(30, dtype=float) * 2.0, 2)
        # A real-time gap after QC must not be collapsed to consecutive row numbers.
        starts[8:30] += 20.0
        return pd.DataFrame({"window_index": np.arange(60),
                             "subject_id": np.repeat(["person_a", "person_b"], 30),
                             "recording_id": np.repeat(["record_a", "record_b"], 30),
                             "start_sec": starts, "end_sec": starts + 2.0})

    def test_time_roles_use_actual_interval_distances(self):
        frame = build_time_blocks(self.time_frame())
        for fold in range(1, 6):
            roles, audit = blocked_roles(frame, fold, return_audit=True)
            self.assertEqual(set(roles), {"train", "validation", "refit", "test"})
            self.assertTrue(all(gap >= 2.0 - 1e-9 for gap in audit["minimum_actual_interval_gaps_seconds"].values()))
            self.assertFalse(set(roles["refit"]) & set(roles["test"]))
        # Candidate at 34 s is retained despite adjacent retained-window ranks.
        # Purging is based on actual intervals, not a fixed number of rows.
        roles = blocked_roles(frame, 1)
        self.assertIn(8, roles["validation"])

    def test_time_metadata_alignment_and_coordinates_are_required(self):
        frame = self.time_frame()
        with self.assertRaisesRegex(ValueError, "align"):
            build_time_blocks(frame, groups=frame.subject_id.to_numpy()[::-1])
        with self.assertRaisesRegex(ValueError, "missing"):
            build_time_blocks(frame.drop(columns="start_sec"))
        wrong = frame.copy()
        wrong.loc[0, "end_sec"] = 3.0
        with self.assertRaisesRegex(ValueError, "2 s"):
            build_time_blocks(wrong)

    def test_representation_folds_are_class_balanced_and_mapping_invariant(self):
        groups = np.repeat([f"participant_{i:02d}" for i in range(50)], 2)
        labels = np.repeat([0] * 25 + [1] * 25, 2)
        reference = representation_folds(labels, groups)
        for fold in range(5):
            for label in (0, 1):
                people = set(groups[labels == label])
                self.assertEqual(sum(reference[subject] == fold for subject in people), 5)
        renamed = np.asarray(["renamed_" + value for value in groups])
        identity = {"renamed_" + subject: subject for subject in reference}
        mapped = representation_folds(labels, renamed, participant_ids_map=identity)
        self.assertEqual({identity[subject]: fold for subject, fold in mapped.items()}, reference)
        with self.assertRaises(ValueError):
            representation_folds(labels, groups, participant_ids_map={})


if __name__ == "__main__":
    unittest.main()
