import shutil
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from pd_eeg.data import Dataset, save_dataset
from pd_eeg.io import sha256, write_json
from pd_eeg.training.collection import collect_predictions
from pd_eeg.training.workflow import Job, split_roles


class CollectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        subjects = np.repeat([f"subject-{i:02d}" for i in range(12)], 5)
        self.dataset = Dataset(np.zeros((60, 3, 512), dtype=np.float32),
                               np.repeat([0, 1] * 6, 5).astype(np.int64), subjects, subjects,
                               ("C3", "Cz", "C4"), 256., 2.)
        self.cohort = "ds002778"
        self.settings = {"path": "input.npz", "subjects": 12, "inner_seed_base": 730000}
        self.path = save_dataset(self.dataset, self.root / self.settings["path"])
        self.settings["sha256"] = sha256(self.path)
        self.config = {"cohorts": {self.cohort: self.settings}}
        self.runs = self.root / "runs"

    def write_run(self, job, *, smoke=False, offset=0, bad_model=None, directory=None):
        out = self.runs / (directory or job.name)
        out.mkdir(parents=True, exist_ok=True)
        indices = split_roles(self.dataset, self.settings, job, self.root)[2] + offset
        digest = self.settings["sha256"]
        frame = pd.DataFrame({
            "cohort": self.cohort, "source": self.cohort, "protocol": job.protocol,
            "experiment": job.experiment, "model": bad_model or job.model,
            "variant": job.variant, "seed": job.seed, "repeat": job.repeat, "fold": job.fold,
            "window_id": indices, "subject_id": self.dataset.groups[indices],
            "prob_pd": 0.25 + 0.5 * self.dataset.y[indices],
        })
        frame.to_csv(out / "predictions.csv", index=False)
        (out / "model.joblib").write_bytes(b"unused by scoring")
        write_json(out / "selection.json", {})
        write_json(out / "run.json", {"job": asdict(job), "smoke": smoke, "threshold": 0.5,
                   "source_sha256": digest, "target_sha256": digest, "test_indices": indices.tolist()})
        files = ("predictions.csv", "run.json", "selection.json", "model.joblib")
        (out / "checksums.sha256").write_text(
            "".join(f"{sha256(out / name)}  {name}\n" for name in files), encoding="ascii",
        )
        return out

    def score(self, *, allow_partial=False):
        return collect_predictions(self.runs, self.config, "benchmark-s", project_root=self.root,
                                   models=["Spectral-LR"], allow_partial=allow_partial)

    def job(self, fold=1):
        return Job("benchmark-s", self.cohort, "S", "Spectral-LR", 0, fold)

    def test_complete_plan_attaches_labels(self):
        for fold in range(1, 13):
            self.write_run(self.job(fold))
        frame, found, planned = self.score()
        self.assertEqual((found, planned), (12, 12))
        np.testing.assert_array_equal(frame.true_label, self.dataset.y[frame.window_id])

    def test_incomplete_requires_opt_in(self):
        self.write_run(self.job())
        with self.assertRaisesRegex(ValueError, "Missing 11 of 12 jobs"):
            self.score()
        self.assertEqual(self.score(allow_partial=True)[1:], (1, 12))

    def test_duplicate_job_is_not_a_replacement(self):
        run = self.write_run(self.job())
        shutil.copytree(run, self.runs / "duplicate")
        with self.assertRaisesRegex(ValueError, "Duplicate job"):
            self.score(allow_partial=True)

    def test_smoke_is_ineligible_even_if_partial(self):
        self.write_run(self.job(), smoke=True)
        with self.assertRaisesRegex(ValueError, "Smoke runs"):
            self.score(allow_partial=True)

    def test_resigned_predictions_with_incorrect_test_windows_are_rejected(self):
        self.write_run(self.job(), offset=1)
        with self.assertRaisesRegex(ValueError, "configured split"):
            self.score(allow_partial=True)

    def test_metadata_must_match_the_run(self):
        self.write_run(self.job(), bad_model="BE-MSNet")
        with self.assertRaisesRegex(ValueError, "metadata disagrees"):
            self.score(allow_partial=True)
