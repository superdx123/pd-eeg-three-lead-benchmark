import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pd_eeg.analysis.metrics import analyze_predictions
from pd_eeg.data import Dataset, save_dataset
from pd_eeg.io import sha256
from pd_eeg.training.workflow import Job, jobs, run_job, validate_job, verify_files


class WorkflowTests(unittest.TestCase):
    def test_study_job_counts(self):
        config = {"ds002778": {"subjects": 31}, "ds003490": {"subjects": 50}}
        for experiment, count in {
            "benchmark-s": 4131, "benchmark-w": 2550, "strict-tdc": 393,
            "transfer": 14, "time-blocked": 40, "representation-control": 30,
            "harmonized-lr": 83, "ablation": 972, "tdc-reference": 655,
        }.items():
            with self.subTest(experiment=experiment):
                plan = list(jobs(experiment, config))
                self.assertEqual(len(plan), count)
                self.assertEqual(len({job.name for job in plan}), count)

    def test_synthetic_train_save_replay_and_transfer_scoring_boundary(self):
        torch.set_num_threads(2)
        rng = np.random.default_rng(19)
        X = rng.normal(size=(60, 3, 512)).astype(np.float32)
        ids = np.repeat([f"person{i:02d}" for i in range(12)], 5)
        y = np.repeat([0, 1] * 6, 5)
        data = Dataset(X, y.astype(np.int64), ids, ids, ("C3", "Cz", "C4"), 256., 2.)
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            save_dataset(data, base / "source.npz")
            save_dataset(data, base / "target.npz")
            settings = {"path": "source.npz", "sha256": sha256(base / "source.npz"), "inner_seed_base": 730000}
            config = {"cohorts": {"ds002778": settings,
                                  "ds003490": {**settings, "path": "target.npz", "sha256": sha256(base / "target.npz")}}}
            options = {"project_root": base, "output": base / "runs", "device": "cpu", "smoke": True}
            job = Job("benchmark-w", "ds002778", "W", "BE-MSNet", 11, 1, 1)
            run = run_job(job, config, **options)
            verify_files(run)
            self.assertLessEqual(validate_job(run, config, project_root=base)["maximum_absolute_error"], 1e-7)
            with self.assertRaises(FileExistsError):
                run_job(job, config, **options)
            transfer = Job("harmonized-lr", "ds002778", "transfer", "Harmonized-LR", 0, target="ds003490")
            run = run_job(transfer, config, **options)
            self.assertNotIn("true_label", (run / "predictions.csv").read_text().splitlines()[0])
            with self.assertRaises(ValueError):
                validate_job(run, config, project_root=base)
            self.assertLessEqual(validate_job(run, config, project_root=base, trust_sklearn=True)["maximum_absolute_error"], 1e-7)

    def test_panels_keep_analysis_variants_and_model_names_separate(self):
        rows = []
        for experiment, model, variant in (
            ("benchmark-s", "BE-MSNet", "full"),
            ("ablation", "BE-MSNet", "full"), ("ablation", "BE-MSNet", "raw_only"),
            ("strict-tdc", "TDCNet", "full"), ("tdc-reference", "TDCNet-reference", "full"),
        ):
            for i, truth in enumerate((0, 1)):
                rows.append({"cohort": "ds002778", "protocol": "S", "model": model, "seed": 11,
                             "subject_id": f"id{i}", "true_label": truth, "prob_pd": .2 + .6 * truth,
                             "experiment": experiment, "variant": variant})
        panels = {item["experiment"]: item for item in analyze_predictions(rows, replicates=3)["panels"]}
        self.assertEqual(set(panels["ablation"]["models"]), {"BE-MSNet", "BE-MSNet (raw_only)"})
        self.assertIn("TDCNet (strict inner fitting)", panels["strict-tdc"]["models"])
        self.assertIn("TDCNet†", panels["tdc-reference"]["models"])
