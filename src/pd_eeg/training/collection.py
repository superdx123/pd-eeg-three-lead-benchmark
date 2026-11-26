"""Check a run set against its job plan before scoring saved probabilities."""
from pathlib import Path

import numpy as np
import pandas as pd

from pd_eeg.data import load_dataset
from pd_eeg.io import read_json, safe_relative_file, sha256
from .specs import model_name
from .workflow import Job, jobs, split_roles, verify_files


def collect_predictions(runs, config, experiment, *, project_root=".", cohort=None,
                        models=None, protocol=None, seeds=None, allow_partial=False):
    """Return labeled predictions from a complete plan or an explicit subset."""
    cohorts = config["cohorts"]
    selected_models = {model_name(name) for name in (models or ())}
    candidates = {
        job for job in jobs(experiment, cohorts)
        if (cohort is None or job.cohort == cohort)
        and (not selected_models or job.model in selected_models)
        and (protocol is None or job.protocol == protocol)
    }
    selected_seeds = set(seeds or ())
    if selected_seeds.difference(job.seed for job in candidates):
        raise ValueError("Requested seed is not part of the experiment plan")
    planned = {job for job in candidates if not selected_seeds or job.seed in selected_seeds}
    if not planned:
        raise ValueError("No jobs match the requested experiment and filters")

    records = {}
    for path in sorted(Path(runs).rglob("predictions.csv")):
        record = read_json(path.parent / "run.json")
        job = Job(**record["job"])
        if job not in candidates:
            raise ValueError(f"Run does not belong to the requested plan: {job.name}")
        if record["smoke"] is not False:
            raise ValueError(f"Smoke runs are not eligible for benchmark scoring: {job.name}")
        if job not in planned:
            continue
        if job in records:
            raise ValueError(f"Duplicate job: {job.name}")
        if record["threshold"] != 0.5:
            raise ValueError(f"Run does not use the fixed threshold: {job.name}")
        records[job] = (path, record)

    if not records:
        raise ValueError("No saved predictions were found")
    missing = planned.difference(records)
    if missing and not allow_partial:
        first = min(job.name for job in missing)
        raise ValueError(f"Missing {len(missing)} of {len(planned)} jobs; first missing: {first}")

    datasets, digests, test_folds, frames = {}, {}, {}, []

    def dataset_for(name):
        if name not in datasets:
            settings = cohorts[name]
            path = safe_relative_file(project_root, settings["path"])
            datasets[name] = load_dataset(path, expected_sha256=settings.get("sha256"))
            digests[name] = settings.get("sha256") or sha256(path)
        return datasets[name]

    for job, (path, record) in sorted(records.items(), key=lambda item: item[0].name):
        verify_files(path.parent)
        source = dataset_for(job.cohort)
        target_name = job.target or job.cohort
        target = dataset_for(target_name)
        if (record["source_sha256"] != digests[job.cohort]
                or record["target_sha256"] != digests[target_name]):
            raise ValueError(f"Run input SHA-256 differs from the scoring data: {job.name}")

        key = (job.cohort, job.target, job.protocol, job.repeat, job.fold)
        if key not in test_folds:
            test_folds[key] = (np.arange(len(target.y)) if job.target else
                               split_roles(source, cohorts[job.cohort], job, project_root)[2])
        indices = test_folds[key]
        frame = pd.read_csv(path, dtype={"subject_id": str}, keep_default_na=False,
                            float_precision="round_trip")
        expected = {"cohort": target_name, "source": job.cohort, "protocol": job.protocol,
                    "experiment": job.experiment, "model": job.model, "variant": job.variant,
                    "seed": job.seed, "repeat": job.repeat, "fold": job.fold}
        if frame.empty or any(not frame[column].eq(value).all() for column, value in expected.items()):
            raise ValueError(f"Prediction metadata disagrees with run metadata: {job.name}")
        if (not np.array_equal(frame.window_id.to_numpy(), indices)
                or not np.array_equal(record["test_indices"], indices)):
            raise ValueError(f"Test windows disagree with the configured split: {job.name}")
        if not np.array_equal(frame.subject_id.to_numpy(), target.groups[indices]):
            raise ValueError(f"Prediction subject-to-window mapping differs: {job.name}")
        probabilities = frame.prob_pd.to_numpy(dtype=float)
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError(f"Invalid saved probabilities: {job.name}")
        truth = target.y[indices]
        if "true_label" in frame and not np.array_equal(frame.true_label.to_numpy(), truth):
            raise ValueError(f"Saved labels differ from the scoring dataset: {job.name}")
        frame["true_label"] = truth
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), len(records), len(planned)
