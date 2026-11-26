"""Paper-wide job planning, training outputs and independent inference checks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd

from pd_eeg.data import (load_dataset, build_nested_loso_manifests, build_protocol_w_manifest,
                         s_outer_indices, s_inner_splits, w_role_indices,
                         validate_s_manifests, validate_w_manifest)
from pd_eeg.io import read_json, safe_relative_file, sha256, write_json
from .specs import MAIN_MODELS, MATCHED_SEEDS, SEEDS, model_name

EXPERIMENTS = ("benchmark-s", "benchmark-w", "transfer", "ablation", "strict-tdc",
               "tdc-reference", "harmonized-lr", "time-blocked", "representation-control")


@dataclass(frozen=True)
class Job:
    experiment: str
    cohort: str
    protocol: str
    model: str
    seed: int
    fold: int = 1
    repeat: int = 0
    variant: str = "full"
    target: str | None = None

    @property
    def name(self):
        model = re.sub(r"[^A-Za-z0-9_-]", "_", self.model)
        cohort = f"{self.cohort}_to_{self.target}" if self.target else self.cohort
        return f"{self.experiment}/{cohort}/{self.protocol}/{model}/{self.variant}/seed{self.seed}/r{self.repeat}_f{self.fold}"


def jobs(experiment: str, cohorts: dict, models=None):
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}")
    selected = tuple(map(model_name, models)) if models else MAIN_MODELS
    for cohort, settings in cohorts.items():
        if experiment in ("benchmark-s", "benchmark-w"):
            for name in selected:
                seeds = (0,) if name == "Spectral-LR" else SEEDS
                for seed in seeds:
                    repeats = (0,) if experiment.endswith("-s") else range(1, 6)
                    nfolds = int(settings["subjects"]) if experiment.endswith("-s") else 5
                    for repeat in repeats:
                        for fold in range(1, nfolds + 1):
                            yield Job(experiment, cohort, "S" if experiment.endswith("-s") else "W", name, seed, fold, repeat)
        elif experiment in ("transfer", "harmonized-lr"):
            if experiment == "harmonized-lr":
                for fold in range(1, int(settings["subjects"]) + 1):
                    yield Job(experiment, cohort, "S", "Harmonized-LR", 0, fold)
            for target in cohorts:
                if target == cohort:
                    continue
                names = ("Harmonized-LR",) if experiment == "harmonized-lr" else ("BE-MSNet", "ATCNet", "Spectral-LR")
                for name in names:
                    for seed in ((0,) if "LR" in name else MATCHED_SEEDS):
                        yield Job(experiment, cohort, "transfer", name, seed, target=target)
        elif experiment == "ablation":
            for variant in ("full", "raw_only", "single_scale", "mean_only"):
                for seed in MATCHED_SEEDS:
                    for fold in range(1, int(settings["subjects"]) + 1):
                        yield Job(experiment, cohort, "S", "BE-MSNet", seed, fold, variant=variant)
        elif experiment in ("strict-tdc", "tdc-reference"):
            name = "TDCNet" if experiment == "strict-tdc" else "TDCNet-reference"
            for protocol in ("S", "W"):
                for seed in (MATCHED_SEEDS if name == "TDCNet" else SEEDS):
                    for repeat in ((0,) if protocol == "S" else range(1, 6)):
                        for fold in range(1, (int(settings["subjects"]) if protocol == "S" else 5) + 1):
                            yield Job(experiment, cohort, protocol, name, seed, fold, repeat)
        elif experiment == "time-blocked":
            for name in ("BE-MSNet", "Harmonized-LR"):
                for seed in ((0,) if name.endswith("LR") else MATCHED_SEEDS):
                    for fold in range(1, 6):
                        yield Job(experiment, cohort, "time-blocked", name, seed, fold, 1)
        elif experiment == "representation-control" and cohort == "ds003490":
            for variant in ("full", "r18"):
                for seed in MATCHED_SEEDS:
                    for fold in range(1, 6):
                        yield Job(experiment, cohort, "representation-control", "BE-MSNet", seed, fold, variant=variant)


def _manifests(dataset, settings, base):
    directory = safe_relative_file(base, settings.get("splits", ""))
    expected = settings.get("split_sha256", {})
    for name, digest in expected.items():
        if sha256(safe_relative_file(directory, name)) != digest:
            raise ValueError(f"Split SHA-256 mismatch: {name}")
    outer_path, inner_path = directory / "s_outer.csv", directory / "s_inner.csv"
    if outer_path.is_file() != inner_path.is_file():
        raise ValueError("Provide both S manifests or neither")
    if outer_path.is_file():
        outer = pd.read_csv(outer_path, dtype={"outer_test_subject_id": str})
        inner = pd.read_csv(inner_path, dtype={"subject_id": str, "outer_test_subject_id": str})
    else:
        outer, inner = build_nested_loso_manifests(dataset.y, dataset.groups,
                       inner_seed_base=int(settings["inner_seed_base"]))
    w_path = directory / "w_windows.csv"
    w = pd.read_csv(w_path, dtype={"subject_id": str}) if w_path.is_file() else build_protocol_w_manifest(dataset.y, dataset.groups)
    validate_s_manifests(dataset.y, dataset.groups, outer, inner)
    validate_w_manifest(dataset.y, dataset.groups, w)
    return outer, inner, w


def split_roles(dataset, settings, job: Job, base):
    if job.protocol in ("S", "transfer"):
        outer, inner, _ = _manifests(dataset, settings, base)
        fold = job.fold if job.protocol == "S" else int(outer.outer_fold.min())
        roles = s_outer_indices(dataset.groups, outer, fold)
        splits = s_inner_splits(dataset.groups, inner, fold)
        final = roles["train"] if job.protocol == "S" else np.arange(len(dataset.y), dtype=np.int64)
        return splits, final, roles["test"]
    if job.protocol == "W":
        _, _, w = _manifests(dataset, settings, base)
        role = w_role_indices(w, job.repeat, job.fold)
        tr, va, te = (np.asarray(role[k], dtype=np.int64) for k in ("train", "validation", "test"))
        return [(0, tr, va)], np.sort(np.concatenate((tr, va))), te
    if job.protocol == "time-blocked":
        from pd_eeg.data.splits import blocked_roles, build_time_blocks
        path = safe_relative_file(base, settings["time_manifest"])
        if settings.get("time_sha256") and sha256(path) != settings["time_sha256"]:
            raise ValueError("Temporal manifest SHA-256 mismatch")
        frame = pd.read_csv(path, dtype={"subject_id": str, "recording_id": str})
        frame = build_time_blocks(frame, groups=dataset.groups, recording_ids=dataset.recording_ids)
        role = blocked_roles(frame, job.fold)
        return [(0, role["train"], role["validation"])], role["refit"], role["test"]
    if job.protocol == "representation-control":
        from pd_eeg.data.splits import representation_folds
        mapping_path = settings.get("participant_ids")
        mapping = None
        if mapping_path:
            frame = pd.read_csv(safe_relative_file(base, mapping_path), dtype=str)
            if frame.subject_id.duplicated().any():
                raise ValueError("Participant mapping contains duplicate subjects")
            mapping = dict(zip(frame.subject_id, frame.participant_id))
        mapping_folds = representation_folds(dataset.y, dataset.groups, participant_ids_map=mapping)
        folds = np.asarray([mapping_folds[sid] for sid in dataset.groups], dtype=np.int64)
        te = np.flatnonzero(folds == job.fold - 1)
        tr = np.flatnonzero(folds != job.fold - 1)
        return [], tr, te
    raise ValueError(f"Unknown protocol: {job.protocol}")


def prediction_inputs(path):
    """Read target features and IDs without requesting the target label array."""
    with np.load(path, allow_pickle=False) as data:
        X = np.asarray(data["X"], dtype=np.float32)
        names = tuple(map(str, data["channel_names"]))
        X = X[:, [names.index(c) for c in ("C3", "Cz", "C4")]]
        groups = np.asarray(data["groups"]).astype(str)
    if X.shape[1:] != (3, 512) or len(X) != len(groups) or not np.isfinite(X).all():
        raise ValueError("Invalid target inputs")
    return X, groups


def run_job(job: Job, config: dict, *, project_root=".", output="outputs", device="cuda",
            smoke=False):
    import torch
    from .neural import fit_neural
    base, output = Path(project_root).resolve(), Path(output).resolve()
    out = output / job.name
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}. Use a new output directory.")
    settings = config["cohorts"][job.cohort]
    source_path = safe_relative_file(base, settings["path"])
    dataset = load_dataset(source_path, expected_sha256=settings.get("sha256"),
                           cohort=None if smoke else job.cohort)
    splits, train, test = split_roles(dataset, settings, job, base)
    if job.target:
        target_settings = config["cohorts"][job.target]
        target_path = safe_relative_file(base, target_settings["path"])
        if target_settings.get("sha256") and sha256(target_path) != target_settings["sha256"]:
            raise ValueError("Target input SHA-256 mismatch")
        target_X, target_groups = prediction_inputs(target_path)
        test = np.arange(len(target_X), dtype=np.int64)
        target_y = None
    else:
        target_path = source_path
        target_X, target_groups, target_y = dataset.X[test], dataset.groups[test], dataset.y[test]
    for _, tr, va in splits:
        if np.intersect1d(tr, va).size:
            raise ValueError("Selection training and validation windows overlap")
        if job.protocol == "S" and (set(dataset.groups[test]) & (set(dataset.groups[tr]) | set(dataset.groups[va]))):
            raise ValueError("S outer-test subject enters model selection")
    representation = job.model in ("PSD-AE-RBFNN", "TDCNet", "TDCNet-reference")
    classical = job.model in ("Spectral-LR", "Harmonized-LR", "RBF-SVM")
    if not classical and not representation:
        kwargs = {"max_epochs": 2} if smoke else {}
        if job.protocol == "representation-control":
            kwargs = {"fixed_epochs": 2 if smoke else 50, "learning_rates": (1e-3,)}
        result = fit_neural(job.model, dataset.X, dataset.y, dataset.groups, splits, train, target_X,
                            seed=job.seed, variant=job.variant, device=device, **kwargs)
        artifact = {"schema_version": 1, "model": job.model, "variant": job.variant,
                    "state_dict": result.state_dict, "preprocessing": result.preprocessing,
                    "selected": result.selected}
        predictions, selection = result.probabilities, {"selected": result.selected,
                    "candidates": result.candidates, "history": result.history, "final_losses": result.losses}
        checkpoint_name = "model.pt"
    else:
        from . import representation as representations
        from . import spectral
        fitter = representations.fit_classifier if representation else spectral.fit_classifier
        kwargs = {"device": device} if representation else {}
        result = fitter(job.model, dataset.X, dataset.y, dataset.groups, splits, train, target_X,
                        seed=job.seed, cohort=job.cohort, transfer=bool(job.target),
                        max_epochs=2 if smoke else None, **kwargs)
        artifact, predictions = result["estimator"], result["probabilities"]
        selection = {k: result[k] for k in ("selected", "candidates", "history")}
        checkpoint_name = "model.joblib"
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.shape != (len(test),) or not np.isfinite(predictions).all() or np.any((predictions < 0) | (predictions > 1)):
        raise RuntimeError("Invalid test predictions")
    out.mkdir(parents=True, exist_ok=False)
    if checkpoint_name.endswith(".pt"):
        torch.save(artifact, out / checkpoint_name)
    else:
        joblib.dump(artifact, out / checkpoint_name)
    frame = pd.DataFrame({"cohort": job.target or job.cohort, "source": job.cohort,
            "protocol": job.protocol, "experiment": job.experiment, "model": job.model,
            "variant": job.variant, "seed": job.seed, "repeat": job.repeat, "fold": job.fold,
            "window_id": test, "subject_id": target_groups, "prob_pd": predictions})
    if target_y is not None:
        frame["true_label"] = target_y
    frame.to_csv(out / "predictions.csv", index=False, lineterminator="\n")
    write_json(out / "selection.json", selection)
    versions = {name: metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch")}
    write_json(out / "run.json", {"job": asdict(job), "checkpoint": checkpoint_name,
               "source_sha256": sha256(source_path), "target_sha256": sha256(target_path),
               "training_indices": train.tolist(), "test_indices": test.tolist(),
               "versions": versions, "smoke": bool(smoke), "threshold": .5})
    names = (checkpoint_name, "predictions.csv", "selection.json", "run.json")
    (out / "checksums.sha256").write_text("".join(f"{sha256(out / name)}  {name}\n" for name in names), encoding="ascii")
    return out


def verify_files(directory):
    directory = Path(directory)
    seen = set()
    for line in (directory / "checksums.sha256").read_text(encoding="ascii").splitlines():
        expected, name = line.split("  ", 1)
        if name in seen or Path(name).name != name or sha256(safe_relative_file(directory, name)) != expected:
            raise ValueError("Saved output checksum mismatch")
        seen.add(name)
    common = {"predictions.csv", "selection.json", "run.json"}
    if seen not in (common | {"model.pt"}, common | {"model.joblib"}):
        raise ValueError("Incomplete output checksum manifest")


def validate_job(directory, config, *, project_root=".", device="cpu", trust_sklearn=False, tolerance=1e-7):
    import torch
    from .neural import predict_neural
    directory = Path(directory)
    verify_files(directory)
    record = read_json(directory / "run.json")
    job = Job(**record["job"])
    path = safe_relative_file(project_root, config["cohorts"][job.target or job.cohort]["path"])
    if sha256(path) != record["target_sha256"]:
        raise ValueError("Prediction input differs from the saved run")
    X, _ = prediction_inputs(path)
    X = X[np.asarray(record["test_indices"], dtype=np.int64)]
    if record["checkpoint"] == "model.pt":
        artifact = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        p = predict_neural(artifact, X, device=device)
    else:
        if not trust_sklearn:
            raise ValueError("Loading a joblib model requires --trust-sklearn for your own trusted artifacts")
        artifact = joblib.load(directory / "model.joblib")
        if job.model in ("PSD-AE-RBFNN", "TDCNet", "TDCNet-reference"):
            from .representation import predict_classifier
        else:
            from .spectral import predict_classifier
        p = (predict_classifier(artifact, X, device=device)
             if job.model in ("PSD-AE-RBFNN", "TDCNet", "TDCNet-reference")
             else predict_classifier(artifact, X))
    saved = pd.read_csv(directory / "predictions.csv").prob_pd.to_numpy(dtype=float)
    p = np.asarray(p)
    if p.shape != saved.shape or not len(p) or not np.isfinite(p).all() or not np.isfinite(saved).all():
        raise ValueError("Invalid saved or regenerated probability vector")
    difference = float(np.max(np.abs(p - saved)))
    if difference > tolerance:
        raise ValueError(f"Prediction mismatch: maximum absolute error {difference:.9g}")
    return {"windows": len(saved), "maximum_absolute_error": difference, "tolerance": tolerance}
