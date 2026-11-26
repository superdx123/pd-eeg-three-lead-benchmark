"""Subject-level probability aggregation and conditional bootstrap intervals."""

from __future__ import annotations

from collections import defaultdict
import csv
from itertools import combinations
from pathlib import Path
from statistics import fmean

import numpy as np

METRICS = ("auc", "balanced_accuracy", "sensitivity", "specificity", "brier_score")
MODEL_NAMES = {
    "DS28": "BE-MSNet", "DS28_BR_MS_AUG_LS_WD3": "BE-MSNet",
    "be_msnet": "BE-MSNet", "be-msnet": "BE-MSNet", "PrimaryCNN": "4L-CNN",
    "primary_cnn": "4L-CNN", "LeeCRNN": "Lee-CRNN", "PSDAE": "PSD-AE-RBFNN",
    "RBFSVM": "RBF-SVM", "LR": "Spectral LR", "harmonized_LR": "Harmonized LR",
    "TDCNet": "TDCNet†", "strict_tdc": "TDCNet (strict inner fitting)",
}


def aggregate_subjects(groups, labels, probabilities):
    """Average float64 window probabilities within lexically sorted subjects."""
    import pandas as pd

    frame = pd.DataFrame({"subject_id": np.asarray(groups).astype(str),
                          "true_label": np.asarray(labels, dtype=np.int64),
                          "prob_pd": np.asarray(probabilities, dtype=np.float64)})
    if frame.empty or frame.isna().any().any() or not frame["true_label"].isin([0, 1]).all():
        raise ValueError("Window predictions must have valid subject IDs and binary labels")
    if not np.isfinite(frame["prob_pd"]).all() or not frame["prob_pd"].between(0, 1).all():
        raise ValueError("Window probabilities must be finite in [0, 1]")
    if (frame.groupby("subject_id", sort=True)["true_label"].nunique() != 1).any():
        raise ValueError("A subject has inconsistent labels")
    result = frame.groupby("subject_id", sort=True).agg(
        true_label=("true_label", "first"), prob_pd=("prob_pd", "mean"),
        n_windows=("prob_pd", "size")).reset_index()
    result["predicted_label"] = (result["prob_pd"] >= 0.5).astype(np.int64)
    return result


def _field(row, *names, default=None):
    for name in names:
        if name in row and str(row[name]).strip() != "":
            return row[name]
    if default is not None:
        return default
    raise ValueError("Missing prediction column: " + names[0])


def normalize_row(row):
    """Accept public CSV names and the corresponding recorded-output aliases."""
    label = int(_field(row, "true_label", "label", "y"))
    probability = float(_field(row, "prob_pd", "probability", "prob"))
    if label not in (0, 1) or not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Labels must be HC=0 / PD=1 and probabilities finite in [0, 1]")
    protocol = str(_field(row, "protocol")).strip()
    if protocol.lower() in ("s", "loso", "s_nested_subject_loso"):
        protocol = "S"
    elif protocol.lower() in ("w", "w_5x5", "within_recording"):
        protocol = "W"
    elif protocol.lower() in ("time_blocked", "time-blocked", "contiguous"):
        protocol = "time_blocked"
    model = str(_field(row, "model")).strip()
    experiment = str(_field(row, "experiment", default="")).strip()
    variant = str(_field(row, "variant", default="full")).strip()
    subject = str(_field(row, "subject_id", "subject")).strip()
    cohort = str(_field(row, "cohort")).strip()
    if not all((model, subject, cohort, protocol)):
        raise ValueError("Cohort, protocol, model and subject must not be empty")
    name = MODEL_NAMES.get(model, model)
    if model in ("TDCNet-reference", "TDCNet-dagger"):
        name = "TDCNet†"
    elif model == "TDCNet" and experiment == "strict-tdc":
        name = "TDCNet (strict inner fitting)"
    if variant != "full" and not name.endswith(f" ({variant})"):
        name += f" ({variant})"
    return {
        "cohort": cohort, "protocol": protocol, "model": name,
        "experiment": experiment, "source": str(_field(row, "source", default="")),
        "variant": variant,
        "seed": int(_field(row, "seed", default=0)), "subject_id": subject,
        "true_label": label, "prob_pd": probability,
        "repeat": int(_field(row, "repeat", "repeat_index", default=0)),
        "fold": int(_field(row, "fold", "test_fold", "outer_fold", default=0)),
        "window_id": str(_field(row, "window_id", "window_index", default="")),
    }


def read_predictions(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return [normalize_row(row) for row in csv.DictReader(stream)]


def subject_metrics(labels, probabilities, threshold=0.5):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.shape != labels.shape:
        raise ValueError("One probability is required per subject")
    if set(labels.tolist()) != {0, 1} or not np.isfinite(probabilities).all():
        raise ValueError("Both diagnostic classes and finite probabilities are required")
    if not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Probabilities must lie in [0, 1]")
    positive = probabilities[labels == 1]
    negative = probabilities[labels == 0]
    auc = ((positive[:, None] > negative[None, :])
           + 0.5 * (positive[:, None] == negative[None, :])).mean()
    sensitivity = (positive >= threshold).mean()
    specificity = (negative < threshold).mean()
    return {
        "auc": float(auc), "balanced_accuracy": float((sensitivity + specificity) / 2),
        "sensitivity": float(sensitivity), "specificity": float(specificity),
        "brier_score": float(np.mean((probabilities - labels) ** 2)),
    }


binary_metrics = subject_metrics


def stratified_counts(labels, replicates=2000, seed=20260930, *, alternating=False):
    """Draw all positive subjects first, then negatives, using NumPy PCG64."""
    labels = np.asarray(labels, dtype=np.int64)
    if set(labels.tolist()) != {0, 1} or replicates < 1:
        raise ValueError("Both classes and a positive number of bootstrap draws are required")
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    generator = np.random.default_rng(seed)
    if alternating:
        # The fixed-fold representation comparison draws HC then PD per replicate.
        draws = [(generator.integers(0, len(negative), size=len(negative)),
                  generator.integers(0, len(positive), size=len(positive))) for _ in range(replicates)]
        inn, ip = map(np.asarray, zip(*draws))
    else:
        ip = generator.choice(len(positive), (replicates, len(positive)), replace=True)
        inn = generator.choice(len(negative), (replicates, len(negative)), replace=True)
    return (np.eye(len(positive), dtype=np.int64)[ip].sum(axis=1),
            np.eye(len(negative), dtype=np.int64)[inn].sum(axis=1))


def bootstrap_fold_metrics(labels, fold_probabilities, positive_counts, negative_counts,
                           threshold=0.5):
    """Reuse each subject's multiplicity across every test fold."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(fold_probabilities, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != len(labels):
        raise ValueError("Expected a folds-by-subjects probability matrix")
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("Invalid fold probabilities")
    positive = scores[:, labels == 1]
    negative = scores[:, labels == 0]
    wp, wn = np.asarray(positive_counts), np.asarray(negative_counts)
    npos, nneg = positive.shape[1], negative.shape[1]
    if wp.ndim != 2 or wn.ndim != 2 or wp.shape[0] != wn.shape[0]:
        raise ValueError("Bootstrap counts have incompatible shapes")
    if wp.shape[1] != npos or wn.shape[1] != nneg:
        raise ValueError("Bootstrap counts differ from class sizes")
    if (wp < 0).any() or (wn < 0).any() or not np.all(wp.sum(axis=1) == npos) or not np.all(wn.sum(axis=1) == nneg):
        raise ValueError("Bootstrap multiplicities must preserve class sizes")
    output = {name: np.zeros(wp.shape[0], dtype=float) for name in METRICS}
    for pos, neg in zip(positive, negative):
        wins = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
        sensitivity = wp @ (pos >= threshold) / npos
        specificity = wn @ (neg < threshold) / nneg
        output["auc"] += np.einsum("bi,ij,bj->b", wp, wins, wn) / (npos * nneg)
        output["sensitivity"] += sensitivity
        output["specificity"] += specificity
        output["balanced_accuracy"] += (sensitivity + specificity) / 2
        output["brier_score"] += (wp @ ((pos - 1) ** 2) + wn @ (neg ** 2)) / (npos + nneg)
    return {name: values / len(scores) for name, values in output.items()}


def _model_scores(rows, protocol, *, ensemble_windows=False):
    labels, window_groups, window_sets = {}, defaultdict(list), defaultdict(set)
    window_probabilities = defaultdict(list)
    for row in rows:
        subject, label = row["subject_id"], row["true_label"]
        if subject in labels and labels[subject] != label:
            raise ValueError("A subject has inconsistent labels")
        labels[subject] = label
        key = (row["repeat"], row["fold"], row["seed"], subject)
        if row["window_id"]:
            if row["window_id"] in window_sets[key]:
                raise ValueError("Duplicate window identifier within a seed and test fold")
            window_sets[key].add(row["window_id"])
        window_groups[key].append(row["prob_pd"])
        if ensemble_windows:
            if not row["window_id"]:
                raise ValueError("Window identifiers are required for a window ensemble")
            window_probabilities[(row["repeat"], row["fold"], subject, row["window_id"])].append(
                row["prob_pd"])
    subjects = sorted(labels)
    truth = np.array([labels[subject] for subject in subjects], dtype=np.int64)
    seeds = sorted({row["seed"] for row in rows})
    average = fmean if ensemble_windows else np.mean
    means = {key: float(average(values)) for key, values in window_groups.items()}
    foldwise = protocol in ("W", "time_blocked")
    fold_keys = sorted({(row["repeat"], row["fold"]) for row in rows}) if foldwise else [(0, 0)]
    by_seed = np.empty((len(seeds), len(fold_keys), len(subjects)), dtype=float)
    for si, seed in enumerate(seeds):
        for fi, (repeat, fold) in enumerate(fold_keys):
            for ji, subject in enumerate(subjects):
                keys = ([(repeat, fold, seed, subject)] if foldwise else
                        [key for key in means if key[2:] == (seed, subject)])
                if len(keys) != 1 or keys[0] not in means:
                    raise ValueError("Every subject must have one held-out score per seed and test unit")
                by_seed[si, fi, ji] = means[keys[0]]
    # Matched seeds must refer to the same test windows, when IDs are supplied.
    for repeat, fold in {(r[0], r[1]) for r in means}:
        for subject in subjects:
            sets = [window_sets[(repeat, fold, seed, subject)] for seed in seeds]
            nonempty = [value for value in sets if value]
            if nonempty and (len(nonempty) != len(sets) or any(value != nonempty[0] for value in nonempty)):
                raise ValueError("Training seeds use different test windows")
    if ensemble_windows:
        window_means = defaultdict(list)
        for (repeat, fold, subject, _), probabilities in window_probabilities.items():
            if len(probabilities) != len(seeds):
                raise ValueError("Each test window must have one probability per matched seed")
            unit = (repeat, fold) if foldwise else (0, 0)
            window_means[(*unit, subject)].append(fmean(probabilities))
        scores = np.array([[fmean(window_means[(repeat, fold, subject)]) for subject in subjects]
                           for repeat, fold in fold_keys], dtype=float)
    else:
        scores = by_seed.mean(axis=0)
    return subjects, truth, scores, by_seed, seeds, fold_keys


def calibration_bins(labels, probabilities):
    """Fixed probability bins; the last bin includes 1.0."""
    edges = np.linspace(0., 1., 6)
    edges[3] = .6
    membership = np.minimum(np.searchsorted(edges, probabilities, side="right") - 1, 4)
    bins = []
    for i in range(5):
        selected = membership == i
        bins.append({"lower": float(edges[i]), "upper": float(edges[i + 1]),
                     "count": int(selected.sum()),
                     "mean_probability": float(probabilities[selected].mean()) if selected.any() else None,
                     "observed_pd_fraction": float(labels[selected].mean()) if selected.any() else None})
    return bins


def cohort_contrast(cohorts):
    """Estimate how the paired BE-MSNet minus FBCNet difference changes by cohort."""
    sequence = np.random.SeedSequence(20261023).spawn(2)
    points, distributions = [], []
    for (labels, left, right), child in zip(cohorts, sequence):
        counts = stratified_counts(labels, 10000, child)
        lboot = bootstrap_fold_metrics(labels, left, *counts)
        rboot = bootstrap_fold_metrics(labels, right, *counts)
        lp, rp = subject_metrics(labels, left[0]), subject_metrics(labels, right[0])
        points.append({key: lp[key] - rp[key] for key in ("auc", "balanced_accuracy")})
        distributions.append({key: lboot[key] - rboot[key] for key in points[-1]})
    return {key: {"difference_778": points[0][key], "difference_490": points[1][key],
                  "point": points[0][key] - points[1][key],
                  "ci95": np.quantile(distributions[0][key] - distributions[1][key], [.025, .975]).tolist(),
                  "within_778_ci95": np.quantile(distributions[0][key], [.025, .975]).tolist(),
                  "within_490_ci95": np.quantile(distributions[1][key], [.025, .975]).tolist()}
            for key in points[0]}


def analyze_predictions(rows, *, replicates=2000, seed=None, threshold=0.5,
                        reference="BE-MSNet", ensemble_windows=False):
    """Aggregate window probabilities, then estimate absolute and paired intervals.

    S, transfer and fixed subject-fold representation controls pool one OOF score
    per subject. W and time-blocked evaluation average metrics over test folds.
    """
    if threshold != 0.5:
        raise ValueError("The published benchmark uses a fixed threshold of 0.5")
    clean = [normalize_row(row) for row in rows]
    if not clean:
        raise ValueError("Prediction table is empty")
    panels = defaultdict(lambda: defaultdict(list))
    for row in clean:
        panels[(row["cohort"], row["protocol"], row["experiment"], row["source"])][row["model"]].append(row)
    result = {"threshold": threshold, "bootstrap": {"replicates": replicates, "seed": seed,
              "sampling_unit": "subject", "stratified_by": "diagnosis",
              "interval": "percentile_95_conditional_on_fixed_predictions"}, "panels": []}
    contrast_inputs = {}
    for (cohort, protocol, experiment, source), model_rows in sorted(panels.items()):
        matrices = {model: _model_scores(values, protocol, ensemble_windows=ensemble_windows)
                    for model, values in model_rows.items()}
        first = next(iter(matrices.values()))
        subjects, truth, _, _, _, folds = first
        for ids, labels, _, _, _, model_folds in matrices.values():
            if ids != subjects or not np.array_equal(labels, truth) or model_folds != folds:
                raise ValueError("Paired models must use the same subjects, labels and test units")
        panel_seed = seed
        if panel_seed is None:
            panel_seed = 20261018 if experiment in ("harmonized-lr", "time-blocked") else 20260930
        wp, wn = stratified_counts(truth, replicates, panel_seed, alternating=experiment == "representation-control")
        panel = {"cohort": cohort, "protocol": protocol, "experiment": experiment, "source": source, "subjects": len(subjects),
                 "test_units": len(folds), "bootstrap_seed": panel_seed, "models": {}, "paired_differences": []}
        distributions = {}
        if protocol == "S" and experiment == "benchmark-s" and {"BE-MSNet", "FBCNet"} <= matrices.keys():
            left, right = matrices["BE-MSNet"], matrices["FBCNet"]
            if left[4] == right[4] == [11, 22, 33, 44, 55] and len(subjects) == {"ds002778": 31, "ds003490": 50}.get(cohort):
                contrast_inputs[cohort] = (truth, left[2], right[2])
        for model, (_, _, scores, per_seed, training_seeds, _) in sorted(matrices.items()):
            point = {name: float(np.mean([subject_metrics(truth, row)[name] for row in scores])) for name in METRICS}
            boots = bootstrap_fold_metrics(truth, scores, wp, wn)
            distributions[model] = boots
            panel["models"][model] = {
                "seeds": training_seeds,
                "metrics": {name: {"point": point[name], "ci95": np.quantile(boots[name], [.025, .975], method="linear").tolist()} for name in METRICS},
                "per_seed": [{"seed": training_seed, **{name: float(np.mean([subject_metrics(truth, row)[name] for row in probability_matrix])) for name in METRICS}}
                             for training_seed, probability_matrix in zip(training_seeds, per_seed)],
            }
            if protocol in ("S", "transfer"):
                panel["models"][model]["calibration_bins"] = calibration_bins(truth, scores[0])
        pairs = ([(reference, name) for name in panel["models"] if name != reference] if reference in panel["models"]
                 else list(combinations(panel["models"], 2)))
        for left, right in pairs:
            panel["paired_differences"].append({
                "left": left, "right": right,
                "metrics": {name: {
                    "difference": panel["models"][left]["metrics"][name]["point"] - panel["models"][right]["metrics"][name]["point"],
                    "ci95": np.quantile(distributions[left][name] - distributions[right][name], [.025, .975], method="linear").tolist(),
                } for name in METRICS},
            })
        result["panels"].append(panel)
    if {"ds002778", "ds003490"} <= contrast_inputs.keys():
        result["cohort_contrast"] = cohort_contrast([contrast_inputs["ds002778"], contrast_inputs["ds003490"]])
    return result
