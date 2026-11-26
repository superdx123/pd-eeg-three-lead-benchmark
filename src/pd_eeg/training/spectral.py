"""Training-only spectral preprocessing, inner selection and final classifiers."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pd_eeg.analysis.metrics import aggregate_subjects
from .specs import model_name

CS = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
SVM_CS = (0.01, 0.1, 1.0, 10.0, 100.0)
SVM_GAMMAS = (0.001, 0.01, "scale", 0.1, 1.0)
LEGACY_BANDS = ((1., 4.), (4., 8.), (8., 13.), (13., 20.), (20., 30.))
COMMON_BANDS = ((1., 4.), (4., 8.), (8., 13.), (13., 30.), (30., 40.))


def _windows(X):
    result = np.asarray(X, dtype=np.float32)
    if result.ndim != 3 or result.shape[1:] != (3, 512) or len(result) == 0:
        raise ValueError("Expected nonempty waveform data shaped [N, 3, 512].")
    if not np.isfinite(result).all():
        raise ValueError("Waveforms contain non-finite values.")
    return result


def validate_inputs(X, y, groups, splits, final_train, prediction_X, *, transfer=False):
    X, prediction_X = _windows(X), _windows(prediction_X)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups).astype(str)
    if y.shape != (len(X),) or groups.shape != y.shape or set(y.tolist()) != {0, 1}:
        raise ValueError("Labels and groups must align with the binary training cohort.")
    for subject in np.unique(groups):
        if len(np.unique(y[groups == subject])) != 1:
            raise ValueError("A subject has inconsistent labels.")
    final_train = np.asarray(final_train, dtype=np.int64)
    if final_train.ndim != 1 or len(final_train) == 0 or len(np.unique(final_train)) != len(final_train):
        raise ValueError("Final training indices must be nonempty and unique.")
    if np.min(final_train) < 0 or np.max(final_train) >= len(X) or set(y[final_train]) != {0, 1}:
        raise ValueError("Invalid final training membership or diagnostic coverage.")
    normalized = []
    for number, train, validation in splits:
        train, validation = np.asarray(train, dtype=np.int64), np.asarray(validation, dtype=np.int64)
        if train.ndim != 1 or validation.ndim != 1 or not len(train) or not len(validation):
            raise ValueError("Inner splits must contain nonempty index vectors.")
        combined = np.r_[train, validation]
        same_partition = np.array_equal(np.sort(combined), np.sort(final_train))
        source_subset = transfer and len(combined) < len(final_train) and np.isin(combined, final_train).all()
        if len(np.unique(combined)) != len(combined) or not (same_partition or source_subset):
            raise ValueError("Each inner split must partition final training data without overlap.")
        if set(y[train]) != {0, 1}:
            raise ValueError("Both classes are required in inner training data.")
        normalized.append((int(number), train, validation))
    return X, y, groups, normalized, final_train, prediction_X


def five_band_features(X):
    """Channel-major 15 features, normalized by the sum of five legacy bands."""
    values = np.asarray(_windows(X), dtype=np.float64)
    frequencies = np.fft.rfftfreq(512, d=1. / 256.)
    centered = values - values.mean(axis=-1, keepdims=True)
    power = np.square(np.abs(np.fft.rfft(centered * np.hanning(512).reshape(1, 1, -1), axis=-1)))
    bands = []
    for index, (low, high) in enumerate(LEGACY_BANDS):
        mask = (frequencies >= low) & ((frequencies <= high) if index == len(LEGACY_BANDS) - 1 else (frequencies < high))
        bands.append(power[..., mask].sum(axis=-1))
    stacked = np.stack(bands, axis=-1)
    denominator = stacked.sum(axis=-1, keepdims=True)
    if np.any(denominator <= 0):
        raise ValueError("A channel-window has no power in the selected bands.")
    return np.log10(stacked / denominator + 1e-12).reshape(len(values), -1).astype(np.float32)


def full_spectrum_features(X):
    """Band-major 15 features, half-open bands and all nonnegative FFT bins."""
    values = _windows(X)
    centered = values - values.mean(axis=-1, keepdims=True)
    power = np.abs(np.fft.rfft(centered * np.hanning(512).astype(np.float32), axis=-1)) ** 2
    relative = power / np.maximum(power.sum(axis=-1, keepdims=True), 1e-12)
    frequencies = np.fft.rfftfreq(512, d=1. / 256.)
    result = np.concatenate([
        np.log10(np.maximum(relative[..., (frequencies >= low) & (frequencies < high)].sum(axis=-1), 1e-12))
        for low, high in COMMON_BANDS
    ], axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Spectral features contain non-finite values.")
    return result


def scalar_fit(X):
    return {"mean": float(np.mean(X, dtype=np.float64)),
            "std": float(np.std(X, dtype=np.float64) + 1e-8)}


def transform_features(X, recipe, scalar=None):
    if recipe == "five_band":
        return five_band_features(X)
    if recipe == "harmonized":
        return full_spectrum_features(X)
    if recipe == "source_scalar":
        if scalar is None:
            raise ValueError("Source-scalar features need training-fitted statistics.")
        return full_spectrum_features((_windows(X) - scalar["mean"]) / scalar["std"])
    raise ValueError(f"Unknown feature recipe: {recipe}")


def _fit(name, features, labels, C, gamma, seed, recipe):
    if name == "RBF-SVM":
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(C=float(C), kernel="rbf", gamma=gamma,
                class_weight="balanced", probability=True, shrinking=True,
                tol=1e-3, max_iter=-1, cache_size=2048.0, random_state=int(seed))),
        ])
        estimator.fit(np.asarray(features, dtype=np.float64), labels)
        if estimator.named_steps["classifier"].fit_status_ != 0:
            raise RuntimeError("RBF-SVM did not converge.")
        return estimator
    if recipe == "source_scalar":
        return LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000).fit(features, labels)
    estimator = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=float(C), class_weight="balanced", solver="lbfgs",
            penalty="l2", max_iter=5000, tol=1e-8, random_state=0)),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(np.asarray(features, dtype=np.float64), labels)
    return estimator


def _probability(classifier, features):
    if not np.array_equal(np.asarray(classifier.classes_), np.array([0, 1])):
        raise ValueError("Expected classes HC=0 and PD=1.")
    values = features if isinstance(classifier, LogisticRegression) else np.asarray(features, dtype=np.float64)
    result = np.asarray(classifier.predict_proba(values)[:, 1], dtype=np.float64)
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError("Classifier returned invalid PD probabilities.")
    return result


def validation_metrics(frames):
    subjects = pd.concat(frames, ignore_index=True).sort_values("subject_id")
    if subjects.subject_id.duplicated().any():
        raise ValueError("Inner validation predictions duplicate a subject across splits.")
    if set(subjects.true_label) != {0, 1}:
        raise ValueError("Pooled validation subjects must contain both classes.")
    return {"auc": float(roc_auc_score(subjects.true_label, subjects.prob_pd)),
            "balanced_accuracy": float(balanced_accuracy_score(subjects.true_label, subjects.prob_pd >= .5))}


def fit_classifier(name, X, y, groups, splits, final_train, prediction_X, seed=0,
                   cohort="ds002778", transfer=False, max_epochs=None):
    """Select using inner subjects, refit on final training data, then predict."""
    name = model_name(name)
    if name not in {"Spectral-LR", "RBF-SVM", "Harmonized-LR"}:
        raise ValueError(f"Not a spectral classifier: {name}")
    cohort = {"778": "ds002778", "490": "ds003490"}.get(str(cohort), str(cohort))
    if cohort not in {"ds002778", "ds003490"}:
        raise ValueError("Choose ds002778 or ds003490.")
    X, y, groups, splits, final_train, prediction_X = validate_inputs(
        X, y, groups, splits, final_train, prediction_X, transfer=transfer)
    recipe = ("five_band" if name == "RBF-SVM" else "harmonized" if name == "Harmonized-LR" else
              "source_scalar" if transfer or cohort == "ds003490" else "five_band")
    candidates = []
    selected = {"C": 1.0, "selection": "fixed"}
    if recipe != "source_scalar":
        if not splits:
            raise ValueError("This configuration requires inner validation splits.")
        grid = [(c, gamma) for c in SVM_CS for gamma in SVM_GAMMAS] if name == "RBF-SVM" else [(c, None) for c in CS]
        for rank, (C, gamma) in enumerate(grid):
            frames = []
            for _, train, validation in splits:
                ftrain = transform_features(X[train], recipe)
                fvalidation = transform_features(X[validation], recipe)
                classifier = _fit(name, ftrain, y[train], C, gamma, seed, recipe)
                p = _probability(classifier, fvalidation)
                frames.append(aggregate_subjects(groups[validation], y[validation], p))
            candidates.append({"C": float(C), "gamma": gamma, "rank": rank, **validation_metrics(frames)})
        if name == "RBF-SVM":
            selected = min(candidates, key=lambda row: (-row["balanced_accuracy"], -row["auc"], row["rank"]))
        else:
            selected = candidates[0]
            for row in candidates[1:]:
                if row["balanced_accuracy"] > selected["balanced_accuracy"] + 1e-12 or (
                    abs(row["balanced_accuracy"] - selected["balanced_accuracy"]) <= 1e-12
                    and row["auc"] > selected["auc"] + 1e-12):
                    selected = row
    scalar = scalar_fit(X[final_train]) if recipe == "source_scalar" else None
    features = transform_features(X[final_train], recipe, scalar)
    classifier = _fit(name, features, y[final_train], selected["C"], selected.get("gamma"), seed, recipe)
    estimator = {"model": name, "recipe": recipe, "scalar": scalar, "classifier": classifier}
    return {"probabilities": predict_classifier(estimator, prediction_X), "estimator": estimator,
            "selected": dict(selected), "candidates": candidates, "history": []}


def predict_classifier(artifact, X):
    """Predict with an already-fitted object; this function performs no loading."""
    estimator = artifact.get("estimator", artifact)
    return _probability(estimator["classifier"], transform_features(X, estimator["recipe"], estimator["scalar"]))
