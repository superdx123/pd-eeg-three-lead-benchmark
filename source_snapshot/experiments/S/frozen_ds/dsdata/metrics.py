"""Metrics and subject aggregation (same semantics as the frozen E21 pipeline)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

THRESHOLD = 0.5
METRIC_NAMES = ("auc", "balanced_accuracy", "accuracy", "f1", "mcc")


def binary_probability_metrics_full(y_true, y_probability, threshold: float = THRESHOLD,
                                    require_both_classes: bool = True) -> dict:
    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    probability = np.asarray(y_probability, dtype=np.float64).reshape(-1)
    if len(truth) != len(probability) or len(truth) == 0:
        raise ValueError("Metric input lengths mismatch or empty")
    if not np.isfinite(probability).all():
        raise ValueError("Probabilities contain NaN/Inf")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("Probabilities outside [0,1]")
    prediction = (probability >= float(threshold)).astype(np.int64)
    unique = np.unique(truth).tolist()
    if set(unique) != {0, 1}:
        if require_both_classes:
            raise ValueError("Metrics require both classes present")
        return {
            "auc": float("nan"),
            "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
            "accuracy": float(accuracy_score(truth, prediction)),
            "f1": float(f1_score(truth, prediction, zero_division=0)),
            "mcc": float(matthews_corrcoef(truth, prediction)),
        }
    return {
        "auc": float(roc_auc_score(truth, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(truth, prediction)),
    }


def _aggregate_probs(values: np.ndarray, agg: str) -> np.ndarray:
    """Aggregate window probabilities within a subject (fixed label-independent rules)."""
    values = np.asarray(values, dtype=np.float64)
    if agg == "mean":
        return np.mean(values)
    if agg == "median":
        return np.median(values)
    if agg == "sig_mean":
        z = np.log(np.clip(values, 1e-7, 1 - 1e-7) / np.clip(1 - values, 1e-7, 1 - 1e-7))
        return 1.0 / (1.0 + np.exp(-np.mean(z)))
    if agg in ("trim10", "trim15", "trim20"):
        frac = {"trim10": 0.10, "trim15": 0.15, "trim20": 0.20}[agg]
        k = int(len(values) * frac)
        if k == 0:
            return np.mean(values)
        if 2 * k >= len(values):
            return np.median(values)
        return np.mean(np.sort(values)[k:-k])
    if agg == "iqr_mean":
        q1, q3 = np.percentile(values, [25, 75])
        sel = values[(values >= q1) & (values <= q3)]
        return np.mean(sel) if len(sel) else np.mean(values)
    raise ValueError(f"Unknown subject aggregation {agg!r}")


def aggregate_true_label_subject_probabilities(
    subject_ids, labels, probabilities, threshold: float = THRESHOLD, agg: str = "mean"
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "subject_id": np.asarray(subject_ids).astype(str),
            "true_label": np.asarray(labels, dtype=np.int64),
            "prob_pd": np.asarray(probabilities, dtype=np.float64),
        }
    )
    if frame.empty or frame.isna().any().any():
        raise ValueError("Aggregation input is empty or has NaNs")
    label_counts = frame.groupby("subject_id", sort=True)["true_label"].nunique()
    if (label_counts != 1).any():
        raise ValueError("A subject has more than one true label")
    aggregated = (
        frame.groupby("subject_id", sort=True, as_index=False)
        .agg(true_label=("true_label", "first"),
             prob_pd=("prob_pd", lambda v: _aggregate_probs(v.to_numpy(), agg)),
             n_windows=("prob_pd", "size"))
        .reset_index(drop=True)
    )
    aggregated["predicted_label"] = (
        aggregated["prob_pd"].to_numpy(dtype=float) >= float(threshold)
    ).astype(np.int64)
    return aggregated


def subject_metrics_from_frame(frame: pd.DataFrame, agg: str = "mean") -> dict:
    subjects = aggregate_true_label_subject_probabilities(
        frame["subject_id"], frame["true_label"], frame["prob_pd"], agg=agg
    )
    return binary_probability_metrics_full(subjects["true_label"], subjects["prob_pd"])


def _softmax_norm(values: np.ndarray) -> np.ndarray:
    z = values - np.max(values)
    e = np.exp(z)
    return e / e.sum()


def _sigmoid(z) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def aggregate_subject_probabilities_attention(
    subject_ids, labels, pd_logits, weights, threshold: float = THRESHOLD
) -> pd.DataFrame:
    """Subject aggregation with learned per-window attention weights (Round-19 MIL).

    Per subject: a = softmax(window weights); subject pd-logit = sum(a * pd_logit);
    subject prob = sigmoid(subject pd-logit). Logit-space pooling, consistent with
    the training-time subject CE (subject logit = weighted-mean of window logits).
    Used by models with params.subject_attn=True (replaces the frozen mean rule
    for that candidate only; the standard mean+0.5 protocol is untouched).
    """
    frame = pd.DataFrame(
        {
            "subject_id": np.asarray(subject_ids).astype(str),
            "true_label": np.asarray(labels, dtype=np.int64),
            "pd_logit": np.asarray(pd_logits, dtype=np.float64),
            "window_weight": np.asarray(weights, dtype=np.float64),
        }
    )
    if frame.empty or frame.isna().any().any():
        raise ValueError("Attention aggregation input is empty or has NaNs")
    label_counts = frame.groupby("subject_id", sort=True)["true_label"].nunique()
    if (label_counts != 1).any():
        raise ValueError("A subject has more than one true label")
    rows = []
    for sid, grp in frame.groupby("subject_id", sort=True):
        a = _softmax_norm(grp["window_weight"].to_numpy(dtype=np.float64))
        score = float(np.sum(a * grp["pd_logit"].to_numpy(dtype=np.float64)))
        rows.append({
            "subject_id": sid,
            "true_label": int(grp["true_label"].iloc[0]),
            "prob_pd": float(_sigmoid(score)),
            "n_windows": int(len(grp)),
        })
    aggregated = pd.DataFrame(rows)
    aggregated["predicted_label"] = (
        aggregated["prob_pd"].to_numpy(dtype=float) >= float(threshold)
    ).astype(np.int64)
    return aggregated


def window_metrics_from_frame(frame: pd.DataFrame) -> dict:
    return binary_probability_metrics_full(frame["true_label"], frame["prob_pd"])
