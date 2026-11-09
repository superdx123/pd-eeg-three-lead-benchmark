"""Independently check saved primary metrics; no EEG, models, fitting or inference."""
from pathlib import Path
from collections import defaultdict
import csv
import json
import math

ROOT = Path(__file__).resolve().parent


def read(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def metrics(rows):
    assert len(rows) == 31 and len({r['subject_id'] for r in rows}) == 31
    labels = [int(r['true_label']) for r in rows]
    probabilities = [float(r['prob_pd']) for r in rows]
    assert labels.count(0) == 16 and labels.count(1) == 15
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
    positive = [p for p, y in zip(probabilities, labels) if y == 1]
    negative = [p for p, y in zip(probabilities, labels) if y == 0]
    auc = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative) / 240
    bacc = (sum(p >= 0.5 for p in positive) / 15 + sum(p < 0.5 for p in negative) / 16) / 2
    return auc, bacc


def main():
    s = metrics(read(ROOT / 'metadata/predictions/S_ensemble.csv'))
    groups = defaultdict(list)
    for row in read(ROOT / 'metadata/predictions/W_ensemble_by_fold.csv'):
        groups[(int(row['repeat_index']), int(row['fold']))].append(row)
    assert set(groups) == {(r, f) for r in range(1, 6) for f in range(1, 6)}
    folds = [metrics(groups[k]) for k in sorted(groups)]
    w = tuple(sum(values[j] for values in folds) / 25 for j in range(2))
    expected_s = (0.8458333333333333, 0.7416666666666667)
    expected_w = (0.9913333333333334, 0.9378333333333333)
    assert all(abs(a - b) <= 1e-12 for a, b in zip(s + w, expected_s + expected_w))
    print(json.dumps({'status': 'PASS_SAVED_PRIMARY_SCORE_CHECK',
                      'S': {'auc': s[0], 'balanced_accuracy': s[1]},
                      'W': {'auc': w[0], 'balanced_accuracy': w[1]},
                      'training_inference_threshold_selection': False,
                      'W_estimator': 'arithmetic_mean_of_25_subject_test_fold_metrics'}))


if __name__ == '__main__':
    main()
