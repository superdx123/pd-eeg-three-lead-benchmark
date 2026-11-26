"""Deterministic subject-LOSO and within-recording window assignments."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


DEFAULT_W_SEEDS = (11001, 22002, 33003, 44004, 55005)


def _metadata(y, groups) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    labels, subjects = np.asarray(y), np.asarray(groups).astype(str)
    if labels.ndim != 1 or subjects.shape != labels.shape or not len(labels):
        raise ValueError("Labels and groups must be nonempty one-dimensional vectors")
    if not np.issubdtype(labels.dtype, np.integer) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Labels must be binary integers")
    rows = []
    for subject in sorted(set(subjects)):
        if not subject.strip():
            raise ValueError("Subject IDs cannot be blank")
        values = np.unique(labels[subjects == subject])
        if len(values) != 1:
            raise ValueError("Each subject must have a single diagnosis label")
        rows.append({"subject_id": subject, "true_label": int(values[0])})
    return labels.astype(np.int64), subjects, pd.DataFrame(rows)


def build_nested_loso_manifests(y, groups, *, n_inner_folds: int = 5,
                                inner_seed_base: int = 730000) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, _, table = _metadata(y, groups)
    if n_inner_folds < 2:
        raise ValueError("At least two inner folds are required")
    outer_rows, inner_rows = [], []
    for outer_fold, test_row in enumerate(table.itertuples(index=False), start=1):
        test_subject = str(test_row.subject_id)
        outer_rows.append(dict(protocol="S_nested_subject_loso", outer_fold=outer_fold,
                               outer_test_subject_id=test_subject, true_label=int(test_row.true_label)))
        train = table.loc[table.subject_id != test_subject].reset_index(drop=True)
        if int(train.true_label.value_counts().min()) < n_inner_folds:
            raise ValueError("Too few subjects per class for the requested inner folds")
        seed = int(inner_seed_base + outer_fold)
        splitter = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=seed)
        assignments = np.zeros(len(train), dtype=np.int64)
        for inner_fold, (_, val) in enumerate(splitter.split(np.zeros(len(train), dtype=np.int8), train.true_label), start=1):
            assignments[val] = inner_fold
        for index, row in train.iterrows():
            inner_rows.append(dict(protocol="S_nested_subject_loso", outer_fold=outer_fold,
                                   outer_test_subject_id=test_subject, inner_split_seed=seed,
                                   subject_id=str(row.subject_id), true_label=int(row.true_label),
                                   inner_validation_fold=int(assignments[index])))
    return pd.DataFrame(outer_rows), pd.DataFrame(inner_rows)


def s_outer_indices(groups, manifest: pd.DataFrame, outer_fold: int) -> dict[str, np.ndarray]:
    subjects = np.asarray(groups).astype(str)
    row = manifest.loc[manifest.outer_fold.astype(int) == int(outer_fold)]
    if len(row) != 1:
        raise ValueError("Expected exactly one outer-fold assignment")
    test_subject = str(row.iloc[0].outer_test_subject_id)
    test = np.flatnonzero(subjects == test_subject)
    if not len(test):
        raise ValueError("Outer test subject is absent from the dataset")
    return {"train": np.flatnonzero(subjects != test_subject), "test": test}


def s_inner_splits(groups, manifest: pd.DataFrame, outer_fold: int) -> list[tuple[int, np.ndarray, np.ndarray]]:
    subjects = np.asarray(groups).astype(str)
    rows = manifest.loc[manifest.outer_fold.astype(int) == int(outer_fold)]
    if rows.empty or rows.subject_id.astype(str).duplicated().any():
        raise ValueError("Inner manifest is missing or duplicates a subject")
    output = []
    for fold in sorted(rows.inner_validation_fold.astype(int).unique()):
        valid = rows.inner_validation_fold.astype(int).eq(fold)
        train_subjects = rows.loc[~valid, "subject_id"].astype(str)
        val_subjects = rows.loc[valid, "subject_id"].astype(str)
        output.append((int(fold), np.flatnonzero(np.isin(subjects, train_subjects)),
                       np.flatnonzero(np.isin(subjects, val_subjects))))
    return output


def validate_s_manifests(y, groups, outer: pd.DataFrame, inner: pd.DataFrame, *, n_inner_folds: int = 5) -> None:
    labels, subjects, table = _metadata(y, groups)
    if len(outer) != len(table) or outer.outer_test_subject_id.astype(str).duplicated().any():
        raise ValueError("LOSO must test each subject exactly once")
    if set(outer.outer_test_subject_id.astype(str)) != set(subjects):
        raise ValueError("Outer manifest does not match dataset subjects")
    if set(outer.outer_fold.astype(int)) != set(range(1, len(table) + 1)):
        raise ValueError("Outer fold IDs must run from one to the subject count")
    for fold in range(1, len(table) + 1):
        roles = s_outer_indices(subjects, outer, fold)
        test_subject = str(subjects[roles["test"]][0])
        row = outer.loc[outer.outer_fold.astype(int) == fold].iloc[0]
        if int(row.true_label) != int(labels[roles["test"]][0]):
            raise ValueError("Outer manifest label mismatch")
        rows = inner.loc[inner.outer_fold.astype(int) == fold]
        expected = set(subjects[roles["train"]])
        if len(rows) != len(expected) or set(rows.subject_id.astype(str)) != expected:
            raise ValueError("Inner manifest does not cover exactly the outer training subjects")
        if set(rows.outer_test_subject_id.astype(str)) != {test_subject}:
            raise ValueError("Inconsistent outer-test reference in inner manifest")
        for entry in rows.itertuples(index=False):
            if int(entry.true_label) != int(np.unique(labels[subjects == str(entry.subject_id)]).item()):
                raise ValueError("Inner manifest label mismatch")
        splits = s_inner_splits(subjects, inner, fold)
        if [item[0] for item in splits] != list(range(1, n_inner_folds + 1)):
            raise ValueError("Unexpected inner fold IDs")
        covered = []
        for _, train, val in splits:
            if set(subjects[train]) & set(subjects[val]) or test_subject in set(subjects[np.r_[train, val]]):
                raise ValueError("Subject overlap between inner roles or outer test")
            if set(train) | set(val) != set(roles["train"]):
                raise ValueError("Inner roles do not cover outer training windows")
            if set(labels[train]) != {0, 1} or set(labels[val]) != {0, 1}:
                raise ValueError("Each inner role must contain both classes")
            covered.extend(val.tolist())
        if not np.array_equal(np.sort(covered), roles["train"]):
            raise ValueError("Each outer-training window must be validated once")


def build_protocol_w_manifest(y, groups, *, split_seeds: Iterable[int] = DEFAULT_W_SEEDS,
                              n_folds: int = 5) -> pd.DataFrame:
    labels, subjects, table = _metadata(y, groups)
    seeds = [int(seed) for seed in split_seeds]
    if n_folds < 2 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Window folds and split seeds must be valid and unique")
    rows = []
    for repeat, seed in enumerate(seeds, start=1):
        assignments = np.zeros(len(labels), dtype=np.int64)
        for position, subject in enumerate(table.subject_id, start=1):
            indices = np.flatnonzero(subjects == subject)
            if len(indices) < n_folds:
                raise ValueError("A subject has fewer windows than folds")
            rng = np.random.default_rng(np.random.SeedSequence([seed, position]))
            for fold, indices_in_fold in enumerate(np.array_split(rng.permutation(indices), n_folds), start=1):
                assignments[indices_in_fold] = fold
        rows.append(pd.DataFrame(dict(protocol="W_subject_overlap_window_cv", repeat_index=repeat,
                                      split_seed=seed, window_index=np.arange(len(labels), dtype=np.int64),
                                      subject_id=subjects, true_label=labels, test_fold=assignments)))
    return pd.concat(rows, ignore_index=True)


def w_role_indices(manifest: pd.DataFrame, repeat_index: int, outer_fold: int,
                    n_folds: int = 5) -> dict[str, np.ndarray]:
    if not 1 <= outer_fold <= n_folds:
        raise ValueError("Invalid test fold")
    rows = manifest.loc[manifest.repeat_index.astype(int) == int(repeat_index)]
    if rows.empty:
        raise ValueError("Missing window-CV repeat")
    folds, indices = rows.test_fold.astype(int), rows.window_index.to_numpy(dtype=np.int64)
    test, val = folds.eq(outer_fold), folds.eq(outer_fold % n_folds + 1)
    return {"train": np.sort(indices[~(test | val)]), "validation": np.sort(indices[val]),
            "test": np.sort(indices[test])}


def validate_w_manifest(y, groups, manifest: pd.DataFrame, *, split_seeds: Iterable[int] = DEFAULT_W_SEEDS,
                         n_folds: int = 5) -> None:
    labels, subjects, _ = _metadata(y, groups)
    seeds = [int(seed) for seed in split_seeds]
    if len(manifest) != len(labels) * len(seeds):
        raise ValueError("Window manifest row count mismatch")
    if set(manifest.repeat_index.astype(int)) != set(range(1, len(seeds) + 1)):
        raise ValueError("Window manifest repeat IDs mismatch")
    for repeat, seed in enumerate(seeds, start=1):
        rows = manifest.loc[manifest.repeat_index.astype(int) == repeat].sort_values("window_index")
        if not np.array_equal(rows.window_index.to_numpy(), np.arange(len(labels))):
            raise ValueError("Each window must appear exactly once per repeat")
        if set(rows.split_seed.astype(int)) != {seed}:
            raise ValueError("Window split seed mismatch")
        if not np.array_equal(rows.subject_id.astype(str).to_numpy(), subjects) or not np.array_equal(rows.true_label.to_numpy(), labels):
            raise ValueError("Window manifest metadata mismatch")
        if set(rows.test_fold.astype(int)) != set(range(1, n_folds + 1)):
            raise ValueError("Invalid window test-fold IDs")
        counts = rows.groupby(["subject_id", "test_fold"]).size().unstack(fill_value=0)
        if (counts == 0).any().any() or ((counts.max(axis=1) - counts.min(axis=1)) > 1).any():
            raise ValueError("Within-subject window folds must be nonempty and balanced")
        for fold in range(1, n_folds + 1):
            roles = w_role_indices(manifest, repeat, fold, n_folds)
            if any(set(roles[a]) & set(roles[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
                raise ValueError("Window roles overlap")
            if set(np.concatenate(list(roles.values()))) != set(range(len(labels))):
                raise ValueError("Window roles do not cover the dataset")
            if any(set(subjects[indices]) != set(subjects) for indices in roles.values()):
                raise ValueError("Every window-CV role must contain every subject")


def build_time_blocks(times: pd.DataFrame, *, groups=None, recording_ids=None) -> pd.DataFrame:
    """Assign five chronological blocks using measured, aligned recording times."""
    frame = times.copy().rename(columns={"start_seconds": "start_sec", "end_seconds": "end_sec"})
    required = {"window_index", "subject_id", "recording_id", "start_sec", "end_sec"}
    if required - set(frame.columns):
        raise ValueError(f"Time metadata is missing {sorted(required - set(frame.columns))}")
    if frame.empty or not np.array_equal(frame.window_index.to_numpy(), np.arange(len(frame))):
        raise ValueError("Time metadata must follow canonical dataset window indices")
    for field in ("subject_id", "recording_id"):
        frame[field] = frame[field].astype(str)
        if frame[field].str.strip().eq("").any():
            raise ValueError("Time metadata contains blank subject or recording IDs")
    for field, expected in (("subject_id", groups), ("recording_id", recording_ids)):
        if expected is not None and not np.array_equal(frame[field].to_numpy(), np.asarray(expected).astype(str)):
            raise ValueError("Time metadata does not align with dataset rows")
    start, stop = frame.start_sec.to_numpy(dtype=float), frame.end_sec.to_numpy(dtype=float)
    if not np.isfinite(start).all() or not np.isfinite(stop).all() or np.any(start < 0):
        raise ValueError("Actual window times must be finite and nonnegative")
    if not np.allclose(stop - start, 2.0, rtol=0.0, atol=1e-9):
        raise ValueError("Time metadata must describe 2 s windows")
    blocks = np.zeros(len(frame), dtype=np.int64)
    for _, part in frame.groupby("subject_id", sort=True):
        if part.recording_id.nunique() != 1 or len(part) < 5:
            raise ValueError("Each subject requires one recording and at least five windows")
        if np.any(np.diff(part.start_sec.to_numpy(dtype=float)) < 2.0 - 1e-9):
            raise ValueError("Retained windows must be chronological and nonoverlapping")
        ordered = part.sort_values("start_sec", kind="stable").window_index.to_numpy(dtype=np.int64)
        for block, indices in enumerate(np.array_split(ordered, 5), start=1):
            blocks[indices] = block
    frame["block"] = blocks
    return frame


def _purge_time_neighbors(frame, candidates, references, gap_seconds: float) -> np.ndarray:
    candidates, references = np.asarray(candidates, dtype=np.int64), np.asarray(references, dtype=np.int64)
    groups, starts = frame.subject_id.to_numpy(dtype=str), frame.start_sec.to_numpy(dtype=float)
    keep = np.ones(len(candidates), dtype=bool)
    for subject in np.unique(groups[candidates]):
        positions = np.flatnonzero(groups[candidates] == subject)
        ref = references[groups[references] == subject]
        if len(ref):
            distance = np.abs(starts[candidates[positions], None] - starts[ref][None, :])
            keep[positions] = np.all(distance >= 2.0 + gap_seconds - 1e-9, axis=1)
    return np.sort(candidates[keep])


def blocked_roles(frame: pd.DataFrame, fold: int, *, gap_seconds: float = 2.0,
                  return_audit: bool = False):
    """Split measured time blocks and purge a 2 s interval between every selection role."""
    if int(fold) not in range(1, 6) or gap_seconds != 2.0:
        raise ValueError("The study uses five time blocks and a fixed 2 s purge")
    checked = build_time_blocks(frame)
    if "block" in frame and not np.array_equal(frame.block.to_numpy(), checked.block.to_numpy()):
        raise ValueError("Time-block assignments differ from chronological construction")
    block = checked.block.to_numpy(dtype=np.int64)
    test = np.flatnonzero(block == fold)
    raw_val = np.flatnonzero(block == fold % 5 + 1)
    raw_train = np.flatnonzero(~np.isin(block, [fold, fold % 5 + 1]))
    val = _purge_time_neighbors(checked, raw_val, test, gap_seconds)
    train = _purge_time_neighbors(checked, raw_train, np.r_[raw_val, test], gap_seconds)
    refit = _purge_time_neighbors(checked, np.flatnonzero(block != fold), test, gap_seconds)
    roles = {"train": train, "validation": val, "refit": refit, "test": test}
    subjects = checked.subject_id.to_numpy(dtype=str)
    starts = checked.start_sec.to_numpy(dtype=float)
    for indices in roles.values():
        if not len(indices) or len(np.unique(indices)) != len(indices) or set(subjects[indices]) != set(subjects):
            raise ValueError("Purging removed a required role from at least one subject")
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"), ("refit", "test"))
    gaps = {}
    for first, second in pairs:
        if np.intersect1d(roles[first], roles[second]).size:
            raise ValueError("Blocked window roles overlap")
        gap = np.inf
        for subject in np.unique(subjects):
            a = roles[first][subjects[roles[first]] == subject]
            b = roles[second][subjects[roles[second]] == subject]
            gap = min(gap, float(np.min(np.abs(starts[a, None] - starts[b][None, :]) - 2.0)))
        if gap < gap_seconds - 1e-9:
            raise ValueError("Actual interval gap is smaller than the required purge")
        gaps[f"{first}_{second}"] = gap
    if not set(train).issubset(refit) or not set(val).issubset(refit):
        raise ValueError("Refit membership is inconsistent with selection roles")
    audit = {"minimum_actual_interval_gaps_seconds": gaps,
             "n_windows": {name: len(indices) for name, indices in roles.items()}}
    return (roles, audit) if return_audit else roles


def representation_folds(y, groups, *, participant_ids_map: dict[str, str] | None = None,
                         study_id: str = "frequency_transport_v1", seed: int = 20261021) -> dict[str, int]:
    """Reproduce the fixed, class-balanced full/R18 split from original participant strings."""
    _, _, table = _metadata(y, groups)
    if len(table) != 50 or table.true_label.value_counts().sort_index().to_dict() != {0: 25, 1: 25}:
        raise ValueError("The representation control requires 25 subjects in each class")
    mapping = {str(value): str(value) for value in table.subject_id} if participant_ids_map is None else participant_ids_map
    if set(mapping) != set(table.subject_id) or len(set(mapping.values())) != len(mapping):
        raise ValueError("Participant mapping must be one-to-one and cover every dataset subject")
    if any(not isinstance(value, str) or not value for value in mapping.values()):
        raise ValueError("Original participant IDs must be nonempty strings")
    result = {}
    for label in (0, 1):
        people = table.loc[table.true_label == label, "subject_id"].astype(str).tolist()
        def order(subject):
            message = f"{study_id}|{seed}|{label}|{mapping[subject]}"
            return hashlib.sha256(message.encode("utf-8")).hexdigest()
        for index, subject in enumerate(sorted(people, key=order)):
            result[subject] = index % 5
    return {subject: result[subject] for subject in sorted(result)}
