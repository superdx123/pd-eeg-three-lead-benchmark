"""Protocol W/S split builders driven by the frozen manifests."""
from __future__ import annotations

import numpy as np
import pandas as pd


def get_protocol_w_role_indices(
    w_manifest: pd.DataFrame, repeat_index: int, test_fold: int, n_folds: int = 5
) -> dict:
    """Return {'train','validation','test'} window-index arrays for one W repeat x fold.

    validation_fold = test_fold % n_folds + 1 (fixed rule).
    """
    repeat_rows = w_manifest[w_manifest["repeat_index"].astype(int) == int(repeat_index)]
    if repeat_rows.empty:
        raise ValueError(f"No W manifest rows for repeat {repeat_index}")

    test_rows = repeat_rows[repeat_rows["test_fold"].astype(int) == int(test_fold)]
    if test_rows.empty:
        raise ValueError(f"No W manifest rows for repeat {repeat_index} fold {test_fold}")
    test_idx = np.sort(test_rows["window_index"].to_numpy(dtype=int))

    validation_fold = int(test_fold) % int(n_folds) + 1
    val_rows = repeat_rows[repeat_rows["test_fold"].astype(int) == validation_fold]
    if val_rows.empty:
        raise ValueError(f"No validation rows for repeat {repeat_index} fold {test_fold}")
    val_idx = np.sort(val_rows["window_index"].to_numpy(dtype=int))

    universe = np.sort(repeat_rows["window_index"].to_numpy(dtype=int))
    train_idx = np.sort(np.setdiff1d(universe, np.concatenate([test_idx, val_idx])))

    for role, arr in (("train", train_idx), ("validation", val_idx), ("test", test_idx)):
        if len(arr) == 0:
            raise ValueError(f"Empty {role} split in W repeat {repeat_index} fold {test_fold}")
    # Each window must be in exactly one role for this repeat.
    assert len(np.unique(np.concatenate([train_idx, val_idx, test_idx]))) == len(universe)
    return {"train": train_idx, "validation": val_idx, "test": test_idx}


def get_protocol_s_outer_subjects(s_outer: pd.DataFrame, groups: np.ndarray, outer_fold: int) -> dict:
    row = s_outer[s_outer["outer_fold"].astype(int) == int(outer_fold)]
    if len(row) != 1:
        raise ValueError(f"S outer manifest fold {outer_fold} has {len(row)} rows")
    test_subject = str(row.iloc[0]["outer_test_subject_id"])
    outer_train_subjects = np.sort(np.unique(groups[groups != test_subject]))
    return {"outer_train_subjects": outer_train_subjects, "outer_test_subject": test_subject}


def build_s_inner_splits(
    s_inner: pd.DataFrame, groups: np.ndarray, outer_fold: int, n_inner_folds: int = 5
):
    """Return [(inner_fold, train_idx, validation_idx)] covering all outer-training subjects."""
    rows_fold = s_inner[s_inner["outer_fold"].astype(int) == int(outer_fold)]
    if rows_fold.empty:
        raise ValueError(f"No inner manifest rows for outer fold {outer_fold}")
    test_subjects = set(rows_fold["outer_test_subject_id"].astype(str).to_numpy())
    if len(test_subjects) != 1:
        raise ValueError("Multiple outer test subjects in inner manifest")
    test_subject = test_subjects.pop()
    # Outer-training subjects are exactly those listed in the inner manifest.
    # (Using manifest rows keeps the partition check valid even when the groups
    # array carries extra training-only rows, e.g. multi-cohort training.)
    all_subjects = set(rows_fold["subject_id"].astype(str).to_numpy())

    splits = []
    seen_val = []
    for inner_fold in range(1, int(n_inner_folds) + 1):
        rows = rows_fold[rows_fold["inner_validation_fold"].astype(int) == int(inner_fold)]
        val_subjects = set(rows["subject_id"].astype(str).to_numpy())
        extra = val_subjects - all_subjects
        if extra:
            raise ValueError(f"Inner fold {inner_fold} validation includes non-training subjects: {extra}")
        if len(val_subjects) == 0:
            raise ValueError(f"Inner fold {inner_fold} has no validation subjects")
        train_subjects = np.sort(np.asarray(sorted(all_subjects - val_subjects)))
        val_arr = np.sort(np.asarray(sorted(val_subjects)))
        train_idx = np.sort(np.flatnonzero(np.isin(groups, train_subjects)))
        val_idx = np.sort(np.flatnonzero(np.isin(groups, val_arr)))
        splits.append((int(inner_fold), train_idx, val_idx))
        seen_val.append(set(val_subjects))
    # Verify each training subject is validation in exactly one inner fold.
    union = set().union(*seen_val)
    if union != all_subjects or sum(len(s) for s in seen_val) != len(all_subjects):
        raise ValueError("Inner validation folds do not partition training subjects")
    return splits


def outer_folds_list(s_outer: pd.DataFrame) -> list:
    return sorted(int(v) for v in s_outer["outer_fold"].astype(int).unique())
