"""Compute the demographic-only LOSO sensitivity analysis from public metadata."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from pd_eeg.analysis.metrics import subject_metrics
from pd_eeg.io import write_json


def participant_key(value):
    return str(value).strip().casefold().removeprefix("sub-")


def fit_demographics(manifest, metadata, *, sex_column):
    people = metadata.copy()
    people.index = people.participant_id.map(participant_key)
    if not people.index.is_unique:
        raise ValueError("Metadata participant identifiers must be unique")
    subjects = manifest.outer_test_subject_id.map(participant_key)
    if not subjects.is_unique or not subjects.isin(people.index).all():
        raise ValueError("Missing or duplicated manifest participants")
    if not manifest.outer_fold.is_unique:
        raise ValueError("Outer folds must identify distinct test subjects")
    people = people.loc[subjects].reset_index(drop=True)
    age = pd.to_numeric(people.age, errors="raise").to_numpy(dtype=float)
    sex_code = people[sex_column].astype("string").str.strip().str.casefold()
    sex = sex_code.map({"m": "M", "male": "M", "f": "F", "female": "F"}).to_numpy(dtype=object)
    if np.any(pd.notna(sex_code).to_numpy() & pd.isna(sex)) or np.isinf(age).any():
        raise ValueError("Unexpected demographic metadata values")
    X = pd.DataFrame({"age": age, "sex": sex})
    y = manifest.true_label.to_numpy(dtype=int)
    columns = {"age": ("age", Pipeline([("impute", SimpleImputer(strategy="median")),
                                         ("scale", StandardScaler())]), ["age"]),
               "sex": ("sex", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                         ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), ["sex"])}
    summary = {}
    for name, features in (("age_only", ("age",)), ("sex_only", ("sex",)), ("age_sex", ("age", "sex"))):
        p = np.empty(len(y), dtype=float)
        with threadpool_limits(limits=1):
            for i in range(len(y)):
                train = np.arange(len(y)) != i
                pipe = Pipeline([("features", ColumnTransformer([columns[f] for f in features])),
                                 ("model", LogisticRegression(C=1., class_weight="balanced", solver="lbfgs",
                                                              max_iter=2000, tol=1e-8, random_state=11))])
                pipe.fit(X.iloc[train], y[train])
                p[i] = pipe.predict_proba(X.iloc[[i]])[0, 1]
        summary[name] = subject_metrics(y, p)
        rng = np.random.default_rng(20261109)
        neg, pos = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
        draws = [np.r_[rng.choice(neg, len(neg), replace=True),
                       rng.choice(pos, len(pos), replace=True)] for _ in range(2000)]
        metrics = [subject_metrics(y[i], p[i]) for i in draws]
        summary[name]["ci95"] = {key: np.percentile([row[key] for row in metrics], [2.5, 97.5]).tolist()
                                  for key in ("auc", "balanced_accuracy", "brier_score")}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--sex-column", choices=("sex", "gender"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = pd.read_csv(args.manifest, dtype={"outer_test_subject_id": str})
    metadata = pd.read_csv(args.metadata, sep="\t" if args.metadata.suffix == ".tsv" else ",",
                           dtype={"participant_id": str}, na_values=["n/a", "NA", "na", "nan"])
    write_json(args.output, fit_demographics(manifest.sort_values("outer_fold"), metadata,
                                            sex_column=args.sex_column))


if __name__ == "__main__":
    main()
