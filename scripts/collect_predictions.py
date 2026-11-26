"""Collect saved predictions and attach labels in a separate scoring step."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pd_eeg.data import load_dataset
from pd_eeg.io import read_json
from pd_eeg.training.workflow import verify_files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/paper.json"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config, frames, datasets = read_json(args.config), [], {}
    for path in sorted(args.runs.rglob("predictions.csv")):
        verify_files(path.parent)
        frame = pd.read_csv(path, dtype={"subject_id": str})
        if frame.empty or len(frame.cohort.unique()) != 1:
            raise ValueError(f"Invalid cohort in {path}")
        cohort = str(frame.cohort.iloc[0])
        if cohort not in datasets:
            settings = config["cohorts"][cohort]
            datasets[cohort] = load_dataset(args.project_root / settings["path"],
                                           expected_sha256=settings.get("sha256"))
        dataset = datasets[cohort]
        indices = frame.window_id.to_numpy(dtype=np.int64)
        if len(indices) != len(np.unique(indices)) or np.any(indices < 0) or np.any(indices >= len(dataset.y)):
            raise ValueError(f"Invalid prediction window indices in {path}")
        if not np.array_equal(frame.subject_id.astype(str), dataset.groups[indices]):
            raise ValueError("Prediction subject-to-window mapping differs from the scoring dataset")
        truth = dataset.y[indices]
        if "true_label" in frame and not np.array_equal(frame.true_label.to_numpy(), truth):
            raise ValueError("Saved labels differ from the scoring dataset")
        frame["true_label"] = truth
        # Keep separate protocols, variants and source directions distinguishable.
        frames.append(frame)
    if not frames:
        parser.error("No predictions were found")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(args.output, index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
