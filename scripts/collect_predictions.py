"""Collect predictions from a checked experiment plan and attach labels."""
from __future__ import annotations

import argparse
from pathlib import Path

from pd_eeg.io import read_json
from pd_eeg.training.collection import collect_predictions
from pd_eeg.training.workflow import EXPERIMENTS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/paper.json"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--cohort", choices=("ds002778", "ds003490"),
                        help="Limit the plan to one cohort (source cohort for transfer)")
    parser.add_argument("--model", action="append", help="Limit the plan to this model; repeatable")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Collect an incomplete run set for exploratory analysis")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output file already exists")
    try:
        frame, found, planned = collect_predictions(
            args.runs, read_json(args.config), args.experiment, project_root=args.project_root,
            cohort=args.cohort, models=args.model, allow_partial=args.allow_partial,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False, lineterminator="\n")
    status = "Complete" if found == planned else "Partial (exploratory)"
    print(f"{status}: {found}/{planned} jobs, {len(frame)} prediction rows.")


if __name__ == "__main__":
    main()
