"""Summarize saved predictions using the benchmark's subject-level estimands."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pd_eeg.analysis import analyze_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--reference", default="BE-MSNet")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty; choose a new destination")
    with args.predictions.open(encoding="utf-8-sig", newline="") as stream:
        result = analyze_predictions(csv.DictReader(stream), replicates=args.replicates,
                                     seed=args.seed, reference=args.reference)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rows = []
    for panel in result["panels"]:
        for model, spec in panel["models"].items():
            for metric, values in spec["metrics"].items():
                rows.append({"cohort": panel["cohort"], "source": panel["source"],
                             "experiment": panel["experiment"], "protocol": panel["protocol"], "model": model,
                             "metric": metric, "point": values["point"], "ci_low": values["ci95"][0], "ci_high": values["ci95"][1]})
    with (args.output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"panels": len(result["panels"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
