"""Regenerate saved test probabilities from checkpoints and compare them."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pd_eeg.io import read_json
from pd_eeg.training.workflow import validate_job


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/paper.json"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trust-sklearn", action="store_true", help="Only for checkpoints you created or independently trust")
    args = parser.parse_args(argv)
    config = read_json(args.config)
    records = sorted(args.runs.rglob("run.json"))
    if not records:
        parser.error("No run.json files were found")
    for path in records:
        report = validate_job(path.parent, config, project_root=args.project_root,
                              device=args.device, trust_sklearn=args.trust_sklearn)
        print(json.dumps({"run": str(path.parent), **report}))


if __name__ == "__main__":
    main()
