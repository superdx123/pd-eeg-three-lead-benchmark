"""List or execute fixed manuscript experiments."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pd_eeg.io import read_json
from pd_eeg.training.workflow import EXPERIMENTS, jobs, run_job


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/paper.json"))
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--cohort", choices=("ds002778", "ds003490"))
    parser.add_argument("--model", action="append")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--variant", choices=("full", "raw_only", "single_scale", "mean_only", "r18"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run", action="store_true", help="Execute selected jobs; default only lists them")
    parser.add_argument("--smoke", action="store_true", help="Two-epoch engineering check in a separate output directory")
    args = parser.parse_args(argv)
    config = read_json(args.config)
    chosen = list(jobs(args.experiment, config["cohorts"]))
    for key in ("cohort", "seed", "fold", "repeat", "variant"):
        value = getattr(args, key)
        if value is not None:
            chosen = [job for job in chosen if getattr(job, key) == value]
    if args.model:
        from pd_eeg.training.specs import model_name
        wanted = set(map(model_name, args.model))
        chosen = [job for job in chosen if job.model in wanted]
    if not chosen:
        parser.error("No jobs match the supplied filters")
    if not args.run:
        print(json.dumps({"jobs": [asdict(job) for job in chosen], "count": len(chosen)}, indent=2))
        return
    output = args.output / "smoke" if args.smoke else args.output
    for job in chosen:
        print(json.dumps({"job": asdict(job), "output": str(run_job(job, config,
              project_root=args.project_root, output=output, device=args.device, smoke=args.smoke))}))


if __name__ == "__main__":
    main()
