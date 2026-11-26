"""Compute paired intervals for three matched seeds of TDCNet and BE-MSNet."""
import argparse
import json
from pathlib import Path

from pd_eeg.analysis.tdc_comparison import compare_tdc
from pd_eeg.io import read_json, sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tdc-runs", type=Path, required=True, help="Strict TDCNet runs")
    parser.add_argument("--be-msnet-runs", type=Path, required=True)
    parser.add_argument("--reference-runs", type=Path, help="Optional validation-inclusive TDCNet runs")
    parser.add_argument("--cohort", choices=("ds002778", "ds003490"), required=True)
    parser.add_argument("--protocol", choices=("S", "W"), required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/paper.json"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True, help="New aggregate JSON file")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output file already exists")
    try:
        result = compare_tdc(
            args.tdc_runs, args.be_msnet_runs, read_json(args.config), cohort=args.cohort,
            protocol=args.protocol, project_root=args.project_root, reference_runs=args.reference_runs,
        )
        result["config_sha256"] = sha256(args.config)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(f"Saved matched {args.cohort} {args.protocol} comparison to {args.output}.")


if __name__ == "__main__":
    main()
