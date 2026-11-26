"""Rebuild all manuscript figures from the supplied aggregate results."""

import argparse
import json
from pathlib import Path

from pd_eeg.analysis.plotting import render_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("figures"))
    parser.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"], choices=["pdf", "svg", "png"])
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty; choose a new destination")
    files = render_all(args.output, formats=args.formats, dpi=args.dpi)
    print(json.dumps({"output": str(args.output), "figure_files": len(files)}))


if __name__ == "__main__":
    main()
