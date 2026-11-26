"""Prepare study inputs and their subject/window cross-validation manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pd_eeg.data import (
    build_nested_loso_manifests, build_protocol_w_manifest, file_sha256,
    import_preprocessed, load_dataset, resolve_output_path,
    validate_s_manifests, validate_w_manifest,
)
from pd_eeg.data.dataset import PROCESSED_SHA256
from pd_eeg.data.eo import build_eo_dataset


def _manifests(dataset, output: Path, *, inner_seed_base: int) -> dict:
    outer, inner = build_nested_loso_manifests(dataset.y, dataset.groups, inner_seed_base=inner_seed_base)
    window = build_protocol_w_manifest(dataset.y, dataset.groups)
    validate_s_manifests(dataset.y, dataset.groups, outer, inner)
    validate_w_manifest(dataset.y, dataset.groups, window)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "s_outer.csv": outer, "s_inner.csv": inner, "w_windows.csv": window,
    }
    if any((output / name).exists() for name in paths):
        raise FileExistsError("A split manifest already exists in the output directory")
    for name, frame in paths.items():
        frame.to_csv(output / name, index=False, lineterminator="\n")
    return {name: file_sha256(output / name) for name in paths}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    imported = sub.add_parser("import", help="Select classifier channels from a prepared study NPZ")
    imported.add_argument("--cohort", choices=tuple(PROCESSED_SHA256), required=True)
    imported.add_argument("--source", type=Path, required=True)
    imported.add_argument("--output", type=Path, required=True)
    imported.add_argument("--expected-sha256", help="Defaults to the stored study input hash")
    raw = sub.add_parser("eo490", help="Build pure eyes-open OFF windows from ds003490 BIDS files")
    raw.add_argument("--bids-root", type=Path, required=True)
    raw.add_argument("--output", type=Path, default=Path("data/processed/ds003490.npz"))
    raw.add_argument("--selection-file", type=Path)
    splits = sub.add_parser("splits", help="Create deterministic S/W manifests for a prepared dataset")
    splits.add_argument("--dataset", type=Path, required=True)
    splits.add_argument("--output", type=Path, required=True)
    splits.add_argument("--cohort", choices=tuple(PROCESSED_SHA256), required=True)
    splits.add_argument("--inner-seed-base", type=int, help="Override the cohort's study split seed base")
    args = parser.parse_args(argv)
    destination = resolve_output_path(args.output)
    if args.command == "import":
        expected = args.expected_sha256 or PROCESSED_SHA256[args.cohort]
        import_preprocessed(args.source, destination, cohort=args.cohort, expected_sha256=expected)
        print(json.dumps({"dataset": str(args.output), "source_sha256": file_sha256(args.source),
                          "prepared_sha256": file_sha256(destination)}))
    elif args.command == "eo490":
        build_eo_dataset(args.bids_root, destination, selection_file=args.selection_file)
        print(json.dumps({"dataset": str(args.output), "prepared_sha256": file_sha256(destination)}))
    else:
        dataset = load_dataset(args.dataset, cohort=args.cohort)
        seed_base = args.inner_seed_base if args.inner_seed_base is not None else {"ds002778": 730000, "ds003490": 830000}[args.cohort]
        print(json.dumps({"manifests": _manifests(dataset, destination, inner_seed_base=seed_base)}))


if __name__ == "__main__":
    main()
