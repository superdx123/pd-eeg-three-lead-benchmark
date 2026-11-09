"""Formal W and S evaluation for a frozen model config + seed.

Saves per-fold artifacts and summary. Supports --only-w / --only-s.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from run.common import add_common_args, resolve_output_dir, save_json
from dsdata.data_io import load_dataset, load_manifests
from dsdata.protocol import outer_folds_list
from dsmodel.models import build_model, get_config, trainable_parameter_count
from dstraining.trainer import configure_device, set_seed
from dstraining.engine import run_s_evaluation, run_w_evaluation


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--run-name", default="formal", help="subdirectory under outputs/")
    parser.add_argument("--only-w", action="store_true")
    parser.add_argument("--only-s", action="store_true")
    parser.add_argument("--s-outer-folds", default=None, help="comma list to run a subset of S folds (development use)")
    parser.add_argument("--lr-candidates", default=None, help="comma list of learning rates (default 3e-4,1e-3)")
    parser.add_argument("--select-metric", default="bacc", choices=["bacc", "auc"],
                        help="inner candidate selection metric (default bacc)")
    parser.add_argument("--calibrate", action="store_true",
                        help="per-fold validation-calibrated threshold (supplementary)")
    args = parser.parse_args()
    if args.lr_candidates:
        lr_candidates = tuple(float(v) for v in args.lr_candidates.split(","))
    else:
        lr_candidates = (3e-4, 1e-3)

    config = get_config(args.model)
    device = configure_device(deterministic=not args.no_deterministic)
    set_seed(args.seed)
    dataset = load_dataset(args.dataset)
    w_manifest, s_outer, s_inner, audit = load_manifests(dataset)

    out_dir = resolve_output_dir(args, f"{args.run_name}/{config.name}_seed{args.seed}")
    t_start = time.time()
    run_meta = {
        "model": config.name, "dataset": dataset.dataset_id, "seed": args.seed,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "max_epochs": args.max_epochs, "patience": args.patience, "batch_size": args.batch_size,
        "learning_rates": list(lr_candidates), "threshold": 0.5,
    }
    save_json(out_dir / "run_meta.json", run_meta)

    summary = {}
    if not args.only_s:
        print(f"[{config.name}] starting W ({dataset.dataset_id}, seed {args.seed})")
        w_summary, w_metrics, w_subjects, w_windows = run_w_evaluation(
            config, dataset, w_manifest, device, args.seed,
            max_epochs=args.max_epochs, patience=args.patience, batch_size=args.batch_size,
            lr_candidates=lr_candidates, selection_metric=args.select_metric,
            calibrate=args.calibrate,
        )
        w_metrics.to_csv(out_dir / "w_metrics.csv", index=False, encoding="utf-8-sig")
        pd.concat(w_subjects, ignore_index=True).to_csv(out_dir / "w_subject_predictions.csv", index=False, encoding="utf-8-sig")
        pd.concat(w_windows, ignore_index=True).to_csv(out_dir / "w_window_predictions.csv", index=False, encoding="utf-8-sig")
        save_json(out_dir / "w_summary.json", w_summary)
        summary["W"] = w_summary
        print("W:", {k: v for k, v in w_summary.items() if k.startswith(("window_", "subject_"))})

    if not args.only_w:
        print(f"[{config.name}] starting S ({dataset.dataset_id}, seed {args.seed})")
        s_folds = None
        if args.s_outer_folds:
            s_folds = [int(v) for v in args.s_outer_folds.split(",")]
        s_summary, s_subjects, s_windows, s_folds_df = run_s_evaluation(
            config, dataset, s_outer, s_inner, device, args.seed,
            outer_folds=s_folds, max_epochs=args.max_epochs, patience=args.patience,
            batch_size=args.batch_size, return_details=True, lr_candidates=lr_candidates,
            selection_metric=args.select_metric, calibrate=args.calibrate,
        )
        s_subjects.to_csv(out_dir / "s_subject_predictions.csv", index=False, encoding="utf-8-sig")
        s_windows.to_csv(out_dir / "s_window_predictions.csv", index=False, encoding="utf-8-sig")
        s_folds_df.to_csv(out_dir / "s_fold_metrics.csv", index=False, encoding="utf-8-sig")
        save_json(out_dir / "s_summary.json", s_summary)
        summary["S"] = s_summary
        print("S:", {k: v for k, v in s_summary.items() if k.startswith("subject_")})

    run_meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    run_meta["elapsed_seconds"] = round(time.time() - t_start, 1)
    save_json(out_dir / "run_meta.json", run_meta)
    save_json(out_dir / "summary.json", summary)
    print(f"DONE {config.name} seed {args.seed} in {run_meta['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    main()
