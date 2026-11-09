"""Analyze formal S/W outputs: per-seed metrics, 5-seed ensemble, TTA-free aggregation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from run.common import add_common_args, resolve_output_dir
from dsdata.metrics import binary_probability_metrics_full, aggregate_true_label_subject_probabilities


def _load_subject_predictions(base: Path, model: str, seed: int):
    d = base / f"{model}_seed{seed}"
    sp = d / "s_subject_predictions.csv"
    if not sp.exists():
        return None
    return pd.read_csv(sp, encoding="utf-8-sig")


def _metrics_frame(subj: pd.DataFrame) -> dict:
    metrics = binary_probability_metrics_full(subj["true_label"], subj["prob_pd"])
    return {"S_auc": metrics["auc"], "S_bacc": metrics["balanced_accuracy"],
            "S_acc": metrics["accuracy"], "S_mcc": metrics["mcc"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="formal", help="outputs/<run-name> root")
    parser.add_argument("--models", required=True, help="comma list of model names")
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--dataset", default="ds002778")
    parser.add_argument("--ens-models", default=None, help="comma list: also report cross-model ensemble (mean over these models' per-seed means)")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    base = Path("/home/jerry/dx/deepseek/outputs") / args.run_name

    report = {}
    # Collect per (model, seed) subject frames for later ensembling.
    frames: dict = {}
    for model in models:
        model_rows = []
        model_ens_subjects = None
        for seed in seeds:
            subj = _load_subject_predictions(base, model, seed)
            d = base / f"{model}_seed{seed}"
            row = {"seed": seed}
            if subj is not None:
                frames[(model, seed)] = subj
                metrics = _metrics_frame(subj)
                row.update(metrics)
            else:
                print(f"missing S predictions for {model} seed {seed}")
            wp = d / "w_metrics.csv"
            if wp.exists():
                w = pd.read_csv(wp, encoding="utf-8-sig")
                row["W_auc"] = float(w["window_auc"].mean())
                row["W_bacc"] = float(w["window_balanced_accuracy"].mean())
                if "subject_auc" in w.columns:
                    row["W_subj_auc"] = float(w["subject_auc"].mean())
                    row["W_subj_bacc"] = float(w["subject_balanced_accuracy"].mean())
            model_rows.append(row)
            # accumulate seed ensemble
            if model_ens_subjects is None:
                model_ens_subjects = subj[["subject_id", "true_label"]].copy()
                model_ens_subjects["prob_sum"] = subj["prob_pd"].to_numpy()
                model_ens_subjects["n"] = 1
            else:
                model_ens_subjects = model_ens_subjects.merge(subj[["subject_id", "prob_pd"]], on="subject_id", how="left")
                model_ens_subjects["prob_sum"] = model_ens_subjects["prob_sum"] + model_ens_subjects["prob_pd"].fillna(0)
                model_ens_subjects["n"] = model_ens_subjects["n"] + 1
                model_ens_subjects.drop(columns=["prob_pd"], inplace=True)
        df = pd.DataFrame(model_rows)
        if model_ens_subjects is not None:
            model_ens_subjects["prob_pd"] = model_ens_subjects["prob_sum"] / model_ens_subjects["n"]
            ens_metrics = _metrics_frame(model_ens_subjects)
            report[model] = {
                "per_seed": df.to_dict("records"),
                "seed_mean_S_auc": float(df["S_auc"].mean()) if len(df) else None,
                "seed_mean_S_bacc": float(df["S_bacc"].mean()) if len(df) else None,
                "seed_sd_S_auc": float(df["S_auc"].std(ddof=1)) if len(df) > 1 else 0.0,
                "seed_sd_S_bacc": float(df["S_bacc"].std(ddof=1)) if len(df) > 1 else 0.0,
                "ensemble_S_auc": ens_metrics["S_auc"],
                "ensemble_S_bacc": ens_metrics["S_bacc"],
                "W_auc_mean": float(df["W_auc"].mean()) if "W_auc" in df and len(df) else None,
                "W_bacc_mean": float(df["W_bacc"].mean()) if "W_bacc" in df and len(df) else None,
                "W_subj_auc_mean": float(df["W_subj_auc"].mean()) if "W_subj_auc" in df and len(df) else None,
                "W_subj_bacc_mean": float(df["W_subj_bacc"].mean()) if "W_subj_bacc" in df and len(df) else None,
            }
    # Cross-model ensemble: average per-seed mean subject probabilities.
    if args.ens_models:
        ens_models = [m.strip() for m in args.ens_models.split(",")]
        cross = None
        for model in ens_models:
            for seed in seeds:
                subj = frames.get((model, seed))
                if subj is None:
                    continue
                if cross is None:
                    cross = subj[["subject_id", "true_label"]].copy()
                    cross["prob_sum"] = subj["prob_pd"].to_numpy()
                    cross["n"] = 1
                else:
                    cross = cross.merge(subj[["subject_id", "prob_pd"]], on="subject_id", how="left")
                    cross["prob_sum"] = cross["prob_sum"] + cross["prob_pd"].fillna(0)
                    cross["n"] = cross["n"] + 1
                    cross.drop(columns=["prob_pd"], inplace=True)
        if cross is not None:
            cross["prob_pd"] = cross["prob_sum"] / cross["n"]
            cross_metrics = _metrics_frame(cross)
            report["CROSS_MODEL_ENSEMBLE_" + "+".join(ens_models)] = {
                "S_auc": cross_metrics["S_auc"], "S_bacc": cross_metrics["S_bacc"],
                "S_acc": cross_metrics["S_acc"], "S_mcc": cross_metrics["S_mcc"],
            }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
