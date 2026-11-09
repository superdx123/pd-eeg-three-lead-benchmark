"""Run one new full-budget ds002778 DS28 W repeat/fold.

Frozen two-LR selection uses one window-disjoint train/validation split. The
selected configuration is refit on train+validation, then predicts test windows.
This reproduces run_w_evaluation's math while preserving full model evidence.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

# This must precede torch import, including an indirect import from common.
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8"):
    raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import torch

from common import (
    ROOT, check_environment_versions, inputs, protocol, save, sha, task_dir,
    verify_frozen,
)

sys.path.insert(0, str(ROOT / "frozen_ds"))

CONFIG_NAME = "DS28_BR_MS_AUG_LS_WD3"
SEEDS = (11, 22, 33, 44, 55)
REPEATS = (1, 2, 3, 4, 5)
FOLDS = (1, 2, 3, 4, 5)
LEARNING_RATES = (3e-4, 1e-3)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_protocol(p: dict) -> None:
    """Fail closed rather than offer a shortened smoke-training mode."""
    expected = {
        "model_name": CONFIG_NAME,
        "threshold": 0.5,
        "max_epochs": 80,
        "patience": 12,
        "batch_size": 256,
        "selection_metric": "bacc",
        "threads_per_worker": 2,
        "expected_parameters": 67190,
        "probability_replay_tolerance": 1e-7,
    }
    for name, value in expected.items():
        require(p.get(name) == value, f"Frozen protocol mismatch: {name}")
    require(tuple(p["seeds"]) == SEEDS, "Frozen seed set mismatch")
    require(tuple(p["lr_candidates"]) == LEARNING_RATES, "Frozen LR grid mismatch")
    require(tuple(p["repeats"]) == REPEATS and tuple(p["folds"]) == FOLDS, "Frozen W grid mismatch")
    require(p["protocol"] == "W_subject_overlap_window_cv", "Not the frozen W protocol")


def safe_checkpoint_value(value, location: str = "checkpoint"):
    """Keep tensors unchanged; normalize only metadata to safe native types.

    In particular TorchVersion is a str subclass and NumPy scalars are not
    automatically safe under torch.load(weights_only=True). Unknown object
    types are rejected rather than allowed through pickle's global loader.
    """
    if isinstance(value, torch.Tensor):
        require(type(value) is torch.Tensor, f"Tensor subclass is not allowed at {location}")
        return value
    if value is None:
        return None
    if isinstance(value, np.generic):
        return safe_checkpoint_value(value.item(), location)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        require(math.isfinite(value), f"Nonfinite metadata at {location}")
        return float(value)
    if isinstance(value, dict):
        require(all(isinstance(key, str) for key in value),
                f"Non-string metadata key at {location}")
        return {str(key): safe_checkpoint_value(member, f"{location}.{key}")
                for key, member in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_checkpoint_value(member, f"{location}[{index}]")
                for index, member in enumerate(value)]
    raise RuntimeError(f"Unsupported checkpoint metadata type at {location}: {type(value).__name__}")


def environment_record(device: torch.device, versions: dict) -> dict:
    properties = torch.cuda.get_device_properties(device)
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[str(name)] = str(distribution.version)
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,pci.bus_id",
             "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        smi = f"UNAVAILABLE: {type(exc).__name__}"
    return {
        "recorded_utc": utc_now(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "environment_versions": dict(versions),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": dict(sorted(packages.items(), key=lambda row: row[0].lower())),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "torch_build_config": torch.__config__.show(),
        "device": str(device),
        "cuda_device_index": int(torch.cuda.current_device()),
        "device_name": str(properties.name),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "total_gpu_memory_bytes": int(properties.total_memory),
        "gpu_multiprocessor_count": int(properties.multi_processor_count),
        "nvidia_smi": smi,
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "environment": {name: os.environ.get(name) for name in (
            "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED",
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        )},
    }


def tagged_history(candidates: list[dict], histories: list[dict]) -> list[dict]:
    """Add provenance labels to returned histories, without rerunning selection."""
    result = []
    position = 0
    for candidate in candidates:
        stop = int(candidate["selection_stop_epoch"])
        block = histories[position:position + stop]
        require([int(row["epoch"]) for row in block] == list(range(1, stop + 1)),
                "Candidate history does not cover epochs through its stop")
        for row in block:
            result.append({**row, "learning_rate": float(candidate["learning_rate"])})
        position += stop
    require(position == len(histories), "Unassigned candidate history rows")
    return result


def run(seed: int, repeat_index: int, fold: int) -> None:
    require(seed in SEEDS and repeat_index in REPEATS and fold in FOLDS, "Task outside fixed W grid")
    verify_frozen()
    p = protocol()
    check_protocol(p)
    versions = {key: str(value) if value is not None else None
                for key, value in check_environment_versions().items()}
    from dsdata.protocol import get_protocol_w_role_indices
    from dsdata.metrics import aggregate_true_label_subject_probabilities
    from dsmodel.models import get_config
    from dstraining import engine, trainer

    X, y, groups, manifest = inputs()
    require(X.shape == (3022, 3, 512) and X.dtype == np.float32, "Unexpected classifier view")
    require(y.shape == groups.shape == (3022,) and y.dtype == np.int64, "Unexpected labels/groups")
    roles = get_protocol_w_role_indices(manifest, repeat_index, fold, n_folds=5)
    train, validation, test = (np.asarray(roles[k], dtype=np.int64) for k in ("train", "validation", "test"))
    for name, indices in (("train", train), ("validation", validation), ("test", test)):
        require(len(indices) and np.array_equal(indices, np.unique(indices)), "Invalid W indices: " + name)
        require(set(y[indices]) == {0, 1}, "W role lacks a class: " + name)
    require(not np.intersect1d(train, validation).size and not np.intersect1d(train, test).size
            and not np.intersect1d(validation, test).size, "W window roles overlap")
    require(np.array_equal(np.sort(np.r_[train, validation, test]), np.arange(3022)),
            "W roles do not partition every window")
    final_train = np.sort(np.concatenate([train, validation]))
    expected_validation_subjects = np.unique(groups[validation])
    require(len(expected_validation_subjects) > 1, "Empty W validation subject set")
    config = get_config(CONFIG_NAME)
    require(config.input_transform == "band_raw" and config.n_channels == 18
            and config.preprocess == "global_scalar" and config.augment == "light"
            and config.label_smoothing == .05 and config.params == {"wd": .001}, "Not complete frozen DS28")
    torch.set_num_threads(int(p["threads_per_worker"]))
    require(torch.cuda.is_available(), "CUDA required; no CPU training")
    device = trainer.configure_device(deterministic=True)
    require(device.type == "cuda" and torch.are_deterministic_algorithms_enabled(), "Native deterministic CUDA required")
    directory = task_dir(seed, repeat_index, fold)
    require(directory.resolve().is_relative_to(ROOT.resolve()) and not directory.exists(),
            "Existing/unsafe W task directory; no overwrite or automatic retry")
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    started_utc, started_clock = utc_now(), time.perf_counter()
    bindings = {
        "data_manifest_sha256": sha(ROOT / "data_manifest.json"),
        "code_manifest_sha256": sha(ROOT / "code_manifest.json"),
        "frozen_sha256": sha(ROOT / "FROZEN.json"),
        "protocol_sha256": sha(ROOT / "config" / "protocol.json"),
    }
    try:
        environment = safe_checkpoint_value(environment_record(device, versions), "environment")
        save(directory / "environment.json", environment)
        save(directory / "started.json", {"status": "RUNNING", "seed": seed, "repeat_index": repeat_index,
                                         "fold": fold, "started_utc": started_utc, **bindings})
        print(json.dumps({"event": "start_W_fold", "seed": seed, "repeat_index": repeat_index, "fold": fold,
                          "device_name": environment["device_name"]}), flush=True)
        torch.cuda.synchronize(device)
        selection_clock = time.perf_counter()
        # Exactly the single split passed by the frozen engine.run_w_evaluation.
        selected, candidates, histories, validation_rows = engine.run_candidate_selection(
            config, LEARNING_RATES, [(0, train, validation)], expected_validation_subjects,
            X, y, groups, device, seed, "W", repeat_index, fold,
            max_epochs=80, patience=12, batch_size=256, selection_metric="bacc", dataset_id="ds002778",
        )
        torch.cuda.synchronize(device)
        selection_seconds = time.perf_counter() - selection_clock
        histories = tagged_history(candidates, histories)
        require(tuple(float(c["learning_rate"]) for c in candidates) == LEARNING_RATES, "LR candidates changed")
        require(1 <= int(selected["selected_epoch"]) <= 80, "Invalid selected epoch")
        final_losses = []
        original_epoch = trainer.train_one_epoch

        def recorded_train_one_epoch(*args, **kwargs):
            value = original_epoch(*args, **kwargs)
            final_losses.append(float(value))
            return value

        final_clock = time.perf_counter()
        trainer.train_one_epoch = recorded_train_one_epoch
        try:
            probabilities, state, preprocess, attention = trainer.train_final_and_predict(
                config, selected["learning_rate"], X, y, groups, final_train, test,
                int(selected["selected_epoch"]), device, seed, 256, return_state=True,
                dataset_id="ds002778", protocol="W", repeat_index=repeat_index, outer_fold=fold,
            )
        finally:
            trainer.train_one_epoch = original_epoch
        torch.cuda.synchronize(device)
        final_seconds = time.perf_counter() - final_clock
        require(attention is None and state is not None and len(final_losses) == int(selected["selected_epoch"]),
                "Incomplete final refit state/history")
        require(np.isfinite(final_losses).all() and np.isfinite(probabilities).all()
                and len(probabilities) == len(test) and ((probabilities >= 0) & (probabilities <= 1)).all(),
                "Invalid final losses/probabilities")
        require(all(t.device.type == "cpu" for t in state.values()), "Non-CPU checkpoint state")
        count = sum(int(t.numel()) for t in state.values())
        require(count == p["expected_parameters"] == 67190, "Unexpected DS28 state size")
        checkpoint = {
            "schema_version": 1, "protocol": "W_subject_overlap_window_cv", "config_name": CONFIG_NAME,
            "model_config": asdict(config), "seed": seed, "repeat_index": repeat_index, "fold": fold,
            "validation_fold": fold % 5 + 1, "threshold": .5, "n_parameters": count,
            "selection_train_indices": torch.as_tensor(train, dtype=torch.int64),
            "validation_indices": torch.as_tensor(validation, dtype=torch.int64),
            "final_train_indices": torch.as_tensor(final_train, dtype=torch.int64),
            "train_indices": torch.as_tensor(final_train, dtype=torch.int64),
            "test_indices": torch.as_tensor(test, dtype=torch.int64),
            "inner_indices": [{"inner_fold": 0, "train_indices": torch.as_tensor(train, dtype=torch.int64),
                               "val_indices": torch.as_tensor(validation, dtype=torch.int64)}],
            "model_state_dict": state, "preprocess": preprocess, "selected": selected,
            "environment": environment, "environment_versions": dict(versions),
            "device_name": environment["device_name"], **bindings,
        }
        torch.save(safe_checkpoint_value(checkpoint), directory / "checkpoint.pt")
        windows = pd.DataFrame({
            "repeat_index": repeat_index, "fold": fold, "window_index": test,
            "subject_id": groups[test], "true_label": y[test], "prob_pd": probabilities,
            "selected_learning_rate": float(selected["learning_rate"]), "selected_epoch": int(selected["selected_epoch"]),
        })
        windows.to_csv(directory / "windows.csv", index=False)
        # Unlike the historical misnamed CSV, this file really contains one row per subject.
        subjects = aggregate_true_label_subject_probabilities(
            windows["subject_id"], windows["true_label"], windows["prob_pd"], threshold=.5,
        )
        subjects.to_csv(directory / "subjects.csv", index=False)
        pd.DataFrame(validation_rows).to_csv(directory / "inner_predictions.csv", index=False)
        pd.DataFrame({"epoch": range(1, len(final_losses) + 1), "training_loss": final_losses}).to_csv(
            directory / "final_training_history.csv", index=False)
        save(directory / "selection.json", {
            "selected": selected, "candidates": candidates, "history": histories, "final_losses": final_losses,
            "validation_scope": "one_window_disjoint_split_with_intentional_subject_overlap",
            "selection_training_windows": len(train), "validation_windows": len(validation),
            "final_training_windows": len(final_train), "test_windows": len(test),
        })
        save(directory / "timing.json", {
            "seed": seed, "repeat_index": repeat_index, "fold": fold, "started_utc": started_utc,
            "finished_utc": utc_now(), "selection_seconds": selection_seconds,
            "final_refit_and_prediction_seconds": final_seconds, "elapsed_seconds": time.perf_counter() - started_clock,
        })
        verify_frozen()
        for key, name in (("data_manifest_sha256", "data_manifest.json"), ("code_manifest_sha256", "code_manifest.json"),
                          ("frozen_sha256", "FROZEN.json"), ("protocol_sha256", "config/protocol.json")):
            require(sha(ROOT / name) == bindings[key], "Frozen binding changed: " + key)
        artifacts = sorted(directory.iterdir())
        require(all(item.is_file() for item in artifacts), "Unexpected task subdirectory")
        save(directory / "complete.json", {
            "status": "TRAINED_PENDING_VALIDATION", "protocol": "W_subject_overlap_window_cv",
            "seed": seed, "repeat_index": repeat_index, "fold": fold, "config_name": CONFIG_NAME, "threshold": .5,
            "checkpoint_sha256": sha(directory / "checkpoint.pt"),
            "files": {item.relative_to(ROOT).as_posix(): sha(item) for item in artifacts},
            "completed_utc": utc_now(), **bindings,
        })
        print(json.dumps({"event": "trained_pending_validation", "seed": seed, "repeat_index": repeat_index,
                          "fold": fold, "elapsed_seconds": time.perf_counter() - started_clock}), flush=True)
    except BaseException as exc:
        save(directory / "failure.json", {"status": "FAILED_NO_AUTOMATIC_RETRY", "seed": seed,
             "repeat_index": repeat_index, "fold": fold, "error_class": type(exc).__name__,
             "message": str(exc), "failed_utc": utc_now(), **bindings})
        raise

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--repeat", dest="repeat_index", type=int, choices=REPEATS, required=True)
    parser.add_argument("--fold", type=int, choices=FOLDS, required=True)
    args = parser.parse_args()
    run(args.seed, args.repeat_index, args.fold)


if __name__ == "__main__":
    main()

