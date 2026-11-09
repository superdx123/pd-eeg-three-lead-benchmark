"""Run exactly one new, full-budget ds002778 nested-S DS28 fold.

The preserved trainer and engine are imported unchanged. Existing output task
directories, including incomplete tasks, are never reused or overwritten.
Only a separate validator can promote TRAINED_PENDING_VALIDATION to acceptance.
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


def run(seed: int, fold: int) -> None:
    require(seed in SEEDS, "Seed is outside the fixed five-seed set")
    verify_frozen()
    p = protocol()
    check_protocol(p)
    # No fallback: a package/runtime mismatch must fail before CUDA setup or
    # exclusive task creation. str() also strips torch.torch_version.TorchVersion.
    versions = {str(key): str(value) if value is not None else None
                for key, value in check_environment_versions().items()}
    from dsdata.protocol import (
        build_s_inner_splits, get_protocol_s_outer_subjects, outer_folds_list,
    )
    from dsdata.metrics import aggregate_true_label_subject_probabilities
    from dsmodel.models import get_config
    from dstraining import engine, trainer

    X, y, groups, outer, inner = inputs()
    require(fold in outer_folds_list(outer), "Fold is absent from the frozen outer manifest")
    require(X.shape == (3022, 3, 512) and X.dtype == np.float32, "Unexpected input tensor")
    require(y.shape == (3022,) and y.dtype == np.int64, "Unexpected label vector")
    require(groups.shape == (3022,), "Unexpected subject vector")
    require(np.isfinite(X).all(), "Nonfinite source input")
    roles = get_protocol_s_outer_subjects(outer, groups, fold)
    train_indices = np.sort(np.flatnonzero(groups != roles["outer_test_subject"]))
    test_indices = np.sort(np.flatnonzero(groups == roles["outer_test_subject"]))
    expected_subjects = np.sort(np.unique(groups[train_indices]))
    require(len(expected_subjects) == 30 and len(test_indices) > 0, "Invalid outer LOSO split")
    require(set(y[train_indices]) == {0, 1}, "Outer training must contain both classes")
    splits = build_s_inner_splits(inner, groups, fold)
    require(len(splits) == 5, "Exactly five frozen inner folds are required")
    all_validation = []
    for _, train, validation in splits:
        require(len(train) > 0 and len(validation) > 0, "Empty inner partition")
        require(set(groups[train]).isdisjoint(set(groups[validation])), "Inner subject overlap")
        require(not np.intersect1d(test_indices, np.r_[train, validation]).size,
                "Outer test windows entered inner selection")
        require(np.array_equal(np.sort(np.r_[train, validation]), train_indices),
                "Inner fold does not partition the outer training windows")
        require(set(y[train]) == set(y[validation]) == {0, 1},
                "Each inner partition must contain both classes")
        all_validation.extend(validation.tolist())
    require(np.array_equal(np.sort(all_validation), train_indices),
            "Inner OOF coverage is not exactly the outer training windows")

    config = get_config(CONFIG_NAME)
    require(config.input_transform == "band_raw" and config.n_channels == 18,
            "Unexpected model input representation")
    require(config.preprocess == "global_scalar" and config.augment == "light",
            "Unexpected preprocessing or augmentation")
    require(config.label_smoothing == 0.05 and config.params == {"wd": 1e-3},
            "Unexpected DS28 loss or optimizer configuration")
    directory = task_dir(seed, fold)
    require(directory.resolve().is_relative_to(ROOT.resolve()), "Task output escapes ROOT")
    require(not directory.exists(), "Existing partial or complete task: refuse overwrite/retrain")
    torch.set_num_threads(int(p["threads_per_worker"]))
    require(torch.cuda.is_available(), "CUDA is required; CPU training is forbidden")
    device = trainer.configure_device(deterministic=True)
    require(device.type == "cuda", "CUDA-only fold runner")
    require(torch.are_deterministic_algorithms_enabled(), "Deterministic algorithms not enabled")

    environment = safe_checkpoint_value(environment_record(device, versions), "environment")
    manifest_hashes = {
        "data_manifest_sha256": sha(ROOT / "data_manifest.json"),
        "code_manifest_sha256": sha(ROOT / "code_manifest.json"),
        "frozen_sha256": sha(ROOT / "FROZEN.json"),
        "protocol_sha256": sha(ROOT / "config" / "protocol.json"),
    }
    directory.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation is also the same-task duplicate-launch guard.
    directory.mkdir(exist_ok=False)
    started_utc, started_clock = utc_now(), time.perf_counter()
    try:
        save(directory / "environment.json", environment)
        save(directory / "started.json", {
            "status": "RUNNING", "seed": seed, "fold": fold,
            "started_utc": started_utc, **manifest_hashes,
        })
        print(json.dumps({"event": "start_fold", "seed": seed, "fold": fold,
                          "started_utc": started_utc, "device_name": environment["device_name"]}),
              flush=True)
        torch.cuda.synchronize(device)
        selection_clock = time.perf_counter()
        selected, candidates, histories, validation_rows = engine.run_candidate_selection(
            config, LEARNING_RATES, splits, expected_subjects, X, y, groups,
            device, seed, "S", 0, fold,
            max_epochs=80, patience=12, batch_size=256,
            selection_metric="bacc", dataset_id="ds002778",
        )
        torch.cuda.synchronize(device)
        selection_seconds = time.perf_counter() - selection_clock
        histories = tagged_history(candidates, histories)
        require(tuple(float(row["learning_rate"]) for row in candidates) == LEARNING_RATES,
                "Candidate selection changed the learning-rate grid")
        require(1 <= int(selected["selected_epoch"]) <= 80, "Invalid selected epoch")

        # Observe the existing final loop's scalar returns only. No new random
        # draws, optimizer steps, losses, selection rules, or tensor operations.
        final_losses = []
        original_train_one_epoch = trainer.train_one_epoch

        def recorded_train_one_epoch(*args, **kwargs):
            loss = original_train_one_epoch(*args, **kwargs)
            final_losses.append(float(loss))
            return loss

        final_clock = time.perf_counter()
        trainer.train_one_epoch = recorded_train_one_epoch
        try:
            probabilities, state, preprocess, attention = trainer.train_final_and_predict(
                config, selected["learning_rate"], X, y, groups,
                train_indices, test_indices, int(selected["selected_epoch"]),
                device, seed, 256, return_state=True,
                dataset_id="ds002778", protocol="S", repeat_index=0, outer_fold=fold,
            )
        finally:
            trainer.train_one_epoch = original_train_one_epoch
        torch.cuda.synchronize(device)
        final_seconds = time.perf_counter() - final_clock
        require(attention is None and state is not None, "Unexpected final model output")
        require(len(final_losses) == int(selected["selected_epoch"]), "Final loss count mismatch")
        require(np.isfinite(final_losses).all(), "Nonfinite final training loss")
        require(len(probabilities) == len(test_indices) and np.isfinite(probabilities).all(),
                "Incomplete or nonfinite final probabilities")
        require(np.logical_and(probabilities >= 0, probabilities <= 1).all(),
                "Final probabilities are outside [0,1]")
        require(all(t.device.type == "cpu" for t in state.values()), "State must be CPU tensors")
        parameter_count = sum(int(t.numel()) for t in state.values())
        require(parameter_count == int(p["expected_parameters"]), "Unexpected DS28 state size")
        checkpoint = {
            "schema_version": 1, "config_name": CONFIG_NAME,
            "model_config": asdict(config), "seed": seed, "fold": fold,
            "threshold": 0.5, "n_parameters": parameter_count,
            "train_indices": torch.as_tensor(train_indices, dtype=torch.int64, device="cpu"),
            "test_indices": torch.as_tensor(test_indices, dtype=torch.int64, device="cpu"),
            "inner_indices": [{
                "inner_fold": int(inner_fold),
                "train_indices": torch.as_tensor(train, dtype=torch.int64, device="cpu"),
                "val_indices": torch.as_tensor(validation, dtype=torch.int64, device="cpu"),
            } for inner_fold, train, validation in splits],
            "model_state_dict": state, "preprocess": preprocess, "selected": selected,
            "environment": environment, "device_name": environment["device_name"],
            "environment_versions": dict(versions),
            **manifest_hashes,
        }
        torch.save(safe_checkpoint_value(checkpoint), directory / "checkpoint.pt")
        windows = pd.DataFrame({
            "outer_fold": fold, "subject_id": groups[test_indices],
            "true_label": y[test_indices], "prob_pd": probabilities,
            "window_index": test_indices,
            "selected_learning_rate": float(selected["learning_rate"]),
            "selected_epoch": int(selected["selected_epoch"]),
        })
        windows.to_csv(directory / "windows.csv", index=False)
        subjects = aggregate_true_label_subject_probabilities(
            windows["subject_id"], windows["true_label"], windows["prob_pd"], threshold=0.5,
        )
        subjects.to_csv(directory / "subjects.csv", index=False)
        pd.DataFrame(validation_rows).to_csv(directory / "inner_predictions.csv", index=False)
        pd.DataFrame({"epoch": range(1, len(final_losses) + 1),
                      "training_loss": final_losses}).to_csv(
                          directory / "final_training_history.csv", index=False)
        save(directory / "selection.json", {
            "selected": selected, "candidates": candidates, "history": histories,
            "final_losses": final_losses,
        })
        save(directory / "timing.json", {
            "seed": seed, "fold": fold, "started_utc": started_utc,
            "finished_utc": utc_now(), "selection_seconds": selection_seconds,
            "final_refit_and_prediction_seconds": final_seconds,
            "elapsed_seconds": time.perf_counter() - started_clock,
        })
        verify_frozen()
        for name, original_hash in manifest_hashes.items():
            manifest_path = {
                "data_manifest_sha256": ROOT / "data_manifest.json",
                "code_manifest_sha256": ROOT / "code_manifest.json",
                "frozen_sha256": ROOT / "FROZEN.json",
                "protocol_sha256": ROOT / "config" / "protocol.json",
            }[name]
            require(sha(manifest_path) == original_hash, f"Frozen binding changed: {name}")
        artifacts = sorted(directory.iterdir())
        require(all(item.is_file() for item in artifacts), "Unexpected task subdirectory")
        save(directory / "complete.json", {
            "status": "TRAINED_PENDING_VALIDATION", "seed": seed, "fold": fold,
            "config_name": CONFIG_NAME, "threshold": 0.5,
            "checkpoint_sha256": sha(directory / "checkpoint.pt"),
            "files": {item.relative_to(ROOT).as_posix(): sha(item) for item in artifacts},
            "completed_utc": utc_now(), **manifest_hashes,
        })
        print(json.dumps({"event": "trained_pending_validation", "seed": seed,
                          "fold": fold, "elapsed_seconds": time.perf_counter() - started_clock}),
              flush=True)
    except BaseException as exc:
        # Preserve partial evidence. A subsequent invocation refuses this task.
        save(directory / "failure.json", {
            "status": "FAILED_NO_AUTOMATIC_RETRY", "seed": seed, "fold": fold,
            "failed_utc": utc_now(), "error_class": type(exc).__name__,
            "message": str(exc), **manifest_hashes,
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--fold", type=int, required=True,
                        help="Actual outer_fold value from the frozen manifest")
    arguments = parser.parse_args()
    run(arguments.seed, arguments.fold)


if __name__ == "__main__":
    main()
