"""Collect the fixed 155 validated folds, without training or GPU inference.

The default invocation checks readiness and prediction coverage without writing.
Use --execute to create the four-file summary once. Existing summaries are only
hash-checked against the same immutable inputs, never overwritten or repaired.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
from common import ROOT, inputs, protocol, sha, task_dir, verify_frozen

SEEDS = (11, 22, 33, 44, 55)
MODEL = "DS28_BR_MS_AUG_LS_WD3"
REQUIRED_ARTIFACTS = {
    "checkpoint.pt", "windows.csv", "subjects.csv", "selection.json",
    "inner_predictions.csv", "final_training_history.csv", "environment.json",
}
ALLOWED_ARTIFACTS = REQUIRED_ARTIFACTS | {"timing.json", "started.json"}
SUMMARY_FILES = {
    "per_seed_subject_predictions.csv", "ensemble_subject_predictions.csv",
    "summary.json", "manifest.json",
}


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def object_sha(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def finite_probability(value):
    value = float(value)
    require(math.isfinite(value) and 0 <= value <= 1, "Invalid probability")
    return value


def integer(value, name):
    number = float(value)
    require(math.isfinite(number) and number.is_integer(), name + " must be integer")
    return int(number)


def metrics(labels, probabilities):
    """Exact pairwise-ranking AUC and fixed-0.5 subject balanced accuracy."""
    require(len(labels) == len(probabilities) and len(labels), "Invalid metric lengths")
    truth = [integer(value, "label") for value in labels]
    require(set(truth) == {0, 1}, "Both classes are required")
    scores = [finite_probability(value) for value in probabilities]
    positive = [score for label, score in zip(truth, scores) if label == 1]
    negative = [score for label, score in zip(truth, scores) if label == 0]
    twice_correct_pairs = sum(2 if p > n else 1 if p == n else 0
                              for p in positive for n in negative)
    tp = sum(p >= 0.5 for p in positive)
    tn = sum(n < 0.5 for n in negative)
    return {
        "auc": twice_correct_pairs / (2 * len(positive) * len(negative)),
        "balanced_accuracy": 0.5 * (tp / len(positive) + tn / len(negative)),
        "sensitivity": tp / len(positive), "specificity": tn / len(negative),
        "subjects": len(truth), "pd_subjects": len(positive), "hc_subjects": len(negative),
    }


def aligned_ensemble(per_seed):
    """Align on subject ID, not row position, before the five-seed mean."""
    require(set(per_seed) == set(SEEDS), "The complete fixed five-seed set is required")
    subjects = set(per_seed[SEEDS[0]])
    require(len(subjects) == 31, "Each seed must contain 31 subjects")
    require(all(set(per_seed[seed]) == subjects for seed in SEEDS), "Seed subject sets differ")
    result = []
    for sid in sorted(subjects):
        rows = [per_seed[seed][sid] for seed in SEEDS]
        require(len({row["true_label"] for row in rows}) == 1, "Seed labels disagree")
        require(len({row["n_windows"] for row in rows}) == 1, "Seed window counts disagree")
        scores = [finite_probability(row["prob_pd"]) for row in rows]
        probability = math.fsum(scores) / len(SEEDS)
        result.append({
            "subject_id": sid, "true_label": rows[0]["true_label"],
            "n_windows": rows[0]["n_windows"], "n_seeds": len(SEEDS),
            "prob_pd": probability, "predicted_label": int(probability >= 0.5),
            **{f"seed_{seed}_prob_pd": score for seed, score in zip(SEEDS, scores)},
        })
    return result


def identities():
    return {
        "data_manifest_sha256": sha(ROOT / "data_manifest.json"),
        "code_manifest_sha256": sha(ROOT / "code_manifest.json"),
        "frozen_sha256": sha(ROOT / "FROZEN.json"),
        "protocol_sha256": sha(ROOT / "config/protocol.json"),
        "tasks_sha256": sha(ROOT / "config/tasks.json"),
        "validator_sha256": sha(ROOT / "scripts/validate_fold.py"),
        "summarizer_sha256": sha(Path(__file__)),
    }


def task_grid():
    p, value = protocol(), read_json(ROOT / "config/tasks.json")
    require(p["dataset_id"] == value["dataset_id"] == "ds002778", "Wrong dataset")
    require(p["model_name"] == MODEL and p["threshold"] == 0.5, "Wrong model or threshold")
    require(tuple(p["seeds"]) == tuple(value["seeds"]) == SEEDS, "Frozen seed grid changed")
    require(value["task_count"] == len(value["tasks"]) == p["outer_tasks"] == 155,
            "Exactly 155 tasks are required")
    require(p["probability_replay_tolerance"] == 1e-7 and p["expected_parameters"] == 67190,
            "Replay tolerance or parameter count changed")
    folds = value["outer_folds"]
    require(len(folds) == len(set(folds)) == 31 and all(type(f) is int for f in folds),
            "Invalid frozen fold grid")
    result, seen = [], set()
    for task in value["tasks"]:
        require(set(task) == {"seed", "outer_fold", "task_id"}, "Unknown task fields")
        seed, fold = task["seed"], task["outer_fold"]
        require(type(seed) is int and seed in SEEDS and type(fold) is int and fold in folds,
                "Unknown task identity")
        require(task["task_id"] == f"seed{seed}_fold{fold:02d}" and (seed, fold) not in seen,
                "Duplicate or inconsistent task identity")
        seen.add((seed, fold))
        result.append(dict(task))
    require(seen == {(seed, fold) for seed in SEEDS for fold in folds}, "Incomplete task grid")
    return p, sorted(result, key=lambda row: (row["seed"], row["outer_fold"]))


def reject_unknown_directories(tasks):
    outputs = ROOT / "outputs"
    require(outputs.is_dir(), "Outputs directory is missing")
    expected_seed_dirs = {f"seed_{seed}" for seed in SEEDS}
    for member in outputs.iterdir():
        require(member.name in expected_seed_dirs | {"summary"} and member.is_dir()
                and not member.is_symlink(), "Unknown output entry: " + member.name)
    for seed in SEEDS:
        directory = outputs / f"seed_{seed}"
        require(directory.is_dir(), "Missing seed directory")
        expected = {f"fold_{task['outer_fold']:02d}" for task in tasks if task["seed"] == seed}
        actual = {member.name for member in directory.iterdir()}
        require(actual == expected, "Unknown or missing fold directory")
        require(all(member.is_dir() and not member.is_symlink() for member in directory.iterdir()),
                "Invalid fold output directory")


def validated_task(task, bindings, p):
    seed, fold = task["seed"], task["outer_fold"]
    directory = task_dir(seed, fold).resolve()
    require(directory.is_relative_to(ROOT.resolve()) and directory.is_dir(), "Unsafe/missing task")
    complete_path, validated_path = directory / "complete.json", directory / "validated.json"
    require(complete_path.is_file() and validated_path.is_file(),
            "Missing complete or validated receipt: " + task["task_id"])
    complete, validated = read_json(complete_path), read_json(validated_path)
    require(complete.get("status") == "TRAINED_PENDING_VALIDATION"
            and validated.get("status") == "PASS", "Unaccepted task: " + task["task_id"])
    for record in (complete, validated):
        require(record.get("seed") == seed and record.get("fold") == fold, "Task receipt identity mismatch")
        for key in ("data_manifest_sha256", "code_manifest_sha256", "frozen_sha256", "protocol_sha256"):
            require(record.get(key) == bindings[key], "Task immutable binding mismatch: " + key)
    require(complete.get("config_name") == validated.get("model") == MODEL
            and complete.get("threshold") == validated.get("threshold") == 0.5,
            "Task model/threshold mismatch")
    require(validated.get("validator_sha256") == bindings["validator_sha256"], "Validator version mismatch")
    require(validated.get("complete_sha256") == sha(complete_path), "Validation completion binding mismatch")
    require(validated.get("native_cuda_validation") is True and validated.get("training_called") is False
            and validated.get("threshold_optimization_called") is False,
            "Validation did not cover the permitted inference-only native scope")
    require(validated.get("probability_replay_tolerance") == 1e-7
            and validated.get("n_parameters") == 67190, "Validation standard changed")
    require(0 <= float(validated["max_window_probability_error"]) <= 1e-7
            and 0 <= float(validated["subject_probability_error"]) <= 1e-7,
            "Validation probability error exceeds the frozen standard")
    require(validated.get("environment_versions") == p["environment_versions"], "Validation environment mismatch")
    require(validated["training_environment"]["device_name"] == validated["validation_environment"]["gpu_name"],
            "Native training/validation GPU mismatch")
    members = complete.get("files")
    require(isinstance(members, dict) and members == validated.get("files"), "Artifact maps differ")
    names = set()
    for relative, digest in members.items():
        require(isinstance(relative, str) and "\\" not in relative and ":" not in relative,
                "Artifact names must be ROOT-relative POSIX paths")
        name = Path(relative)
        require(not name.is_absolute() and ".." not in name.parts, "Unsafe artifact member")
        member = (ROOT / name).resolve()
        require(member.parent == directory and member.name in ALLOWED_ARTIFACTS
                and member.is_file() and not member.is_symlink(), "Unknown/missing artifact member")
        require(member.name not in names and isinstance(digest, str)
                and re.fullmatch(r"[0-9a-f]{64}", digest), "Invalid artifact hash or duplicate member")
        require(sha(member) == digest, "Artifact SHA mismatch: " + relative)
        names.add(member.name)
    require(REQUIRED_ARTIFACTS <= names, "Required validated artifacts are missing")
    require({member.name for member in directory.iterdir()} == names | {"complete.json", "validated.json"},
            "Unknown/partial file in a completed task directory")
    checkpoint_sha = sha(directory / "checkpoint.pt")
    require(complete.get("checkpoint_sha256") == validated.get("checkpoint_sha256") == checkpoint_sha,
            "Checkpoint binding mismatch")
    return {**task, "complete_sha256": sha(complete_path), "validated_sha256": sha(validated_path),
            "checkpoint_sha256": checkpoint_sha, "artifact_map_sha256": object_sha(members)}


def csv_rows(path, columns):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None and len(reader.fieldnames) == len(columns)
                and set(reader.fieldnames) == set(columns), "Unexpected CSV schema: " + str(path))
        rows = list(reader)
    require(rows and all(set(row) == set(columns) and all(value is not None for value in row.values())
                         for row in rows), "Empty or malformed CSV")
    return rows


def collect_predictions(tasks):
    X, y, groups, outer, _inner = inputs()
    del X
    group_list, label_list = [str(g) for g in groups], [int(v) for v in y]
    require(len(group_list) == len(label_list) == 3022, "Wrong source window count")
    labels, counts = {}, {}
    for sid, label in zip(group_list, label_list):
        require(label in (0, 1) and (sid not in labels or labels[sid] == label), "Inconsistent source label")
        labels[sid] = label
        counts[sid] = counts.get(sid, 0) + 1
    require(len(labels) == 31 and list(labels.values()).count(0) == 16
            and list(labels.values()).count(1) == 15, "Wrong source subject/class count")
    test_subjects = {int(row.outer_fold): str(row.outer_test_subject_id)
                     for row in outer.itertuples(index=False)}
    require(len(test_subjects) == 31 and set(test_subjects.values()) == set(labels), "Outer manifest incomplete")
    require({task["outer_fold"] for task in tasks} == set(test_subjects), "Task/manifest fold mismatch")
    per_seed = {seed: {} for seed in SEEDS}
    covered = {seed: set() for seed in SEEDS}
    for task in tasks:
        seed, fold = task["seed"], task["outer_fold"]
        directory = task_dir(seed, fold)
        sid = test_subjects[fold]
        windows = csv_rows(directory / "windows.csv", {
            "outer_fold", "subject_id", "true_label", "prob_pd", "window_index",
            "selected_learning_rate", "selected_epoch",
        })
        local_indices, probabilities = set(), []
        for row in windows:
            index = integer(row["window_index"], "window_index")
            require(0 <= index < 3022 and index not in local_indices and index not in covered[seed],
                    "Duplicate/unknown source window")
            require(row["subject_id"] == group_list[index] == sid
                    and integer(row["true_label"], "window_label") == label_list[index]
                    and integer(row["outer_fold"], "outer_fold") == fold, "Window/manifest mapping mismatch")
            require(float(row["selected_learning_rate"]) in (3e-4, 1e-3)
                    and 1 <= integer(row["selected_epoch"], "selected_epoch") <= 80,
                    "Final training selection outside the frozen budget")
            local_indices.add(index)
            probabilities.append(finite_probability(row["prob_pd"]))
        require(len(windows) == counts[sid], "Incomplete held-out subject windows")
        covered[seed].update(local_indices)
        subjects = csv_rows(directory / "subjects.csv", {
            "subject_id", "true_label", "prob_pd", "n_windows", "predicted_label",
        })
        require(len(subjects) == 1 and subjects[0]["subject_id"] == sid, "Wrong held-out subject row")
        row = subjects[0]
        score = finite_probability(row["prob_pd"])
        require(integer(row["true_label"], "subject_label") == labels[sid]
                and integer(row["n_windows"], "n_windows") == counts[sid]
                and integer(row["predicted_label"], "predicted_label") == int(score >= 0.5),
                "Subject label/count/decision mismatch")
        require(abs(score - math.fsum(probabilities) / len(probabilities)) <= 1e-12,
                "Subject probability is not the saved window mean")
        require(sid not in per_seed[seed], "Subject duplicated within seed")
        per_seed[seed][sid] = {
            "seed": seed, "outer_fold": fold, "subject_id": sid,
            "true_label": labels[sid], "prob_pd": score,
            "n_windows": counts[sid], "predicted_label": int(score >= 0.5),
        }
    for seed in SEEDS:
        require(covered[seed] == set(range(3022)) and set(per_seed[seed]) == set(labels),
                "Per-seed OOF window/subject coverage incomplete")
    return per_seed


def csv_bytes(rows, columns):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: format(row[name], ".17g") if isinstance(row[name], float)
                         else row[name] for name in columns})
    return stream.getvalue().encode("utf-8")


def check_existing_summary(directory, bindings, receipts):
    require(directory.is_dir() and not directory.is_symlink(), "Existing summary is not a safe directory")
    require({path.name for path in directory.iterdir()} == SUMMARY_FILES,
            "Partial or unknown existing summary; refusing overwrite/repair")
    manifest = read_json(directory / "manifest.json")
    require(manifest.get("status") == "PASS" and manifest.get("source_bindings") == bindings
            and manifest.get("task_receipt_set_sha256") == object_sha(receipts),
            "Existing summary input bindings differ; refusing overwrite")
    expected_names = {path.relative_to(ROOT).as_posix() for path in directory.iterdir()
                      if path.name != "manifest.json"}
    require(isinstance(manifest.get("files"), dict) and set(manifest["files"]) == expected_names,
            "Existing summary manifest membership mismatch")
    for relative, digest in manifest["files"].items():
        member = ROOT / relative
        require(member.is_file() and not member.is_symlink() and sha(member) == digest,
                "Existing summary content/hash differs; refusing overwrite")
    summary = read_json(directory / "summary.json")
    require(summary.get("status") == "ALL_155_FOLDS_VALIDATED"
            and summary.get("source_bindings") == bindings
            and summary.get("validated_task_receipts") == receipts,
            "Existing summary does not match the validated inputs")
    return {"status": "EXISTING_IDENTICAL_SUMMARY_VERIFIED", "files": 4, "tasks": 155,
            "manifest_sha256": sha(directory / "manifest.json"), "written": False}


def summarize(execute=False):
    verify_frozen()
    p, tasks = task_grid()
    reject_unknown_directories(tasks)
    bindings = identities()
    receipts = [validated_task(task, bindings, p) for task in tasks]
    directory = ROOT / "outputs/summary"
    if directory.exists():
        result = check_existing_summary(directory, bindings, receipts)
        verify_frozen()
        require(identities() == bindings, "Frozen source changed during summary check")
        return result
    per_seed = collect_predictions(tasks)
    if not execute:
        verify_frozen()
        require(identities() == bindings, "Frozen source changed during readiness check")
        return {"status": "READY_ALL_155_VALIDATED", "tasks": 155, "seeds": list(SEEDS),
                "subjects_per_seed": 31, "windows_per_seed": 3022, "written": False}
    ensemble = aligned_ensemble(per_seed)
    ordered = [per_seed[seed][sid] for seed in SEEDS for sid in sorted(per_seed[seed])]
    seed_metrics = []
    for seed in SEEDS:
        rows = [per_seed[seed][sid] for sid in sorted(per_seed[seed])]
        seed_metrics.append({"seed": seed, "windows": 3022,
                             **metrics([row["true_label"] for row in rows],
                                       [row["prob_pd"] for row in rows])})
    summary = {
        "schema_version": 1, "status": "ALL_155_FOLDS_VALIDATED",
        "study_id": "ds28_778_s_reproduction_v1", "dataset_id": "ds002778",
        "model": MODEL, "protocol": "S_nested_subject_loso", "threshold": 0.5,
        "seeds": list(SEEDS), "outer_tasks": 155, "subjects": 31, "windows_per_seed": 3022,
        "subject_score": "saved arithmetic mean of held-out window probabilities, checked against windows",
        "ensemble_score": "subject-ID-aligned equal mean of all five fixed-seed probabilities using math.fsum",
        "metrics_implementation": "pairwise AUC (ties=0.5); bAcc=(PD sensitivity+HC specificity)/2 at >=0.5",
        "per_seed": seed_metrics,
        "five_seed_probability_ensemble": metrics([row["true_label"] for row in ensemble],
                                                   [row["prob_pd"] for row in ensemble]),
        "historical_reference_only": {
            "auc": 0.8458, "balanced_accuracy": 0.8063,
            "role": "Rounded historical S reference only, retained separately from this new retraining.",
            "original_weights_recovered": False, "bitwise_historical_reproduction_claimed": False,
        },
        "interpretation": "New fixed-protocol retraining with complete checkpoints and native independent acceptance; not recovered historical weights or a search for higher test scores.",
        "source_bindings": bindings, "validated_task_receipts": receipts,
        "training_called": False, "checkpoint_inference_called": False,
    }
    contents = {
        "per_seed_subject_predictions.csv": csv_bytes(ordered, [
            "seed", "outer_fold", "subject_id", "true_label", "prob_pd", "n_windows", "predicted_label"]),
        "ensemble_subject_predictions.csv": csv_bytes(ensemble, [
            "subject_id", "true_label", "n_windows", "n_seeds", "prob_pd", "predicted_label",
            *[f"seed_{seed}_prob_pd" for seed in SEEDS]]),
        "summary.json": json_bytes(summary),
    }
    manifest = {
        "schema_version": 1, "status": "PASS", "source_bindings": bindings,
        "task_receipt_set_sha256": object_sha(receipts),
        "files": {(directory / name).relative_to(ROOT).as_posix(): hashlib.sha256(body).hexdigest()
                  for name, body in contents.items()},
    }
    contents["manifest.json"] = json_bytes(manifest)
    verify_frozen()
    require(identities() == bindings
            and [validated_task(task, bindings, p) for task in tasks] == receipts,
            "Inputs/artifacts changed while preparing summary")
    directory.mkdir(exist_ok=False)
    for name, body in contents.items():
        with (directory / name).open("xb") as stream:
            stream.write(body)
    result = check_existing_summary(directory, bindings, receipts)
    return {**result, "status": "SUMMARY_CREATED_AND_HASH_VERIFIED", "written": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Create summary only after all 155 tasks pass")
    args = parser.parse_args()
    print(json.dumps(summarize(args.execute), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
