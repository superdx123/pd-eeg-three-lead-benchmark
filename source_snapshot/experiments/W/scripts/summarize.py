"""Close out all 125 native-validated W tasks, preserving both W estimators.

Default: readiness/hash checks only. --execute creates deterministic outputs once.
Primary: five-seed window-probability ensemble, subject means within each test
fold, then equal mean of the 25 fold metrics. Secondary: pool the five OOF test
folds within each repeat before subject scoring, then mean five repeat metrics.
No GPU, fitting, inference, threshold selection or bootstrap is performed.
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
from common import ROOT, inputs, protocol, sha, task_dir, verify_frozen, role_indices
SEEDS = (11,22,33,44,55)
REPEATS = FOLDS = (1,2,3,4,5)
MODEL = "DS28_BR_MS_AUG_LS_WD3"
REQUIRED_ARTIFACTS = {"checkpoint.pt","windows.csv","subjects.csv","selection.json",
                     "inner_predictions.csv","final_training_history.csv","environment.json"}
ALLOWED_ARTIFACTS = REQUIRED_ARTIFACTS | {"timing.json","started.json"}
PREDICTION_FILES = {
 "per_seed_fold_subject_predictions.csv":"per_seed_fold_subject_predictions",
 "per_seed_repeat_oof_subject_predictions.csv":"per_seed_repeat_oof_subject_predictions",
 "ensemble_fold_subject_predictions.csv":"ensemble_fold_subject_predictions",
 "ensemble_repeat_oof_subject_predictions.csv":"ensemble_repeat_oof_subject_predictions",
}
SUMMARY_FILES = set(PREDICTION_FILES) | {"summary.json","manifest.json"}


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

def csv_rows(path, columns):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None and len(reader.fieldnames) == len(columns)
                and set(reader.fieldnames) == set(columns), "Unexpected CSV schema: " + str(path))
        rows = list(reader)
    require(rows and all(set(row) == set(columns) and all(value is not None for value in row.values())
                         for row in rows), "Empty or malformed CSV")
    return rows

def csv_bytes(rows, columns):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: format(row[name], ".17g") if isinstance(row[name], float)
                         else row[name] for name in columns})
    return stream.getvalue().encode("utf-8")

def validated_task(task, bindings, p):
    seed, repeat, fold = task["seed"], task["repeat_index"], task["outer_fold"]
    directory = task_dir(seed, repeat, fold).resolve()
    require(directory.is_relative_to(ROOT.resolve()) and directory.is_dir(), "Unsafe/missing task")
    complete_path, validated_path = directory / "complete.json", directory / "validated.json"
    require(complete_path.is_file() and validated_path.is_file(),
            "Missing complete or validated receipt: " + task["task_id"])
    complete, validated = read_json(complete_path), read_json(validated_path)
    require(complete.get("status") == "TRAINED_PENDING_VALIDATION"
            and validated.get("status") == "PASS", "Unaccepted task: " + task["task_id"])
    for record in (complete, validated):
        require(record.get("seed") == seed and record.get("repeat_index") == repeat and record.get("fold") == fold, "Task receipt identity mismatch")
        for key in ("data_manifest_sha256", "code_manifest_sha256", "frozen_sha256", "protocol_sha256"):
            require(record.get(key) == bindings[key], "Task immutable binding mismatch: " + key)
    require(complete.get("config_name") == validated.get("model") == MODEL
            and complete.get("protocol") == validated.get("protocol") == "W_subject_overlap_window_cv"
            and complete.get("threshold") == validated.get("threshold") == 0.5,
            "Task model/threshold mismatch")
    require(validated.get("validator_sha256") == bindings["validator_sha256"], "Validator version mismatch")
    require(validated.get("complete_sha256") == sha(complete_path), "Validation completion binding mismatch")
    require(validated.get("native_cuda_validation") is True and validated.get("training_called") is False
            and validated.get("threshold_optimization_called") is False,
            "Validation did not cover the permitted inference-only native scope")
    require(validated.get("probability_replay_tolerance") == 1e-7
            and validated.get("n_parameters") == 67190, "Validation standard changed")
    require(validated.get("validation_fold") == fold % 5 + 1
            and validated.get("subject_overlap_is_intentional") is True
            and validated.get("window_roles_disjoint") is True,
            "Validation W-role semantics changed")
    require(0 <= float(validated["max_window_probability_error"]) <= 1e-7
            and 0 <= float(validated["max_subject_probability_error"]) <= 1e-7,
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


def task_grid():
    p, grid = protocol(), read_json(ROOT/"config/tasks.json")
    require(p["dataset_id"] == grid["dataset_id"] == "ds002778"
            and p["protocol"] == grid["protocol"] == "W_subject_overlap_window_cv",
            "Wrong W cohort/protocol")
    require(p["model_name"] == MODEL and p["threshold"] == .5, "Wrong model/threshold")
    require(tuple(p["seeds"]) == tuple(grid["seeds"]) == SEEDS, "Fixed W seeds changed")
    require(tuple(p["repeats"]) == tuple(grid["repeats"]) == REPEATS
            and tuple(p["folds"]) == tuple(grid["outer_folds"]) == FOLDS, "W repeat/fold grid changed")
    require(p["outer_tasks"] == grid["task_count"] == len(grid["tasks"]) == 125, "Exactly125 W tasks required")
    require(p["expected_parameters"] == 67190 and p["probability_replay_tolerance"] == 1e-7,
            "Frozen validation standard changed")
    seen, tasks = set(), []
    for row in grid["tasks"]:
        require(set(row) == {"seed","repeat_index","outer_fold","task_id"}, "Unknown W task fields")
        seed,repeat,fold = row["seed"],row["repeat_index"],row["outer_fold"]
        require(type(seed) is int and seed in SEEDS and type(repeat) is int and repeat in REPEATS
                and type(fold) is int and fold in FOLDS, "Unknown W task")
        key = (seed,repeat,fold)
        require(key not in seen and row["task_id"] == f"seed{seed}_repeat{repeat:02d}_fold{fold:02d}",
                "Duplicate or invalid W task identity")
        seen.add(key); tasks.append(dict(row))
    require(seen == {(s,r,f) for s in SEEDS for r in REPEATS for f in FOLDS}, "Incomplete W grid")
    return p, sorted(tasks,key=lambda row:(row["seed"],row["repeat_index"],row["outer_fold"]))


def reject_unknown_directories():
    outputs = ROOT/"outputs"
    require(outputs.is_dir(), "W outputs missing")
    require(all(d.name in {f"seed_{s}" for s in SEEDS}|{"summary"} and d.is_dir()
                and not d.is_symlink() for d in outputs.iterdir()), "Unknown W output entry")
    for seed in SEEDS:
        base = outputs/f"seed_{seed}"
        require(base.is_dir() and {d.name for d in base.iterdir()} == {f"repeat_{r:02d}" for r in REPEATS},
                "Missing/unknown W repeat")
        for repeat in REPEATS:
            rd = base/f"repeat_{repeat:02d}"
            require(rd.is_dir() and not rd.is_symlink()
                    and {d.name for d in rd.iterdir()} == {f"fold_{f:02d}" for f in FOLDS},
                    "Missing/unknown W fold")
            require(all(d.is_dir() and not d.is_symlink() for d in rd.iterdir()), "Invalid W fold directory")


def subject_means(scores, indices, groups, labels):
    grouped = {}
    for index in indices:
        index = int(index)
        require(index in scores, "Missing window score")
        grouped.setdefault(groups[index],[]).append(finite_probability(scores[index]))
    require(set(grouped) == set(labels) and len(grouped) == 31, "Incomplete31-subject W scoring set")
    return [{"subject_id":sid,"true_label":labels[sid],"n_windows":len(grouped[sid]),
             "prob_pd":math.fsum(grouped[sid])/len(grouped[sid]),
             "predicted_label":int(math.fsum(grouped[sid])/len(grouped[sid]) >= .5)}
            for sid in sorted(grouped)]


def collect_predictions(tasks):
    X,y,groups,manifest = inputs()
    del X
    groups = [str(value) for value in groups]
    truth = [int(value) for value in y]
    require(len(groups) == len(truth) == 3022, "W source dimensions changed")
    labels = {}
    for sid,label in zip(groups,truth):
        require(label in (0,1) and (sid not in labels or labels[sid] == label), "Inconsistent source label")
        labels[sid] = label
    require(len(labels) == 31 and list(labels.values()).count(0) == 16
            and list(labels.values()).count(1) == 15, "Wrong31-subject class composition")
    roles = {(r,f):[int(i) for i in role_indices(manifest,r,f)["test"]] for r in REPEATS for f in FOLDS}
    raw = {(seed,repeat):{} for seed in SEEDS for repeat in REPEATS}
    fold_subjects = []
    for task in tasks:
        seed,repeat,fold = task["seed"],task["repeat_index"],task["outer_fold"]
        directory = task_dir(seed,repeat,fold)
        windows = csv_rows(directory/"windows.csv", {
            "repeat_index","fold","subject_id","true_label","prob_pd","window_index",
            "selected_learning_rate","selected_epoch"})
        expected = set(roles[repeat,fold])
        local = {}
        for row in windows:
            index = integer(row["window_index"],"window_index")
            require(index in expected and index not in local and index not in raw[seed,repeat],
                    "Unknown/duplicated W test window")
            require(row["subject_id"] == groups[index] and integer(row["true_label"],"label") == truth[index]
                    and integer(row["repeat_index"],"repeat") == repeat and integer(row["fold"],"fold") == fold,
                    "Saved W window does not match frozen manifest")
            require(float(row["selected_learning_rate"]) in (.0003,.001)
                    and 1 <= integer(row["selected_epoch"],"epoch") <= 80, "W final selection outside protocol")
            local[index] = finite_probability(row["prob_pd"])
        require(set(local) == expected, "Missing W test windows")
        raw[seed,repeat].update(local)
        derived = {row["subject_id"]:row for row in subject_means(local,roles[repeat,fold],groups,labels)}
        saved = csv_rows(directory/"subjects.csv", {"subject_id","true_label","prob_pd","n_windows","predicted_label"})
        require(len(saved) == 31 and len({row["subject_id"] for row in saved}) == 31
                and {row["subject_id"] for row in saved} == set(labels), "W subject rows missing/duplicated")
        for row in saved:
            sid = row["subject_id"]; score = finite_probability(row["prob_pd"]); expected_row = derived[sid]
            require(integer(row["true_label"],"subject label") == labels[sid]
                    and integer(row["n_windows"],"subject windows") == expected_row["n_windows"]
                    and integer(row["predicted_label"],"subject prediction") == int(score >= .5),
                    "W subject labels/counts/decisions disagree")
            require(abs(score-expected_row["prob_pd"]) <= 1e-12, "Saved W subject score is not test-window mean")
            fold_subjects.append({"seed":seed,"repeat_index":repeat,"fold":fold,**expected_row,"prob_pd":score,
                                  "predicted_label":int(score >= .5)})
    require(all(set(scores) == set(range(3022)) for scores in raw.values()), "Incomplete per-seed/repeat3022 OOF windows")
    return {"window_probabilities":raw,"groups":groups,"labels":labels,"roles":roles,
            "per_seed_fold_subject_predictions":sorted(fold_subjects,key=lambda r:(r["seed"],r["repeat_index"],r["fold"],r["subject_id"]))}


def score_rows(rows):
    return metrics([row["true_label"] for row in rows],[row["prob_pd"] for row in rows])


def metric_mean(rows, units):
    require(len(rows) == units, "Wrong estimator unit count")
    return {**{name:math.fsum(row[name] for row in rows)/units
               for name in ("auc","balanced_accuracy","sensitivity","specificity")},
            "test_units":units,"subjects_per_test_unit":31,"hc_subjects":16,"pd_subjects":15}


def calculate_results(collected):
    raw,groups,labels,roles = (collected[key] for key in ("window_probabilities","groups","labels","roles"))
    fold_rows = collected["per_seed_fold_subject_predictions"]
    require(set(raw) == {(s,r) for s in SEEDS for r in REPEATS}, "Incomplete fixed seed/repeat score sets")
    require(all(set(raw[key]) == set(range(3022)) for key in raw), "Unknown/missing OOF windows")
    per_seed_fold_metrics = []
    for seed in SEEDS:
        for repeat in REPEATS:
            for fold in FOLDS:
                rows = [row for row in fold_rows if (row["seed"],row["repeat_index"],row["fold"]) == (seed,repeat,fold)]
                require(len(rows) == 31 and len({row["subject_id"] for row in rows}) == 31, "Fold subject coverage changed")
                per_seed_fold_metrics.append({"seed":seed,"repeat_index":repeat,"fold":fold,**score_rows(rows)})
    seed_repeat_rows, seed_repeat_metrics = [], []
    for seed in SEEDS:
        for repeat in REPEATS:
            rows = subject_means(raw[seed,repeat],range(3022),groups,labels)
            seed_repeat_rows.extend({"seed":seed,"repeat_index":repeat,**row} for row in rows)
            seed_repeat_metrics.append({"seed":seed,"repeat_index":repeat,**score_rows(rows)})
    ensemble_windows = {repeat:{index:math.fsum(raw[seed,repeat][index] for seed in SEEDS)/len(SEEDS)
                               for index in range(3022)} for repeat in REPEATS}
    ensemble_fold_rows, ensemble_fold_metrics = [], []
    ensemble_repeat_rows, ensemble_repeat_metrics = [], []
    for repeat in REPEATS:
        for fold in FOLDS:
            rows = subject_means(ensemble_windows[repeat],roles[repeat,fold],groups,labels)
            ensemble_fold_rows.extend({"repeat_index":repeat,"fold":fold,**row} for row in rows)
            ensemble_fold_metrics.append({"repeat_index":repeat,"fold":fold,**score_rows(rows)})
        rows = subject_means(ensemble_windows[repeat],range(3022),groups,labels)
        ensemble_repeat_rows.extend({"repeat_index":repeat,**row} for row in rows)
        ensemble_repeat_metrics.append({"repeat_index":repeat,**score_rows(rows)})
    return {
        "per_seed_fold_subject_predictions":fold_rows,
        "per_seed_repeat_oof_subject_predictions":seed_repeat_rows,
        "ensemble_fold_subject_predictions":ensemble_fold_rows,
        "ensemble_repeat_oof_subject_predictions":ensemble_repeat_rows,
        "per_seed_fold_metrics":per_seed_fold_metrics,"per_seed_repeat_oof_metrics":seed_repeat_metrics,
        "ensemble_fold_metrics":ensemble_fold_metrics,"ensemble_repeat_oof_metrics":ensemble_repeat_metrics,
        "per_seed_primary":[{"seed":seed,**metric_mean([row for row in per_seed_fold_metrics if row["seed"] == seed],25)}
                            for seed in SEEDS],
        "five_seed_probability_ensemble_primary":metric_mean(ensemble_fold_metrics,25),
        "per_seed_secondary_repeat_oof":[{"seed":seed,**metric_mean([row for row in seed_repeat_metrics if row["seed"] == seed],5)}
                                        for seed in SEEDS],
        "five_seed_probability_ensemble_secondary_repeat_oof":metric_mean(ensemble_repeat_metrics,5),
    }


def check_existing(directory,bindings,receipts):
    require(directory.is_dir() and not directory.is_symlink()
            and {path.name for path in directory.iterdir()} == SUMMARY_FILES,
            "Partial/unknown existing W summary; no overwrite/repair")
    manifest = read_json(directory/"manifest.json")
    require(manifest.get("status") == "PASS" and manifest.get("source_bindings") == bindings
            and manifest.get("task_receipt_set_sha256") == object_sha(receipts),
            "Existing W summary has changed inputs")
    expected = {(directory/name).relative_to(ROOT).as_posix() for name in SUMMARY_FILES-{"manifest.json"}}
    require(set(manifest.get("files",{})) == expected, "Existing W summary membership mismatch")
    for relative,digest in manifest["files"].items():
        member = ROOT/relative
        require(member.is_file() and not member.is_symlink() and sha(member) == digest, "W summary file SHA changed")
    summary = read_json(directory/"summary.json")
    require(summary.get("status") == "ALL_125_W_FOLDS_VALIDATED" and summary.get("source_bindings") == bindings
            and summary.get("validated_task_receipts") == receipts, "Existing W summary receipt bindings differ")
    return {"status":"EXISTING_IDENTICAL_W_SUMMARY_VERIFIED","tasks":125,"files":len(SUMMARY_FILES),
            "manifest_sha256":sha(directory/"manifest.json"),"written":False}


def summarize(execute=False):
    verify_frozen()
    p,tasks = task_grid()
    reject_unknown_directories()
    bindings = identities()
    receipts = [validated_task(task,bindings,p) for task in tasks]
    directory = ROOT/"outputs/summary"
    if directory.exists():
        result = check_existing(directory,bindings,receipts)
        verify_frozen(); require(identities() == bindings,"W frozen source changed")
        return result
    collected = collect_predictions(tasks)
    if not execute:
        verify_frozen(); require(identities() == bindings,"W source changed during readiness check")
        return {"status":"READY_ALL_125_W_TASKS_VALIDATED","tasks":125,"written":False}
    results = calculate_results(collected)
    summary = {
        "schema_version":1,"status":"ALL_125_W_FOLDS_VALIDATED",
        "study_id":"ds28_778_w_reproduction_v1","dataset_id":"ds002778","model":MODEL,
        "protocol":"W_subject_overlap_window_cv","threshold":.5,"seeds":list(SEEDS),
        "repeats":list(REPEATS),"folds":list(FOLDS),"tasks":125,"independent_subjects":31,
        "windows_per_repeat_per_seed":3022,
        "primary_estimator":"Align five seeds by repeat and original window_index; mean their probabilities; mean within subject in each test fold; equally average25 fold-level subject metrics.",
        "secondary_estimator":"Within repeat combine five test folds OOF windows; mean within subject; score31 subjects; equally average5 repeat metrics. Not the primary estimator.",
        "metrics_implementation":"pairwise AUC with half credit for ties; balanced accuracy at fixed probability>=0.5; float64-native Python math.fsum means",
        **{key:value for key,value in results.items() if key not in PREDICTION_FILES.values()},
        "historical_reference_only":{"primary_auc":.9915,"primary_balanced_accuracy":.9428,
            "role":"Rounded historical W reference only; retained unchanged separately.",
            "original_weights_recovered":False,"bitwise_historical_reproduction_claimed":False},
        "interpretation":"New full-checkpoint retraining, not recovery of original parameters; same31 people recur across W folds/repeats, which do not increase independent subject count.",
        "source_bindings":bindings,"validated_task_receipts":receipts,
        "training_called":False,"checkpoint_inference_called":False,
    }
    contents = {}
    for filename,key in PREDICTION_FILES.items():
        leading = (["seed"] if key.startswith("per_seed") else [])+["repeat_index"]
        if "fold_subject" in key: leading += ["fold"]
        columns = leading+["subject_id","true_label","prob_pd","n_windows","predicted_label"]
        contents[filename] = csv_bytes(results[key],columns)
    contents["summary.json"] = json_bytes(summary)
    manifest = {"schema_version":1,"status":"PASS","source_bindings":bindings,
        "task_receipt_set_sha256":object_sha(receipts),
        "files":{(directory/name).relative_to(ROOT).as_posix():hashlib.sha256(value).hexdigest()
                 for name,value in contents.items()}}
    contents["manifest.json"] = json_bytes(manifest)
    verify_frozen()
    require(identities() == bindings and [validated_task(task,bindings,p) for task in tasks] == receipts,
            "W artifacts changed during summary construction")
    directory.mkdir(exist_ok=False)
    for name,value in contents.items():
        with (directory/name).open("xb") as stream: stream.write(value)
    result = check_existing(directory,bindings,receipts)
    return {**result,"status":"W_SUMMARY_CREATED_AND_HASH_VERIFIED","written":True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true",help="Create summary only after125 formal native PASS tasks")
    args = parser.parse_args()
    print(json.dumps(summarize(args.execute),ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
