"""Matched three-seed comparison of saved TDCNet and BE-MSNet runs."""
from .metrics import analyze_predictions, normalize_row
from pd_eeg.training.collection import collect_predictions
from pd_eeg.training.specs import MATCHED_SEEDS

BOOTSTRAP_SEED = 20260930
BOOTSTRAP_DRAWS = 2000
ANALYSIS = "strict-tdc-matched"
STRICT_NAME = "TDCNet (strict inner fitting)"


def _aligned_window_rows(frame, experiment, model, cohort, protocol):
    rows = [normalize_row(row) for row in frame.to_dict(orient="records")]
    if {row["seed"] for row in rows} != set(MATCHED_SEEDS):
        raise ValueError(f"{model} must supply exactly seeds {MATCHED_SEEDS}")
    keys = {}
    for row in rows:
        settings = (row["experiment"], row["model"], row["cohort"], row["protocol"],
                    row["source"], row["variant"])
        if settings != (experiment, model, cohort, protocol, cohort, "full"):
            raise ValueError(f"Prediction settings do not match the {model} comparison arm")
        key = (row["repeat"], row["fold"], row["seed"], row["window_id"], row["subject_id"])
        if not row["window_id"] or key in keys:
            raise ValueError(f"Missing or duplicated test-window key for {model}")
        keys[key] = row["true_label"]
        row["experiment"] = ANALYSIS
    return rows, keys


def compare_tdc(tdc_runs, be_msnet_runs, config, *, cohort, protocol,
                project_root=".", reference_runs=None):
    """Score complete checked run sets on the same windows and training seeds."""
    if protocol not in ("S", "W"):
        raise ValueError("The TDCNet comparison requires protocol S or W")
    if cohort not in ("ds002778", "ds003490") or cohort not in config["cohorts"]:
        raise ValueError("Select one of the two benchmark cohorts")
    arms = [
        ("strict-tdc", "TDCNet", STRICT_NAME, tdc_runs),
        (f"benchmark-{protocol.lower()}", "BE-MSNet", "BE-MSNet", be_msnet_runs),
    ]
    if reference_runs is not None:
        arms.append(("tdc-reference", "TDCNet-reference", "TDCNet†", reference_runs))

    results_rows, run_sets = [], {}
    expected_keys = None
    for experiment, model, name, path in arms:
        frame, count, _ = collect_predictions(
            path, config, experiment, project_root=project_root, cohort=cohort,
            models=(model,), protocol=protocol, seeds=MATCHED_SEEDS,
        )
        rows, keys = _aligned_window_rows(frame, experiment, name, cohort, protocol)
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise ValueError(f"Test windows, subjects or labels do not align for {name}")
        results_rows.extend(rows)
        run_sets[name] = {"experiment": experiment, "jobs": count, "seeds": list(MATCHED_SEEDS)}

    result = analyze_predictions(
        results_rows, replicates=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED, reference=STRICT_NAME,
        ensemble_windows=True,
    )
    return {"analysis": ANALYSIS, "run_sets": run_sets, **result}
