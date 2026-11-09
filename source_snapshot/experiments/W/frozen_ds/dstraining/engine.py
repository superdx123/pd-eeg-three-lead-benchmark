"""Protocol W and S evaluation engines using frozen manifests.

Selection rule is fixed: validation subject-level balanced accuracy primary,
validation subject-level AUC tie-break, smaller lr, earlier epoch.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dsdata.metrics import (
    THRESHOLD,
    aggregate_subject_probabilities_attention,
    aggregate_true_label_subject_probabilities,
    binary_probability_metrics_full,
    subject_metrics_from_frame,
    window_metrics_from_frame,
)
from dsdata.protocol import (
    build_s_inner_splits,
    get_protocol_s_outer_subjects,
    get_protocol_w_role_indices,
    outer_folds_list,
)
from dstraining.trainer import (
    LEARNING_RATES,
    MAX_EPOCHS,
    METRIC_TIE_TOLERANCE,
    PATIENCE,
    balanced_class_weights,
    build_subject_discriminator,
    choose_candidate,
    configure_device,
    cpu_state_dict,
    fit_preprocess,
    infer_probabilities,
    infer_probabilities_weights,
    is_adv_subject,
    make_subject_labels,
    parameter_count,
    set_seed,
    train_final_and_predict,
    train_one_epoch,
)
from dsmodel.ssl import apply_pretrained, ensure_pretrained, ssl_mode


def _prepare_state(config, learning_rate, train_idx, val_idx, X, y, groups, device, seed,
                   dataset_id=None, protocol=None, repeat_index=None, outer_fold=None):
    transform, meta = fit_preprocess(X[train_idx], config.preprocess, config.input_transform, n_bands=config.n_bands,
                                    groups=groups[train_idx])
    X_train = transform(X[train_idx], groups[train_idx])
    X_val = transform(X[val_idx], groups[val_idx])
    data_std = float(np.std(X_train, dtype=np.float64) + 1e-8) if config.augment != "none" else 1.0
    set_seed(seed)
    from dsmodel.models import build_model
    model = build_model(config, device)
    if ssl_mode(config) and dataset_id is not None:
        apply_pretrained(model, config, dataset_id, protocol, repeat_index, outer_fold)
    discriminator = None
    subject_labels = None
    if is_adv_subject(config):
        subject_labels, n_subjects = make_subject_labels(groups, train_idx)
        subject_labels = subject_labels.to(device)
        discriminator = build_subject_discriminator(config, n_subjects, device)
    criterion = nn.CrossEntropyLoss(
        weight=balanced_class_weights(y[train_idx], device),
        label_smoothing=float(config.label_smoothing),
    )
    optimizer_params = list(model.parameters())
    if discriminator is not None:
        optimizer_params = optimizer_params + list(discriminator.parameters())
    optimizer = torch.optim.Adam(optimizer_params, lr=float(learning_rate), weight_decay=float(config.params.get("wd", 1e-4)))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return {
        "model": model, "criterion": criterion, "optimizer": optimizer, "generator": generator,
        "train_features": torch.as_tensor(X_train, device=device),
        "train_labels": torch.as_tensor(y[train_idx], dtype=torch.int64, device=device),
        "val_features": torch.as_tensor(X_val, device=device),
        "val_indices": np.asarray(val_idx, dtype=int),
        "train_groups": groups[train_idx].copy(),
        "data_std": data_std,
        "discriminator": discriminator,
        "subject_labels": subject_labels,
        "cpu_rng": torch.get_rng_state().clone(),
        "cuda_rng": torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None,
    }


def run_candidate_selection_sync(
    config, learning_rate, splits, expected_subjects, X, y, groups, device, seed,
    protocol, repeat_index, outer_fold, max_epochs=MAX_EPOCHS, patience=PATIENCE,
    batch_size=256, selection_metric="bacc", dataset_id=None,
):
    """Synchronized multi-split candidate selection with pooled OOF early stopping."""
    expected = {str(v) for v in expected_subjects}
    states = []
    for inner_fold, train_idx, val_idx in splits:
        state = _prepare_state(config, learning_rate, train_idx, val_idx, X, y, groups, device, seed,
                                dataset_id=dataset_id, protocol=protocol,
                                repeat_index=repeat_index, outer_fold=outer_fold)
        state["inner_fold"] = int(inner_fold)
        states.append(state)

    history = []
    validation_rows = []
    best_epoch = 0
    best_bacc = -np.inf
    best_auc = -np.inf
    patience_counter = 0
    stop_epoch = int(max_epochs)

    try:
        for epoch in range(1, int(max_epochs) + 1):
            epoch_losses = []
            epoch_subject_rows = []
            for state in states:
                torch.set_rng_state(state["cpu_rng"])
                if state["cuda_rng"] is not None:
                    torch.cuda.set_rng_state(state["cuda_rng"], device)
                epoch_gen = torch.Generator(device="cpu").manual_seed(seed * 100003 + epoch)
                loss_value = train_one_epoch(
                    state["model"], state["optimizer"], state["criterion"],
                    state["train_features"], state["train_labels"], batch_size,
                    epoch_gen, config, state["data_std"],
                    subject_labels=state.get("subject_labels"), discriminator=state.get("discriminator"),
                    train_groups=state.get("train_groups"),
                )
                state["cpu_rng"] = torch.get_rng_state().clone()
                if state["cuda_rng"] is not None:
                    state["cuda_rng"] = torch.cuda.get_rng_state(device).clone()
                idx = state["val_indices"]
                if bool(config.params.get("subject_attn", False)):
                    _p, pd_logits, weights = infer_probabilities_weights(state["model"], state["val_features"], batch_size)
                    subjects = aggregate_subject_probabilities_attention(
                        groups[idx], y[idx], pd_logits, weights,
                        threshold=float(config.params.get("threshold", THRESHOLD)),
                    )
                else:
                    probabilities = infer_probabilities(state["model"], state["val_features"], batch_size, tta=int(config.params.get("tta", 0)), config=config)
                    subjects = aggregate_true_label_subject_probabilities(
                        groups[idx], y[idx], probabilities,
                        threshold=float(config.params.get("threshold", THRESHOLD)),
                        agg=str(config.params.get("subject_agg", "mean")),
                    )
                epoch_losses.append(float(loss_value))
                for row in subjects.itertuples(index=False):
                    epoch_subject_rows.append({
                        "inner_fold": int(state["inner_fold"]), "epoch": epoch,
                        "subject_id": str(row.subject_id), "true_label": int(row.true_label),
                        "prob_pd": float(row.prob_pd), "n_windows": int(row.n_windows),
                        "predicted_label": int(row.predicted_label),
                    })
            pooled = pd.DataFrame(epoch_subject_rows)
            if set(pooled["subject_id"].astype(str)) != expected:
                raise RuntimeError(f"Pooled validation subjects incomplete at epoch {epoch}")
            if pooled["subject_id"].astype(str).duplicated().any():
                raise RuntimeError("Duplicate subject in pooled validation")
            metrics = binary_probability_metrics_full(pooled["true_label"], pooled["prob_pd"], threshold=THRESHOLD)
            bacc = float(metrics["balanced_accuracy"])
            auc = float(metrics["auc"])
            if selection_metric == "auc":
                improved = auc > best_auc + METRIC_TIE_TOLERANCE
                if abs(auc - best_auc) <= METRIC_TIE_TOLERANCE and bacc > best_bacc + METRIC_TIE_TOLERANCE:
                    improved = True
            else:
                improved = bacc > best_bacc + METRIC_TIE_TOLERANCE
                if abs(bacc - best_bacc) <= METRIC_TIE_TOLERANCE and auc > best_auc + METRIC_TIE_TOLERANCE:
                    improved = True
            if improved:
                best_epoch = epoch
                best_bacc = bacc
                best_auc = auc
                patience_counter = 0
            else:
                patience_counter += 1
            history.append({
                "epoch": epoch, "mean_training_loss": float(np.mean(epoch_losses)),
                "validation_subject_balanced_accuracy": bacc, "validation_subject_auc": auc,
                "improved": bool(improved), "patience_counter": int(patience_counter),
            })
            validation_rows.extend(epoch_subject_rows)
            if patience_counter >= int(patience):
                stop_epoch = epoch
                break
    finally:
        for state in states:
            del state["model"], state["optimizer"], state["criterion"]
            del state["train_features"], state["train_labels"], state["val_features"]
            if state.get("discriminator") is not None:
                del state["discriminator"], state["subject_labels"]
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if best_epoch < 1:
        raise RuntimeError("Epoch selection failed")
    for row in history:
        row["selection_stopped_here"] = bool(int(row["epoch"]) == stop_epoch)
    candidate = {
        "learning_rate": float(learning_rate), "selected_epoch": int(best_epoch),
        "best_validation_subject_balanced_accuracy": float(best_bacc),
        "best_validation_subject_auc": float(best_auc), "selection_stop_epoch": int(stop_epoch),
    }
    retained = [row for row in validation_rows if int(row["epoch"]) <= stop_epoch]
    return candidate, history, retained


def run_candidate_selection(
    config, lr_candidates, splits, expected_subjects, X, y, groups, device, seed,
    protocol, repeat_index, outer_fold, max_epochs=MAX_EPOCHS, patience=PATIENCE, batch_size=256,
    selection_metric="bacc", dataset_id=None,
):
    """Loop over lr candidates and apply the fixed selection rule."""
    if ssl_mode(config) and dataset_id is not None:
        pretrain_idx = np.unique(np.concatenate([np.asarray(ti, dtype=int) for _f, ti, _v in splits]))
        ensure_pretrained(config, dataset_id, X, y, groups, pretrain_idx, device, seed,
                          protocol, repeat_index, outer_fold, batch_size)
    candidates = []
    histories_all = []
    val_rows_all = []
    for lr in lr_candidates:
        candidate, histories, val_rows = run_candidate_selection_sync(
            config, float(lr), splits, expected_subjects, X, y, groups, device, seed,
            protocol, repeat_index, outer_fold, max_epochs, patience, batch_size,
            selection_metric, dataset_id,
        )
        candidates.append(candidate)
        histories_all.extend(histories)
        for row in val_rows:
            tagged = dict(row)
            tagged["learning_rate"] = float(lr)
            val_rows_all.append(tagged)
    selected = choose_candidate(candidates, metric=selection_metric)
    return selected, candidates, histories_all, val_rows_all


def _calibration_threshold_from_validation(val_rows, learning_rate, selected_epoch):
    """Select bAcc-optimal threshold on pooled inner-OOF validation subject predictions
    at the selected epoch for the selected lr candidate (no outer-test data involved)."""
    rows = [
        r for r in val_rows
        if math.isclose(float(r.get("learning_rate", -1.0)), float(learning_rate), rel_tol=0.0, abs_tol=1e-12)
        and int(r["epoch"]) == int(selected_epoch)
    ]
    if not rows:
        return THRESHOLD
    frame = pd.DataFrame(rows)
    subjects = aggregate_true_label_subject_probabilities(
        frame["subject_id"], frame["true_label"], frame["prob_pd"], threshold=THRESHOLD
    )
    y = subjects["true_label"].to_numpy()
    p = subjects["prob_pd"].to_numpy()
    best_t, best_bacc = THRESHOLD, -np.inf
    for t in np.arange(0.30, 0.71, 0.01):
        pred = (p >= t).astype(int)
        tp = float(((pred == 1) & (y == 1)).sum())
        tn = float(((pred == 0) & (y == 0)).sum())
        fn = float(((pred == 0) & (y == 1)).sum())
        fp = float(((pred == 1) & (y == 0)).sum())
        bacc = 0.5 * (tp / max(tp + fn, 1e-12) + tn / max(tn + fp, 1e-12))
        if bacc > best_bacc + 1e-12:
            best_bacc, best_t = bacc, t
    return float(best_t)


def _final_predict_lr_ensemble(config, candidates, X, y, groups, train_idx, test_idx,
                               device, seed, batch_size, dataset_id=None,
                               protocol=None, repeat_index=None, outer_fold=None):
    """Average final predictions across lr candidates (each at its own epoch).

    Default: equal average over all candidates. If params['lr_ensemble_tol'] is
    set, only candidates whose validation subject bAcc is within `tol` of the
    best are averaged (equal weights), which avoids dragging in clearly worse
    candidates (fixes the DS8_ENS 31-fold regression).
    """
    tol = float(config.params.get("lr_ensemble_tol", 0.0) or 0.0)
    if tol > 0.0:
        best = max(float(c["best_validation_subject_balanced_accuracy"]) for c in candidates)
        candidates = [c for c in candidates
                      if best - float(c["best_validation_subject_balanced_accuracy"]) <= tol]
        if not candidates:
            raise RuntimeError("lr_ensemble_tol filtered out all candidates")
    probs = []
    for cand in candidates:
        p, _, _, _ = train_final_and_predict(
            config, float(cand["learning_rate"]), X, y, groups, train_idx, test_idx,
            int(cand["selected_epoch"]), device, seed, batch_size, return_state=False,
            dataset_id=dataset_id, protocol=protocol, repeat_index=repeat_index,
            outer_fold=outer_fold,
        )
        probs.append(p)
    return np.mean(np.stack(probs, axis=0), axis=0)

def run_w_evaluation(
    config, dataset, w_manifest, device, seed, output_dir=None,
    repeats=(1, 2, 3, 4, 5), folds=(1, 2, 3, 4, 5),
    max_epochs=MAX_EPOCHS, patience=PATIENCE, batch_size=256,
    lr_candidates=LEARNING_RATES, selection_metric="bacc", calibrate=False,
):
    X, y, groups = dataset.X, dataset.y, dataset.groups
    result_rows = []
    subject_rows = []
    window_rows = []
    summary = {"model": config.name, "protocol": "W", "seed": int(seed)}
    for repeat_index in repeats:
        for test_fold in folds:
            roles = get_protocol_w_role_indices(w_manifest, repeat_index, test_fold)
            val_subjects = np.unique(groups[roles["validation"]])
            selected, candidates, histories, val_rows = run_candidate_selection(
                config, lr_candidates, [(0, roles["train"], roles["validation"])],
                val_subjects, X, y, groups, device, seed, "W", repeat_index, test_fold,
                max_epochs, patience, batch_size, selection_metric,
                dataset_id=dataset.dataset_id,
            )
            cal_threshold = float(config.params.get("threshold", THRESHOLD))
            if calibrate:
                cal_threshold = _calibration_threshold_from_validation(
                    val_rows, selected["learning_rate"], selected["selected_epoch"])
            final_train = np.sort(np.concatenate([roles["train"], roles["validation"]]))
            if config.params.get("lr_ensemble"):
                probabilities = _final_predict_lr_ensemble(
                    config, candidates, X, y, groups, final_train, roles["test"],
                    device, seed, batch_size,
                    dataset_id=dataset.dataset_id, protocol="W",
                    repeat_index=repeat_index, outer_fold=test_fold,
                )
                state, meta, attn_outputs = None, {}, None
            else:
                probabilities, state, meta, attn_outputs = train_final_and_predict(
                    config, selected["learning_rate"], X, y, groups, final_train, roles["test"],
                    selected["selected_epoch"], device, seed, batch_size, return_state=True,
                    dataset_id=dataset.dataset_id, protocol="W",
                    repeat_index=repeat_index, outer_fold=test_fold,
                )
            frame = pd.DataFrame({
                "window_index": roles["test"].astype(int),
                "subject_id": groups[roles["test"]],
                "true_label": y[roles["test"]].astype(int),
                "prob_pd": probabilities,
            })
            if attn_outputs is not None:
                frame["window_pd_logit"] = attn_outputs[0]
                frame["window_weight"] = attn_outputs[1]
            eval_threshold = float(config.params.get("threshold", THRESHOLD))
            window_metrics = binary_probability_metrics_full(frame["true_label"], frame["prob_pd"], threshold=eval_threshold)
            if attn_outputs is not None:
                subjects_frame = aggregate_subject_probabilities_attention(
                    frame["subject_id"], frame["true_label"], frame["window_pd_logit"],
                    frame["window_weight"], threshold=eval_threshold,
                )
            else:
                subjects_frame = aggregate_true_label_subject_probabilities(
                    frame["subject_id"], frame["true_label"], frame["prob_pd"],
                    threshold=eval_threshold, agg=str(config.params.get("subject_agg", "mean")),
                )
            subject_metrics = binary_probability_metrics_full(subjects_frame["true_label"], subjects_frame["prob_pd"], threshold=eval_threshold)
            if calibrate:
                cal_pred = (probabilities >= cal_threshold).astype(int)
                frame = frame.assign(calibrated_threshold=cal_threshold,
                                     cal_predicted_label=cal_pred)
                cal_subj = aggregate_true_label_subject_probabilities(
                    frame["subject_id"], frame["true_label"], frame["prob_pd"],
                    threshold=cal_threshold,
                )
                cal_metrics = binary_probability_metrics_full(
                    cal_subj["true_label"], cal_subj["prob_pd"], threshold=cal_threshold,
                    require_both_classes=False,
                )
            row = {
                "repeat_index": int(repeat_index), "fold": int(test_fold),
                "selected_learning_rate": float(selected["learning_rate"]),
                "selected_epoch": int(selected["selected_epoch"]),
            }
            if calibrate:
                row["calibrated_threshold"] = float(cal_threshold)
                row["calibrated_subject_bacc"] = float(cal_metrics["balanced_accuracy"])
            row.update({f"window_{k}": v for k, v in window_metrics.items()})
            row.update({f"subject_{k}": v for k, v in subject_metrics.items()})
            result_rows.append(row)
            subject_rows.append(pd.DataFrame({
                "repeat_index": int(repeat_index), "fold": int(test_fold),
                **frame.to_dict("list"),
            }))
            window_rows.append(frame.assign(repeat_index=repeat_index, fold=test_fold))
    metrics = pd.DataFrame(result_rows)
    summary.update({
        "window_auc_mean": float(metrics["window_auc"].mean()),
        "window_bacc_mean": float(metrics["window_balanced_accuracy"].mean()),
        "window_auc_sd": float(metrics["window_auc"].std(ddof=1)) if len(metrics) > 1 else 0.0,
        "window_bacc_sd": float(metrics["window_balanced_accuracy"].std(ddof=1)) if len(metrics) > 1 else 0.0,
        "subject_auc_mean": float(metrics["subject_auc"].mean()),
        "subject_bacc_mean": float(metrics["subject_balanced_accuracy"].mean()),
        "n_folds": int(len(metrics)),
    })
    return summary, metrics, subject_rows, window_rows


def run_s_evaluation(
    config, dataset, s_outer, s_inner, device, seed, output_dir=None,
    outer_folds=None, max_epochs=MAX_EPOCHS, patience=PATIENCE, batch_size=256,
    return_details=False, lr_candidates=LEARNING_RATES, selection_metric="bacc",
    calibrate=False,
):
    X, y, groups = dataset.X, dataset.y, dataset.groups
    extra_idx = getattr(dataset, "extra_train_idx", None)
    if outer_folds is None:
        outer_folds = outer_folds_list(s_outer)
    subject_frames = []
    window_frames = []
    candidate_rows = []
    fold_rows = []
    for outer_fold in outer_folds:
        roles = get_protocol_s_outer_subjects(s_outer, groups, outer_fold)
        outer_train_idx = np.sort(np.flatnonzero(np.isin(groups, roles["outer_train_subjects"])))
        outer_test_idx = np.sort(np.flatnonzero(np.isin(groups, roles["outer_test_subject"])))
        if extra_idx is not None and len(extra_idx):
            outer_train_idx = np.union1d(outer_train_idx, np.asarray(extra_idx, dtype=int))
        splits = build_s_inner_splits(s_inner, groups, outer_fold)
        if extra_idx is not None and len(extra_idx):
            extra_arr = np.asarray(extra_idx, dtype=int)
            splits = [(fold, np.union1d(np.asarray(ti, dtype=int), extra_arr), np.asarray(vi, dtype=int))
                      for fold, ti, vi in splits]
        # Expected validation subjects = outer-train subjects listed in the inner
        # manifest (excludes any extra training-only rows such as multi-cohort data).
        manifest_train = s_inner[s_inner["outer_fold"].astype(int) == int(outer_fold)]
        expected = np.sort(np.unique(manifest_train["subject_id"].astype(str).to_numpy()))
        selected, candidates, histories, val_rows = run_candidate_selection(
            config, lr_candidates, splits, expected,
            X, y, groups, device, seed, "S", 0, outer_fold, max_epochs, patience, batch_size,
            selection_metric, dataset_id=dataset.dataset_id,
        )
        cal_threshold = THRESHOLD
        if calibrate:
            cal_threshold = _calibration_threshold_from_validation(
                val_rows, selected["learning_rate"], selected["selected_epoch"])
        if config.params.get("lr_ensemble"):
            probabilities = _final_predict_lr_ensemble(
                config, candidates, X, y, groups, outer_train_idx, outer_test_idx,
                device, seed, batch_size,
                dataset_id=dataset.dataset_id, protocol="S",
                repeat_index=0, outer_fold=outer_fold,
            )
            state, meta, attn_outputs = None, {}, None
        else:
            probabilities, state, meta, attn_outputs = train_final_and_predict(
                config, selected["learning_rate"], X, y, groups, outer_train_idx, outer_test_idx,
                selected["selected_epoch"], device, seed, batch_size, return_state=True,
                dataset_id=dataset.dataset_id, protocol="S",
                repeat_index=0, outer_fold=outer_fold,
            )
        frame = pd.DataFrame({
            "outer_fold": int(outer_fold),
            "subject_id": groups[outer_test_idx],
            "true_label": y[outer_test_idx].astype(int),
            "prob_pd": probabilities,
            "window_index": outer_test_idx.astype(int),
            "selected_learning_rate": float(selected["learning_rate"]),
            "selected_epoch": int(selected["selected_epoch"]),
        })
        if attn_outputs is not None:
            frame["window_pd_logit"] = attn_outputs[0]
            frame["window_weight"] = attn_outputs[1]
        if calibrate:
            frame = frame.assign(calibrated_threshold=cal_threshold,
                                 cal_predicted_label=(probabilities >= cal_threshold).astype(int))
        window_frames.append(frame)
        candidate_rows.append(selected | {"outer_fold": int(outer_fold)})
        fold_row = {
            "outer_fold": int(outer_fold), "n_test_windows": int(len(frame)),
            "selected_learning_rate": float(selected["learning_rate"]),
            "selected_epoch": int(selected["selected_epoch"]),
            "test_subject": str(roles["outer_test_subject"]),
        }
        if calibrate:
            fold_row["calibrated_threshold"] = float(cal_threshold)
        fold_rows.append(fold_row)
    all_windows = pd.concat(window_frames, ignore_index=True)
    eval_threshold = float(config.params.get("threshold", THRESHOLD))
    if bool(config.params.get("subject_attn", False)):
        subjects = aggregate_subject_probabilities_attention(
            all_windows["subject_id"], all_windows["true_label"], all_windows["window_pd_logit"],
            all_windows["window_weight"], threshold=eval_threshold,
        )
    else:
        subjects = aggregate_true_label_subject_probabilities(
            all_windows["subject_id"], all_windows["true_label"], all_windows["prob_pd"], threshold=eval_threshold,
            agg=str(config.params.get("subject_agg", "mean")),
        )
    subject_metrics = binary_probability_metrics_full(subjects["true_label"], subjects["prob_pd"], threshold=eval_threshold,
                                                       require_both_classes=False)
    summary = {"model": config.name, "protocol": "S", "seed": int(seed),
               "n_outer_folds": int(len(outer_folds))}
    summary.update({f"subject_{k}": v for k, v in subject_metrics.items()})
    if calibrate:
        cal_subjects = all_windows.groupby("subject_id", sort=True, as_index=False).agg(
            true_label=("true_label", "first"),
            prob_pd=("prob_pd", "mean"),
            cal_predicted_label=("cal_predicted_label", "first"),
            n_windows=("prob_pd", "size"),
        ).reset_index(drop=True)
        ycal = cal_subjects["true_label"].to_numpy()
        pcal = cal_subjects["cal_predicted_label"].to_numpy()
        tp = float(((pcal == 1) & (ycal == 1)).sum())
        tn = float(((pcal == 0) & (ycal == 0)).sum())
        fn = float(((pcal == 0) & (ycal == 1)).sum())
        fp = float(((pcal == 1) & (ycal == 0)).sum())
        cal_bacc = 0.5 * (tp / max(tp + fn, 1e-12) + tn / max(tn + fp, 1e-12))
        cal_acc = float((pcal == ycal).mean())
        summary["calibrated_subject_balanced_accuracy"] = float(cal_bacc)
        summary["calibrated_subject_accuracy"] = cal_acc
        summary["calibrated_mean_threshold"] = float(all_windows["calibrated_threshold"].mean())
    if return_details:
        return summary, subjects, all_windows, pd.DataFrame(fold_rows)
    return summary


