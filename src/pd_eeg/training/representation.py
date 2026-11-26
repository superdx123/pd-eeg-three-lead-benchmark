"""Fixed-budget PSD-AE-RBFNN and TDCNet representation/classifier pipelines."""

from __future__ import annotations

import numpy as np
import torch

from pd_eeg.analysis.metrics import aggregate_subjects
from pd_eeg.models import psd_ae, tdcnet
from .spectral import validate_inputs, validation_metrics
from .specs import model_name

RBF_LEARNING_RATES = (0.005, 0.05, 0.5)


def _device(value):
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return device


def _epochs(max_epochs):
    epochs = 50 if max_epochs is None else int(max_epochs)
    if epochs < 1 or epochs > 50:
        raise ValueError("Use 50 epochs for the reported protocol, or 1..50 for a shortened run.")
    return epochs


def _history(rows, *, phase, inner_fold, stage, learning_rate=None):
    result = []
    for epoch, value in enumerate(rows, 1):
        row = dict(value) if isinstance(value, dict) else {"loss": float(value)}
        result.append({**row, "epoch": epoch, "phase": phase, "inner_fold": inner_fold,
                       "stage": stage, "learning_rate": learning_rate})
    return result


def _fit_psd(X, y, groups, splits, final_train, prediction_X, seed, epochs, device):
    if not splits:
        raise ValueError("PSD-AE-RBFNN learning-rate selection needs inner splits.")
    features, _, _ = psd_ae.max_normalized_welch_psd_features(X, 256., ("C3", "Cz", "C4"))
    candidates = [{"learning_rate": rate, "valid": True} for rate in RBF_LEARNING_RATES]
    predictions = {rate: [] for rate in RBF_LEARNING_RATES}
    history = []
    for fold, train, validation in splits:
        encoder, losses = psd_ae.train_autoencoder(features[train], device, epochs, 64, seed, .05)
        history.extend(_history(losses, phase="selection", inner_fold=fold, stage="autoencoder"))
        train_latent = psd_ae.encode_features(encoder, features[train], device, 64)
        validation_latent = psd_ae.encode_features(encoder, features[validation], device, 64)
        centers, sigma = psd_ae.initialize_rbf_centers(train_latent, seed)
        for row in candidates:
            rate = row["learning_rate"]
            try:
                classifier, losses = psd_ae.train_rbfnn(train_latent, y[train], centers, sigma,
                    device, epochs, 64, seed, rate)
                p = psd_ae.infer_rbfnn_probabilities(classifier, validation_latent, device, 64)
                predictions[rate].append(aggregate_subjects(groups[validation], y[validation], p))
                history.extend(_history(losses, phase="selection", inner_fold=fold,
                                        stage="rbfnn", learning_rate=rate))
            except FloatingPointError as error:
                row["valid"] = False
                row.setdefault("failures", []).append({"inner_fold": fold, "error": str(error)})
    for row in candidates:
        if row["valid"]:
            row.update(validation_metrics(predictions[row["learning_rate"]]))
    valid = [row for row in candidates if row["valid"]]
    if not valid:
        raise RuntimeError("No finite PSD-AE-RBFNN learning-rate candidate remains.")
    selected = min(valid, key=lambda row: (-row["balanced_accuracy"], -row["auc"], row["learning_rate"]))
    encoder, losses = psd_ae.train_autoencoder(features[final_train], device, epochs, 64, seed, .05)
    history.extend(_history(losses, phase="final", inner_fold=None, stage="autoencoder"))
    latent = psd_ae.encode_features(encoder, features[final_train], device, 64)
    centers, sigma = psd_ae.initialize_rbf_centers(latent, seed)
    classifier, losses = psd_ae.train_rbfnn(latent, y[final_train], centers, sigma,
        device, epochs, 64, seed, selected["learning_rate"])
    history.extend(_history(losses, phase="final", inner_fold=None, stage="rbfnn",
                            learning_rate=selected["learning_rate"]))
    artifact = {"kind": "psd_ae_rbfnn", "autoencoder_state_dict": psd_ae.cpu_state_dict(encoder),
                "rbf_state_dict": psd_ae.cpu_state_dict(classifier), "initial_sigma": float(sigma)}
    return {"probabilities": predict_classifier(artifact, prediction_X, device=device),
            "estimator": artifact, "selected": {**selected, "epochs": epochs},
            "candidates": candidates, "history": history}


def _fit_tdc(name, X, y, groups, splits, final_train, prediction_X, seed, epochs, device):
    if not splits:
        raise ValueError("TDCNet SVM selection needs inner splits.")
    reference = name == "TDCNet-reference"
    history = []
    shared_model = None
    shared_embeddings = None
    if reference:
        shared_model, rows = tdcnet.train_tdcnet_representation(X[final_train], groups[final_train], device, epochs, seed)
        history.extend(_history(rows, phase="selection_and_final", inner_fold=None, stage="representation"))
        final_embeddings = tdcnet.encode_tdcnet_embeddings(shared_model, X[final_train], device)
        shared_embeddings = np.empty((len(X), final_embeddings.shape[1]), dtype=np.float32)
        shared_embeddings[final_train] = final_embeddings
    prediction_frames = [[] for _ in tdcnet.SVM_C_GRID]
    for fold, train, validation in splits:
        if reference:
            train_embeddings, validation_embeddings = shared_embeddings[train], shared_embeddings[validation]
        else:
            network, rows = tdcnet.train_tdcnet_representation(X[train], groups[train], device, epochs, seed)
            history.extend(_history(rows, phase="selection", inner_fold=fold, stage="representation"))
            train_embeddings = tdcnet.encode_tdcnet_embeddings(network, X[train], device)
            validation_embeddings = tdcnet.encode_tdcnet_embeddings(network, X[validation], device)
        for rank, C in enumerate(tdcnet.SVM_C_GRID):
            classifier = tdcnet.fit_rbf_svm(train_embeddings, y[train], C, seed)
            p = tdcnet.positive_class_probabilities(classifier, validation_embeddings)
            prediction_frames[rank].append(aggregate_subjects(groups[validation], y[validation], p))
    candidates = []
    for rank, C in enumerate(tdcnet.SVM_C_GRID):
        candidates.append({"C": "inf" if np.isinf(C) else float(C), "rank": rank,
                           **validation_metrics(prediction_frames[rank])})
    selected = min(candidates, key=lambda row: (-row["balanced_accuracy"], -row["auc"], row["rank"]))
    C = tdcnet.SVM_C_GRID[selected["rank"]]
    if reference:
        network, embeddings = shared_model, shared_embeddings[final_train]
    else:
        network, rows = tdcnet.train_tdcnet_representation(X[final_train], groups[final_train], device, epochs, seed)
        history.extend(_history(rows, phase="final", inner_fold=None, stage="representation"))
        embeddings = tdcnet.encode_tdcnet_embeddings(network, X[final_train], device)
    classifier = tdcnet.fit_rbf_svm(embeddings, y[final_train], C, seed)
    artifact = {"kind": "tdcnet", "model_config": dict(tdcnet.MODEL_CONFIG),
                "state_dict": tdcnet.cpu_state_dict(network), "classifier": classifier,
                "representation_scope": "outer_training_with_validation" if reference else "inner_training_only"}
    return {"probabilities": predict_classifier(artifact, prediction_X, device=device),
            "estimator": artifact, "selected": {**selected, "epochs": epochs},
            "candidates": candidates, "history": history}


def fit_classifier(name, X, y, groups, splits, final_train, prediction_X, seed=11,
                   cohort="ds002778", transfer=False, max_epochs=None, device="cpu"):
    """Fit the fixed representation pipeline; target data never select parameters."""
    name = model_name(name)
    if name not in {"PSD-AE-RBFNN", "TDCNet", "TDCNet-reference"}:
        raise ValueError(f"Not a representation classifier: {name}")
    X, y, groups, splits, final_train, prediction_X = validate_inputs(X, y, groups, splits, final_train, prediction_X)
    device, epochs = _device(device), _epochs(max_epochs)
    if name == "PSD-AE-RBFNN":
        return _fit_psd(X, y, groups, splits, final_train, prediction_X, seed, epochs, device)
    return _fit_tdc(name, X, y, groups, splits, final_train, prediction_X, seed, epochs, device)


def predict_classifier(artifact, X, *, device="cpu"):
    """Predict using supplied fitted state; no serialization or file reads occur."""
    artifact = artifact.get("estimator", artifact)
    device = _device(device)
    if artifact["kind"] == "psd_ae_rbfnn":
        features, _, _ = psd_ae.max_normalized_welch_psd_features(X, 256., ("C3", "Cz", "C4"))
        encoder = psd_ae.PSDAutoencoder().to(device)
        encoder.load_state_dict(artifact["autoencoder_state_dict"], strict=True)
        latent = psd_ae.encode_features(encoder, features, device, 64)
        centers = artifact["rbf_state_dict"]["centers"].detach().cpu().numpy()
        classifier = psd_ae.TrainableRBFNN(centers, artifact["initial_sigma"]).to(device)
        classifier.load_state_dict(artifact["rbf_state_dict"], strict=True)
        return psd_ae.infer_rbfnn_probabilities(classifier, latent, device, 64)
    if artifact["kind"] == "tdcnet":
        network = tdcnet.TDCContrastiveModel(artifact["model_config"]).to(device)
        network.load_state_dict(artifact["state_dict"], strict=True)
        embeddings = tdcnet.encode_tdcnet_embeddings(network, X, device)
        return tdcnet.positive_class_probabilities(artifact["classifier"], embeddings)
    raise ValueError("Unknown representation artifact kind.")
