"""Supervised neural model selection with training-only preprocessing."""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import random

import numpy as np
import pandas as pd
import torch
from torch import nn

from pd_eeg.analysis.metrics import aggregate_subjects, binary_metrics
from pd_eeg.models import build_model, prepare_input
from .specs import BRAINDDECODE_MODELS, model_name, neural_policy, preferred


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def configure_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return device


def fit_scaler(values: np.ndarray) -> dict:
    return {"mean": float(np.mean(values, dtype=np.float64)),
            "std": float(np.std(values, dtype=np.float64) + 1e-8)}


def scale(values: np.ndarray, scaler: dict) -> np.ndarray:
    return np.asarray((np.asarray(values, dtype=np.float32) - float(scaler["mean"])) /
                      float(scaler["std"]), dtype=np.float32)


def class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2)
    if np.any(counts == 0):
        raise ValueError("Both classes must occur in each training split")
    return torch.as_tensor(len(labels) / (2.0 * counts), dtype=torch.float32, device=device)


def augment_light(values: torch.Tensor, generator: torch.Generator, std: float) -> torch.Tensor:
    noise = torch.randn(values.shape, generator=generator, device="cpu").to(values.device) * (.05 * std)
    amplitude = 1.0 + (torch.rand(len(values), 1, 1, generator=generator, device="cpu") - .5).to(values.device) * .2
    return values * amplitude + noise


def _epoch(model, optimizer, criterion, x, y, batch_size, generator, std) -> float:
    model.train()
    order = torch.randperm(len(y), generator=generator, device="cpu")
    total = 0.0
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size].to(x.device)
        values, labels = x.index_select(0, indices), y.index_select(0, indices)
        values = augment_light(values, generator, std)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(values), labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(indices)
    return total / len(y)


@torch.no_grad()
def probabilities(model, values: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    result = np.concatenate([torch.softmax(model(values[i:i + batch_size]), dim=1)[:, 1].cpu().numpy()
                             for i in range(0, len(values), batch_size)]).astype(np.float64)
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise FloatingPointError("Invalid prediction probabilities")
    return result


def helper_module(name: str):
    if name in BRAINDDECODE_MODELS:
        suffix = "braindecode"
    else:
        suffix = {"EEGNet": "eegnet", "4L-CNN": "cnn", "Lee-CRNN": "lee_crnn"}[name]
    return importlib.import_module("pd_eeg.models." + suffix)


def _state(name, variant, values, labels, train, validation, seed, lr, device, policy):
    scaler = fit_scaler(values[train])
    train_values = scale(values[train], scaler)
    validation_values = scale(values[validation], scaler)
    seed_all(seed)
    model = build_model(name, variant=variant, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(labels[train], device) if policy.weighted else None,
                                    label_smoothing=policy.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=policy.weight_decay)
    return {"model": model, "optimizer": optimizer, "criterion": criterion,
            "train": torch.as_tensor(train_values, device=device),
            "labels": torch.as_tensor(labels[train], dtype=torch.int64, device=device),
            "validation": torch.as_tensor(validation_values, device=device),
            "validation_indices": np.asarray(validation), "scaler": scaler,
            "data_std": float(np.std(train_values, dtype=np.float64) + 1e-8),
            "generator": torch.Generator(device="cpu").manual_seed(seed),
            "cpu_rng": torch.get_rng_state().clone(),
            "cuda_rng": torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None}


def _step(name, state, epoch, seed, policy):
    torch.set_rng_state(state["cpu_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng"], state["train"].device)
    if name == "BE-MSNet":
        generator = torch.Generator(device="cpu").manual_seed(seed * 100003 + epoch)
        loss = _epoch(state["model"], state["optimizer"], state["criterion"],
                      state["train"], state["labels"], policy.batch_size, generator, state["data_std"])
    else:
        loss = helper_module(name).train_one_epoch(
            state["model"], state["optimizer"], state["criterion"],
            state["train"], state["labels"], policy.batch_size, state["generator"])
    state["cpu_rng"] = torch.get_rng_state().clone()
    if state["cuda_rng"] is not None:
        state["cuda_rng"] = torch.cuda.get_rng_state(state["train"].device).clone()
    return float(loss)


@dataclass
class NeuralResult:
    probabilities: np.ndarray
    state_dict: dict
    preprocessing: dict
    selected: dict
    candidates: list[dict]
    history: list[dict]
    losses: list[float]


def fit_neural(name, X, y, groups, splits, final_train, prediction_X, *, seed: int,
               variant="full", device="cuda", fixed_epochs=None, max_epochs=None,
               learning_rates=None) -> NeuralResult:
    name = model_name(name)
    policy = neural_policy(name)
    device = configure_device(str(device))
    y, groups = np.asarray(y, dtype=np.int64), np.asarray(groups).astype(str)
    X = np.asarray(X, dtype=np.float32)
    prediction_X = np.asarray(prediction_X, dtype=np.float32)
    values = prepare_input(X, variant) if name == "BE-MSNet" else X
    target = prepare_input(prediction_X, variant) if name == "BE-MSNet" else prediction_X
    limit = policy.max_epochs if max_epochs is None else int(max_epochs)
    rates = policy.learning_rates if learning_rates is None else tuple(map(float, learning_rates))
    candidates, history = [], []
    if fixed_epochs is not None:
        if int(fixed_epochs) < 1 or len(rates) != 1:
            raise ValueError("Fixed-epoch fitting needs a positive budget and one learning rate")
        selected = {"epoch": int(fixed_epochs), "learning_rate": rates[0], "selection": "fixed"}
    else:
        if not splits:
            raise ValueError("Model selection needs training/validation splits")
        expected = set().union(*(set(groups[v]) for _, _, v in splits))
        selected = None
        for learning_rate in rates:
            states = [_state(name, variant, values, y, tr, va, seed, learning_rate, device, policy)
                      for _, tr, va in splits]
            best, stale, stopped = None, 0, False
            for epoch in range(1, limit + 1):
                frames, losses = [], []
                for state in states:
                    losses.append(_step(name, state, epoch, seed, policy))
                    indices = state["validation_indices"]
                    p = probabilities(state["model"], state["validation"], policy.batch_size)
                    frames.append(aggregate_subjects(groups[indices], y[indices], p))
                pooled = pd.concat(frames, ignore_index=True)
                if pooled.subject_id.duplicated().any() or set(pooled.subject_id) != expected:
                    raise ValueError("Inner predictions must cover each validation subject exactly once")
                row = {"epoch": epoch, "learning_rate": float(learning_rate),
                       **binary_metrics(pooled.true_label, pooled.prob_pd),
                       "training_loss": float(np.mean(losses))}
                if not stopped:
                    if preferred(row, best):
                        best, stale = row.copy(), 0
                    else:
                        stale += 1
                    stopped = stale >= policy.patience
                history.append({**row, "selection_active": not stopped})
                if stopped and policy.online_stopping:
                    break
            if best is None:
                raise RuntimeError("No model-selection candidate was obtained")
            candidates.append(best)
            if preferred(best, selected):
                selected = best.copy()
            del states
            if device.type == "cuda":
                torch.cuda.empty_cache()
    final_train = np.asarray(final_train, dtype=np.int64)
    # A separate construction resets initialization and training RNG for the refit.
    state = _state(name, variant, values, y, final_train, final_train[:1], seed,
                   selected["learning_rate"], device, policy)
    losses = [_step(name, state, ep, seed, policy) for ep in range(1, int(selected["epoch"]) + 1)]
    target_values = torch.as_tensor(scale(target, state["scaler"]), device=device)
    p = probabilities(state["model"], target_values, policy.batch_size)
    state_dict = {key: value.detach().cpu().clone() for key, value in state["model"].state_dict().items()}
    return NeuralResult(p, state_dict, state["scaler"], selected, candidates, history, losses)


def predict_neural(checkpoint: dict, X: np.ndarray, device="cpu") -> np.ndarray:
    name, variant = checkpoint["model"], checkpoint.get("variant", "full")
    device = configure_device(device)
    values = prepare_input(X, variant) if name == "BE-MSNet" else np.asarray(X, dtype=np.float32)
    model = build_model(name, variant=variant, device=device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    x = torch.as_tensor(scale(values, checkpoint["preprocessing"]), device=device)
    return probabilities(model, x, neural_policy(name).batch_size)
