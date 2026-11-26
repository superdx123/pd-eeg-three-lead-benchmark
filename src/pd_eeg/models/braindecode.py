"""Braindecode model constructors and training helpers."""

from __future__ import annotations

import importlib.metadata
import os
import random
import warnings
from functools import lru_cache

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn


BRAINDECODE_VERSION = "1.2.0"
MODEL_ORDER = ("EEGConformer", "ATCNet", "EEGNeX", "FBCNet")
MODEL_COMMON_KWARGS = {
    "n_chans": 3,
    "n_outputs": 2,
    "n_times": 512,
    "sfreq": 256,
}


@lru_cache(maxsize=1)
def _model_classes() -> dict[str, type[nn.Module]]:
    installed = importlib.metadata.version("braindecode")
    if installed != BRAINDECODE_VERSION:
        raise RuntimeError(
            f"requires braindecode {BRAINDECODE_VERSION}, found {installed}."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from braindecode.models import ATCNet, EEGConformer, EEGNeX, FBCNet

    return {
        "EEGConformer": EEGConformer,
        "ATCNet": ATCNet,
        "EEGNeX": EEGNeX,
        "FBCNet": FBCNet,
    }


def model_config(model_id: str) -> dict:
    if model_id not in MODEL_ORDER:
        raise KeyError(f"Unknown public model: {model_id}.")
    return {
        "model_id": model_id,
        "implementation": "Braindecode",
        "braindecode_version": BRAINDECODE_VERSION,
        "constructor_kwargs": dict(MODEL_COMMON_KWARGS),
    }


def build_model(model_id: str, device: torch.device) -> nn.Module:
    classes = _model_classes()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = classes[model_id](**MODEL_COMMON_KWARGS)
    return model.to(device)


def configure_deterministic_cuda(require_cuda: bool = True) -> torch.device:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("requires a CUDA-capable GPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return device


def set_deterministic_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def train_scalar_standardization(
    X_train,
    X_other,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    train = np.asarray(X_train, dtype=np.float32)
    other = np.asarray(X_other, dtype=np.float32)
    if train.ndim != 3 or other.ndim != 3 or train.shape[1:] != other.shape[1:]:
        raise ValueError("standardization arrays have incompatible shapes.")
    if not np.isfinite(train).all() or not np.isfinite(other).all():
        raise ValueError("standardization input contains NaN or Inf values.")
    mean_value = float(np.mean(train, dtype=np.float64))
    std_value = float(np.std(train, dtype=np.float64) + 1e-8)
    train_standardized = np.asarray((train - mean_value) / std_value, dtype=np.float32)
    other_standardized = np.asarray((other - mean_value) / std_value, dtype=np.float32)
    if not np.isfinite(train_standardized).all() or not np.isfinite(other_standardized).all():
        raise ValueError("standardization produced NaN or Inf values.")
    return train_standardized, other_standardized, mean_value, std_value


def apply_scalar_standardization(X, mean_value: float, std_value: float) -> np.ndarray:
    array = np.asarray(X, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("inference input contains NaN or Inf values.")
    if not np.isfinite(mean_value) or not np.isfinite(std_value) or float(std_value) <= 0:
        raise ValueError("inference standardization parameters are invalid.")
    result = np.asarray((array - float(mean_value)) / float(std_value), dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("inference standardization produced NaN or Inf values.")
    return result


def balanced_class_weights(labels, device: torch.device) -> torch.Tensor:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    counts = np.bincount(y, minlength=2)
    if len(y) == 0 or np.any(counts <= 0):
        raise ValueError("training labels must contain both classes.")
    weights = len(y) / (2.0 * counts.astype(np.float64))
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def make_training_tensors(X, y, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
    labels = torch.as_tensor(np.asarray(y, dtype=np.int64), device=device)
    return features, labels


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
) -> float:
    model.train()
    order = torch.randperm(len(labels), generator=generator, device="cpu")
    total_loss = 0.0
    total_count = 0
    for start in range(0, len(order), int(batch_size)):
        batch_indices = order[start : start + int(batch_size)].to(features.device)
        inputs = features.index_select(0, batch_indices)
        targets = labels.index_select(0, batch_indices)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        if logits.shape != (len(targets), 2):
            raise RuntimeError(
                f"model returned logits with shape {tuple(logits.shape)}."
            )
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        count = int(len(targets))
        total_loss += float(loss.detach().item()) * count
        total_count += count
    if total_count != len(labels):
        raise RuntimeError("epoch did not consume every training window.")
    value = total_loss / max(total_count, 1)
    if not np.isfinite(value):
        raise RuntimeError("training loss is NaN or Inf.")
    return float(value)


@torch.no_grad()
def infer_probabilities(
    model: nn.Module,
    features: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    probabilities = []
    for start in range(0, len(features), int(batch_size)):
        logits = model(features[start : start + int(batch_size)])
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise RuntimeError(
                f"model returned logits with shape {tuple(logits.shape)}."
            )
        probabilities.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
    result = np.concatenate(probabilities).astype(np.float64, copy=False)
    if len(result) != len(features) or not np.isfinite(result).all():
        raise RuntimeError("inference probabilities are invalid.")
    if np.any((result < 0.0) | (result > 1.0)):
        raise RuntimeError("inference probabilities fall outside [0, 1].")
    return result


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
