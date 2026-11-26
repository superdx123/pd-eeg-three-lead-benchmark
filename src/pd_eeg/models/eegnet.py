"""Three-lead EEGNet adaptation (Lawhern et al., 2018)."""

from __future__ import annotations

import os
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn


MODEL_CONFIG = {
    "in_channels": 3,
    "n_samples": 512,
    "num_classes": 2,
    "f1": 8,
    "depth_multiplier": 2,
    "f2": 16,
    "temporal_kernel": 64,
    "separable_kernel": 16,
    "dropout": 0.5,
}


class EEGNet(nn.Module):
    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg = dict(MODEL_CONFIG if config is None else config)
        in_channels = int(cfg["in_channels"])
        n_samples = int(cfg["n_samples"])
        f1 = int(cfg["f1"])
        depth_multiplier = int(cfg["depth_multiplier"])
        f2 = int(cfg["f2"])
        temporal_kernel = int(cfg["temporal_kernel"])
        separable_kernel = int(cfg["separable_kernel"])
        dropout = float(cfg["dropout"])

        self.block1 = nn.Sequential(
            nn.Conv2d(
                1,
                f1,
                kernel_size=(1, temporal_kernel),
                padding=(0, temporal_kernel // 2),
                bias=False,
            ),
            nn.BatchNorm2d(f1),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                f1,
                f1 * depth_multiplier,
                kernel_size=(in_channels, 1),
                groups=f1,
                bias=False,
            ),
            nn.BatchNorm2d(f1 * depth_multiplier),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(
                f1 * depth_multiplier,
                f1 * depth_multiplier,
                kernel_size=(1, separable_kernel),
                padding=(0, separable_kernel // 2),
                groups=f1 * depth_multiplier,
                bias=False,
            ),
            nn.Conv2d(f1 * depth_multiplier, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_channels, n_samples)
            feature_dim = int(self._forward_features(dummy).shape[1])
        self.classifier = nn.Linear(feature_dim, int(cfg["num_classes"]))

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.depthwise(x)
        x = self.separable(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._forward_features(x.unsqueeze(1)))


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


def train_scalar_standardization(X_train, X_other) -> tuple[np.ndarray, np.ndarray, float, float]:
    train = np.asarray(X_train, dtype=np.float32)
    other = np.asarray(X_other, dtype=np.float32)
    if train.ndim != 3 or other.ndim != 3 or train.shape[1:] != other.shape[1:]:
        raise ValueError("EEGNet standardization arrays have incompatible shapes.")
    if not np.isfinite(train).all() or not np.isfinite(other).all():
        raise ValueError("EEGNet standardization input contains NaN or Inf values.")
    mean_value = float(np.mean(train, dtype=np.float64))
    std_value = float(np.std(train, dtype=np.float64) + 1e-8)
    train_standardized = np.asarray((train - mean_value) / std_value, dtype=np.float32)
    other_standardized = np.asarray((other - mean_value) / std_value, dtype=np.float32)
    if not np.isfinite(train_standardized).all() or not np.isfinite(other_standardized).all():
        raise ValueError("EEGNet standardized arrays contain NaN or Inf values.")
    return train_standardized, other_standardized, mean_value, std_value


def apply_scalar_standardization(X, mean_value: float, std_value: float) -> np.ndarray:
    array = np.asarray(X, dtype=np.float32)
    if not np.isfinite(array).all() or not np.isfinite(mean_value) or not np.isfinite(std_value):
        raise ValueError("EEGNet inference standardization input is invalid.")
    if float(std_value) <= 0:
        raise ValueError("EEGNet inference standard deviation must be positive.")
    result = np.asarray((array - float(mean_value)) / float(std_value), dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("EEGNet inference standardization produced NaN or Inf values.")
    return result


def balanced_class_weights(labels, device: torch.device) -> torch.Tensor:
    y = np.asarray(labels, dtype=int).reshape(-1)
    counts = np.bincount(y, minlength=2)
    if len(y) == 0 or np.any(counts <= 0):
        raise ValueError("EEGNet training labels must contain both classes.")
    weights = len(y) / (2.0 * counts.astype(np.float64))
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def build_model(device: torch.device) -> EEGNet:
    model = EEGNet(MODEL_CONFIG).to(device)
    return model


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
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        count = int(len(targets))
        total_loss += float(loss.detach().item()) * count
        total_count += count
    if total_count != len(labels):
        raise RuntimeError("EEGNet epoch did not consume every training window.")
    value = total_loss / max(total_count, 1)
    if not np.isfinite(value):
        raise RuntimeError("EEGNet training loss is NaN or Inf.")
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
        probability = torch.softmax(logits, dim=1)[:, 1]
        probabilities.append(probability.detach().cpu().numpy())
    result = np.concatenate(probabilities).astype(np.float64, copy=False)
    if len(result) != len(features) or not np.isfinite(result).all():
        raise RuntimeError("EEGNet inference probabilities are invalid.")
    if np.any((result < 0.0) | (result > 1.0)):
        raise RuntimeError("EEGNet inference probabilities fall outside [0, 1].")
    return result


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
