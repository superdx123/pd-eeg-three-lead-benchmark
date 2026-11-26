"""Four-layer one-dimensional convolutional baseline."""

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
    "conv_channels": [32, 64, 128, 128],
    "kernel_size": 7,
    "pool_size": 2,
    "hidden_units": 64,
    "dropout": 0.3,
}


class FourLayerCNN(nn.Module):
    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg = dict(MODEL_CONFIG if config is None else config)
        in_channels = int(cfg["in_channels"])
        num_classes = int(cfg["num_classes"])
        channels = [int(value) for value in cfg["conv_channels"]]
        kernel_size = int(cfg["kernel_size"])
        padding = kernel_size // 2

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(
                in_channels,
                channels[0],
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(
                channels[0],
                channels[1],
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=int(cfg["pool_size"]), stride=int(cfg["pool_size"])),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(
                channels[1],
                channels[2],
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=int(cfg["pool_size"]), stride=int(cfg["pool_size"])),
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv1d(
                channels[2],
                channels[3],
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(channels[3]),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[3], int(cfg["hidden_units"])),
            nn.ReLU(inplace=True),
            nn.Dropout(float(cfg["dropout"])),
            nn.Linear(int(cfg["hidden_units"]), num_classes),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.global_pool(x)
        return self.classifier(x)


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
        raise ValueError("PrimaryCNN standardization arrays have incompatible shapes.")
    if not np.isfinite(train).all() or not np.isfinite(other).all():
        raise ValueError("PrimaryCNN standardization input contains NaN or Inf values.")
    mean_value = float(np.mean(train, dtype=np.float64))
    std_value = float(np.std(train, dtype=np.float64) + 1e-8)
    train_standardized = np.asarray((train - mean_value) / std_value, dtype=np.float32)
    other_standardized = np.asarray((other - mean_value) / std_value, dtype=np.float32)
    if not np.isfinite(train_standardized).all() or not np.isfinite(other_standardized).all():
        raise ValueError("PrimaryCNN standardized arrays contain NaN or Inf values.")
    return train_standardized, other_standardized, mean_value, std_value


def apply_scalar_standardization(X, mean_value: float, std_value: float) -> np.ndarray:
    array = np.asarray(X, dtype=np.float32)
    if not np.isfinite(array).all() or not np.isfinite(mean_value) or not np.isfinite(std_value):
        raise ValueError("PrimaryCNN inference standardization input is invalid.")
    if float(std_value) <= 0:
        raise ValueError("PrimaryCNN inference standard deviation must be positive.")
    result = np.asarray((array - float(mean_value)) / float(std_value), dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("PrimaryCNN inference standardization produced NaN or Inf values.")
    return result


def balanced_class_weights(labels, device: torch.device) -> torch.Tensor:
    y = np.asarray(labels, dtype=int).reshape(-1)
    counts = np.bincount(y, minlength=2)
    if len(y) == 0 or np.any(counts <= 0):
        raise ValueError("PrimaryCNN training labels must contain both classes.")
    weights = len(y) / (2.0 * counts.astype(np.float64))
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def build_model(device: torch.device) -> FourLayerCNN:
    model = FourLayerCNN(MODEL_CONFIG).to(device)
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
        raise RuntimeError("PrimaryCNN epoch did not consume every training window.")
    value = total_loss / max(total_count, 1)
    if not np.isfinite(value):
        raise RuntimeError("PrimaryCNN training loss is NaN or Inf.")
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
        raise RuntimeError("PrimaryCNN inference probabilities are invalid.")
    if np.any((result < 0.0) | (result > 1.0)):
        raise RuntimeError("PrimaryCNN inference probabilities fall outside [0, 1].")
    return result


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))

