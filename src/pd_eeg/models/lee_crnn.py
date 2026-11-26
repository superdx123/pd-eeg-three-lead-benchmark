"""Three-lead convolutional recurrent classifier (Lee et al., 2021)."""

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
    "conv_filters": [32, 64],
    "conv_kernel_size": 3,
    "pool_kernel_size": 3,
    "pool_stride": 2,
    "pool_padding": 1,
    "dropout": 0.5,
    "gru_hidden_units": 35,
    "time_distributed_units": 35,
    "gru_layers": 1,
    "bidirectional": False,
}


class LeeCRNN(nn.Module):
    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg = dict(MODEL_CONFIG if config is None else config)
        filters = [int(value) for value in cfg["conv_filters"]]
        kernel_size = int(cfg["conv_kernel_size"])
        pool_kernel = int(cfg["pool_kernel_size"])
        pool_stride = int(cfg["pool_stride"])
        pool_padding = int(cfg["pool_padding"])
        dropout = float(cfg["dropout"])
        hidden_units = int(cfg["gru_hidden_units"])
        time_units = int(cfg["time_distributed_units"])
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(
                int(cfg["in_channels"]),
                filters[0],
                kernel_size=kernel_size,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=pool_kernel,
                stride=pool_stride,
                padding=pool_padding,
            ),
            nn.Dropout(dropout),
            nn.Conv1d(
                filters[0],
                filters[1],
                kernel_size=kernel_size,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=pool_kernel,
                stride=pool_stride,
                padding=pool_padding,
            ),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=filters[1],
            hidden_size=hidden_units,
            num_layers=int(cfg["gru_layers"]),
            batch_first=True,
            bidirectional=bool(cfg["bidirectional"]),
        )
        self.time_distributed = nn.Linear(hidden_units, time_units)
        self.classifier = nn.Linear(time_units, int(cfg["num_classes"]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = x.transpose(1, 2).contiguous()
        x, _ = self.gru(x)
        x = torch.relu(self.time_distributed(x))
        x = x.mean(dim=1)
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


def train_scalar_standardization(
    X_train,
    X_other,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    train = np.asarray(X_train, dtype=np.float32)
    other = np.asarray(X_other, dtype=np.float32)
    if train.ndim != 3 or other.ndim != 3 or train.shape[1:] != other.shape[1:]:
        raise ValueError("Lee-style CRNN standardization arrays have incompatible shapes.")
    if not np.isfinite(train).all() or not np.isfinite(other).all():
        raise ValueError("Lee-style CRNN standardization input contains NaN or Inf.")
    mean_value = float(np.mean(train, dtype=np.float64))
    std_value = float(np.std(train, dtype=np.float64) + 1e-8)
    train_standardized = np.asarray((train - mean_value) / std_value, dtype=np.float32)
    other_standardized = np.asarray((other - mean_value) / std_value, dtype=np.float32)
    if not np.isfinite(train_standardized).all() or not np.isfinite(other_standardized).all():
        raise ValueError("Lee-style CRNN standardization produced NaN or Inf.")
    return train_standardized, other_standardized, mean_value, std_value


def apply_scalar_standardization(X, mean_value: float, std_value: float) -> np.ndarray:
    array = np.asarray(X, dtype=np.float32)
    if not np.isfinite(array).all() or not np.isfinite(mean_value) or not np.isfinite(std_value):
        raise ValueError("Lee-style CRNN inference standardization input is invalid.")
    if float(std_value) <= 0:
        raise ValueError("Lee-style CRNN inference standard deviation must be positive.")
    result = np.asarray((array - float(mean_value)) / float(std_value), dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Lee-style CRNN inference standardization produced NaN or Inf.")
    return result


def build_model(device: torch.device) -> LeeCRNN:
    return LeeCRNN(MODEL_CONFIG).to(device)


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
        raise RuntimeError("Lee-style CRNN epoch did not consume every training window.")
    value = total_loss / max(total_count, 1)
    if not np.isfinite(value):
        raise RuntimeError("Lee-style CRNN training loss is NaN or Inf.")
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
        probabilities.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
    result = np.concatenate(probabilities).astype(np.float64, copy=False)
    if len(result) != len(features) or not np.isfinite(result).all():
        raise RuntimeError("Lee-style CRNN inference probabilities are invalid.")
    if np.any((result < 0.0) | (result > 1.0)):
        raise RuntimeError("Lee-style CRNN probabilities fall outside [0, 1].")
    return result


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
