"""Band-Expanded Multi-Scale Network and the manuscript's matched controls."""

from __future__ import annotations

import math

import torch
from torch import nn


VARIANTS = {
    "full": {"input_channels": 18, "kernels": (7, 15, 31), "pooling": "mean_std"},
    "raw_only": {"input_channels": 3, "kernels": (7, 15, 31), "pooling": "mean_std"},
    "single_scale": {"input_channels": 18, "kernels": (15,), "pooling": "mean_std"},
    "mean_only": {"input_channels": 18, "kernels": (7, 15, 31), "pooling": "mean"},
    "r18": {"input_channels": 18, "kernels": (7, 15, 31), "pooling": "mean_std"},
}


class VectorGroupNorm(nn.Module):
    def __init__(self, num_groups: int, num_features: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_features)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.norm(values.unsqueeze(-1)).squeeze(-1)


class TemporalScaleBranch(nn.Module):
    def __init__(self, kernel_size: int, n_channels: int, pooling: str = "mean_std"):
        super().__init__()
        if kernel_size % 2 != 1 or pooling not in {"mean_std", "mean"}:
            raise ValueError("Use an odd kernel and mean or mean_std pooling.")
        divisor = abs(6 * n_channels) // math.gcd(6, n_channels)
        hidden_channels = max(divisor, ((24 + divisor - 1) // divisor) * divisor)
        self.pooling = pooling
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, hidden_channels, kernel_size=kernel_size, stride=4,
                      padding=kernel_size // 2, bias=False),
            nn.GroupNorm(6, hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, stride=2, padding=2,
                      groups=hidden_channels, bias=False),
            nn.GroupNorm(6, hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, 32, kernel_size=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        features = self.features(windows)
        if self.pooling == "mean":
            return features.mean(dim=-1)
        return torch.cat((features.mean(dim=-1), features.std(dim=-1, correction=0)), dim=-1)


class MultiScaleTemporalEncoder(nn.Module):
    def __init__(self, n_channels: int, kernels: tuple[int, ...], pooling: str):
        super().__init__()
        self.embedding_dim = 96
        self.branches = nn.ModuleList(
            TemporalScaleBranch(kernel, n_channels=n_channels, pooling=pooling)
            for kernel in kernels
        )
        self.extra_stack = nn.Identity()
        features_per_branch = 64 if pooling == "mean_std" else 32
        self.projection = nn.Sequential(
            nn.Linear(len(kernels) * features_per_branch, self.embedding_dim, bias=False),
            VectorGroupNorm(8, self.embedding_dim),
            nn.GELU(),
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([branch(windows) for branch in self.branches], dim=-1)
        return self.projection(self.extra_stack(combined))


class BEMSNet(nn.Module):
    """Classify prepared waveforms; use ``bands.prepare_input`` before scaling.

    State-dictionary names and construction order match the reported full model.
    ``r18`` changes only the external input representation, not this architecture.
    """

    def __init__(self, variant: str = "full"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown BE-MSNet variant: {variant}")
        specification = VARIANTS[variant]
        self.variant = variant
        self.input_channels = specification["input_channels"]
        self.encoder = MultiScaleTemporalEncoder(
            n_channels=self.input_channels,
            kernels=specification["kernels"],
            pooling=specification["pooling"],
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(96, 96, bias=False),
            VectorGroupNorm(8, 96),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(p=0.0)
        self.head = nn.Linear(96, 2, bias=True)

    def embed(self, windows: torch.Tensor) -> torch.Tensor:
        return self.fusion_proj(self.encoder(windows))

    def classify(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(embedding))

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 3 or windows.shape[1:] != (self.input_channels, 512):
            raise ValueError(f"Expected prepared input [N, {self.input_channels}, 512].")
        return self.classify(self.embed(windows))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
