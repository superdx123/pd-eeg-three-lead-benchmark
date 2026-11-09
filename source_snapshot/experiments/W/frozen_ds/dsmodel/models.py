"""Pluggable lightweight EEG models for the self-model experiments.

Contract: model(x) -> logits of shape [N, 2] for flat [N, C, 512] windows.
Channel-input transforms are applied by the trainer before the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from dsmodel.transforms import BANDS, n_input_channels

INPUT_SAMPLES = 512
DEFAULT_EMBEDDING_DIM = 192


def _lcm(a: int, b: int) -> int:
    import math
    return a * b // math.gcd(a, b)


def _divisible(value: int, divisor: int) -> int:
    return max(divisor, ((value + divisor - 1) // divisor) * divisor)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    description: str = ""
    encoder: str = "long_depthwise"        # long_depthwise | multi_scale
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    spectral: str = "none"                 # none | bandpower | stft
    spectral_dim: int = 5 * 3
    dropout: float = 0.0
    preprocess: str = "global_scalar"      # global_scalar | window_zscore
    augment: str = "none"                  # none | light | heavy
    mixup_alpha: float = 0.0
    label_smoothing: float = 0.0
    width_mult: float = 1.0
    input_transform: str = "raw"           # raw | bipolar | band | band_bipolar | band_raw
    n_bands: int = 5                       # 5 or 8 frequency bands for band transforms
    params: dict = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        base = int(self.params.get("n_channels", 3))
        return n_input_channels(self.input_transform, base=base, n_bands=self.n_bands)


REGISTRY: dict = {}


def register(name: str, config: ModelConfig) -> None:
    if name in REGISTRY:
        raise ValueError(f"Duplicate model registration: {name}")
    REGISTRY[name] = config


def get_config(name: str) -> ModelConfig:
    try:
        return REGISTRY[name.upper()]
    except KeyError:
        raise KeyError(f"Unknown model {name!r}; registered: {sorted(REGISTRY)}")


def build_model(config: ModelConfig, device: torch.device) -> "DSModel":
    return DSModel(config).to(device)


# ---------------------------------------------------------------------------
# Spectral features
# ---------------------------------------------------------------------------

class FixedBandPower(nn.Module):
    def __init__(self, sfreq: float = 256.0, n_channels: int = 3, bands_hz=BANDS):
        super().__init__()
        freqs = np.fft.rfftfreq(INPUT_SAMPLES, d=1.0 / sfreq)
        self.register_buffer("freqs", torch.as_tensor(freqs, dtype=torch.float32))
        masks = [torch.as_tensor(((freqs >= lo) & (freqs < hi)).astype(np.float32)) for lo, hi in bands_hz]
        self.register_buffer("band_masks", torch.stack(masks, dim=0))
        self.n_bands = len(bands_hz)
        self.n_channels = int(n_channels)
        self.out_dim = self.n_bands * self.n_channels

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        x = windows - windows.mean(dim=-1, keepdim=True)
        windowed = x * torch.hann_window(INPUT_SAMPLES, device=x.device, dtype=x.dtype)
        spectrum = torch.fft.rfft(windowed, dim=-1).abs() ** 2
        total = spectrum.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        rel = spectrum / total
        band = torch.einsum("bcf,nf->bcn", rel, self.band_masks)
        flat = band.reshape(band.shape[0], -1)
        return torch.log10(flat.clamp_min(1e-12) + 1e-12)


class BandRatioBranch(nn.Module):
    """Relative band-power ratios (Round-20 candidate, 490 cohort-offset hypothesis).

    Per-window per-channel relative band power (5 bands) on the RAW channels,
    then log10 ratios of adjacent bands + theta/delta + alpha/delta:
    [theta/delta, alpha/delta, beta/alpha, gamma/beta] x C channels.
    Ratios remove the absolute power level (cohort offset) while keeping
    spectral SHAPE information.
    """

    def __init__(self, sfreq: float = 256.0, n_channels: int = 3, bands_hz=BANDS):
        super().__init__()
        freqs = np.fft.rfftfreq(INPUT_SAMPLES, d=1.0 / sfreq)
        self.register_buffer("freqs", torch.as_tensor(freqs, dtype=torch.float32))
        masks = [torch.as_tensor(((freqs >= lo) & (freqs < hi)).astype(np.float32)) for lo, hi in bands_hz]
        self.register_buffer("band_masks", torch.stack(masks, dim=0))
        self.n_bands = len(bands_hz)
        self.n_channels = int(n_channels)
        # ratio pairs over band indices: delta=0 theta=1 alpha=2 beta=3 gamma=4
        self.out_dim = 4 * self.n_channels

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        x = windows[:, :self.n_channels] - windows[:, :self.n_channels].mean(dim=-1, keepdim=True)
        windowed = x * torch.hann_window(INPUT_SAMPLES, device=x.device, dtype=x.dtype)
        spectrum = torch.fft.rfft(windowed, dim=-1).abs() ** 2
        total = spectrum.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        rel = spectrum / total
        band = torch.einsum("bcf,nf->bcn", rel, self.band_masks).clamp_min(1e-12)  # [B, C, 5]
        ratios = [
            torch.log10(band[:, :, 1] / band[:, :, 0]),  # theta/delta
            torch.log10(band[:, :, 2] / band[:, :, 0]),  # alpha/delta
            torch.log10(band[:, :, 3] / band[:, :, 1]),  # beta/theta
            torch.log10(band[:, :, 4] / band[:, :, 3]),  # gamma/beta
        ]
        return torch.stack(ratios, dim=-1).reshape(band.shape[0], -1)  # [B, 4*C]


class STFTBranch(nn.Module):
    def __init__(self, embed_dim: int, n_channels: int = 3, n_fft: int = 64, hop: int = 16):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        n_freq = n_fft // 2 + 1
        self.net = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2), bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, embed_dim, bias=False),
        )
        self.register_buffer("window", torch.hann_window(n_fft, dtype=torch.float32))

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        x = windows - windows.mean(dim=-1, keepdim=True)
        n, c = x.shape[:2]
        flat = x.reshape(n * c, -1)
        stft = torch.stft(flat, n_fft=self.n_fft, hop_length=self.hop,
                          window=self.window.to(x.device), return_complex=True)
        mag = stft.abs().reshape(n, c, stft.shape[1], stft.shape[2])
        return self.net(torch.log10(mag.clamp_min(1e-6)))


# ---------------------------------------------------------------------------
# Temporal encoders
# ---------------------------------------------------------------------------

class VectorGroupNorm(nn.Module):
    def __init__(self, num_groups: int, num_features: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_features)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.norm(values.unsqueeze(-1)).squeeze(-1)


class LongDepthwiseTemporalBranch(nn.Module):
    def __init__(self, kernel_size: int, n_channels: int = 3, output_channels: int = 64, width_mult: float = 1.0):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernels must be odd")
        channelwise_filters = _divisible(int(48 * width_mult), _lcm(12, n_channels))
        output_channels = max(8, int(output_channels * width_mult))
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, channelwise_filters, kernel_size=kernel_size, stride=2,
                      padding=kernel_size // 2, groups=n_channels, bias=False),
            nn.GroupNorm(12, channelwise_filters),
            nn.GELU(),
            nn.Conv1d(channelwise_filters, output_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, output_channels),
            nn.GELU(),
            nn.Conv1d(output_channels, output_channels, kernel_size=9, stride=2, padding=4,
                      groups=output_channels, bias=False),
            nn.GroupNorm(8, output_channels),
            nn.GELU(),
            nn.Conv1d(output_channels, output_channels, kernel_size=5, padding=2,
                      groups=output_channels, bias=False),
            nn.GroupNorm(8, output_channels),
            nn.GELU(),
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        features = self.features(windows)
        return torch.cat((features.mean(dim=-1), features.std(dim=-1, correction=0)), dim=-1)


class LongDepthwiseTemporalEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 192, n_channels: int = 3, width_mult: float = 1.0):
        super().__init__()
        if embedding_dim <= 0 or embedding_dim % 8 != 0:
            raise ValueError("embedding_dim must be positive multiple of 8")
        self.embedding_dim = int(embedding_dim)
        self.branches = nn.ModuleList(
            LongDepthwiseTemporalBranch(k, n_channels=n_channels, width_mult=width_mult)
            for k in (15, 31, 63, 127)
        )
        out_per = max(8, int(64 * width_mult))
        self.projection = nn.Sequential(
            nn.Linear(4 * 2 * out_per, self.embedding_dim, bias=False),
            VectorGroupNorm(8, self.embedding_dim),
            nn.GELU(),
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.projection(torch.cat([b(windows) for b in self.branches], dim=-1))


class TemporalScaleBranch(nn.Module):
    def __init__(self, kernel_size: int, n_channels: int = 3, branch_channels: int = 32,
                 pooling: str = "mean_std"):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernels must be odd")
        hidden_channels = _divisible(24, _lcm(6, n_channels))
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
            nn.Conv1d(hidden_channels, branch_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, branch_channels),
            nn.GELU(),
        )
        if pooling == "attn":
            self.attn_proj = nn.Sequential(
                nn.Conv1d(branch_channels, 8, kernel_size=1, bias=False),
                nn.GELU(),
                nn.Conv1d(8, 1, kernel_size=1, bias=False),
            )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        features = self.features(windows)
        if self.pooling == "mean_std":
            return torch.cat((features.mean(dim=-1), features.std(dim=-1, correction=0)), dim=-1)
        if self.pooling == "attn":
            attn = torch.softmax(self.attn_proj(features), dim=-1)
            pooled = (features * attn).sum(dim=-1)
            return torch.cat((pooled, features.std(dim=-1, correction=0)), dim=-1)
        raise ValueError(f"Unknown pooling {self.pooling!r}")


class MultiScaleTemporalEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 96, n_channels: int = 3, kernels=(7, 15, 31),
                 pooling: str = "mean_std", depth: int = 1):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.depth = int(depth)
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.branches = nn.ModuleList(
            TemporalScaleBranch(k, n_channels=n_channels, branch_channels=32, pooling=pooling)
            for k in kernels
        )
        out_per = 2 * 32
        if depth > 1:
            branch_out = len(kernels) * out_per
            extra = []
            for _ in range(depth - 1):
                extra.append(nn.Linear(branch_out, branch_out, bias=False))
                extra.append(nn.GELU())
                extra.append(nn.Dropout(p=0.1))
            self.extra_stack = nn.Sequential(*extra) if extra else nn.Identity()
        else:
            self.extra_stack = nn.Identity()
        self.projection = nn.Sequential(
            nn.Linear(len(kernels) * out_per, self.embedding_dim, bias=False),
            VectorGroupNorm(8, self.embedding_dim),
            nn.GELU(),
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([b(windows) for b in self.branches], dim=-1)
        return self.projection(self.extra_stack(combined))


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class DualBranchTemporalEncoder(nn.Module):
    """Two independent multi-scale encoders on raw (first `split` channels) and band
    (remaining channels), embeddings concatenated. Input is band_raw (raw+band)."""

    def __init__(self, embedding_dim: int = 96, n_channels: int = 18, split: int = 3,
                 pooling: str = "mean_std", depth: int = 1):
        super().__init__()
        if not (0 < split < n_channels):
            raise ValueError("dual_branch requires 0 < split < n_channels")
        self.split = int(split)
        self.raw_branch = MultiScaleTemporalEncoder(embedding_dim, n_channels=split,
                                                    pooling=pooling, depth=depth)
        self.band_branch = MultiScaleTemporalEncoder(embedding_dim, n_channels=n_channels - split,
                                                     pooling=pooling, depth=depth)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        raw = windows[:, :self.split]
        band = windows[:, self.split:]
        return torch.cat((self.raw_branch(raw), self.band_branch(band)), dim=-1)


class HybridTemporalEncoder(nn.Module):
    """Cross-architecture hybrid: multi-scale short kernels + long depthwise kernels.

    Round-22 candidate: single-model embodiment of the empirically established
    complementarity (T5 long-temporal architecture leads ds003490 bAcc while the
    multi-scale short-kernel DS28 leads ds002778). Both branches see the same
    input; embeddings are concatenated and fused by DSModel.fusion_proj.
    """

    def __init__(self, embedding_dim: int = 96, n_channels: int = 3,
                 short_kernels=(7, 15, 31), long_kernels=(15, 31, 63, 127),
                 width_mult: float = 1.0):
        super().__init__()
        self.short_enc = MultiScaleTemporalEncoder(embedding_dim, n_channels=n_channels,
                                                   kernels=short_kernels)
        self.long_enc = LongDepthwiseTemporalEncoder(embedding_dim, n_channels=n_channels,
                                                     width_mult=width_mult)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.short_enc(windows), self.long_enc(windows)), dim=-1)


class SubjectAttentionHead(nn.Module):
    """Learned per-window importance weight for subject aggregation (Round-19 MIL).

    weight = v^T act(W h + b); per-subject softmax at aggregation time.
    Trained end-to-end via the subject-level CE term (trainer subject_attn branch).
    """

    def __init__(self, embedding_dim: int, hidden: int = 32, act: str = "gelu"):
        super().__init__()
        activation = nn.Tanh() if str(act) == "tanh" else nn.GELU()
        self.net = nn.Sequential(
            nn.Linear(int(embedding_dim), int(hidden), bias=True),
            activation,
            nn.Linear(int(hidden), 1, bias=True),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)  # [N, 1]


class EEGNet(nn.Module):
    """EEGNet (Lawhern et al. 2018) baseline for fair comparison (Round-23).

    Standard configuration: F1=8, D=2, F2=16, dropout 0.25, input [N, C, T].
    Trained through the SAME frozen protocol as DS28 (2 lr candidates, inner
    subject-level selection, final refit, threshold 0.5).
    """

    def __init__(self, n_channels: int = 3, T: int = 512, F1: int = 8, D: int = 2,
                 F2: int = 16, dropout_rate: float = 0.25):
        super().__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, D * F1, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
        )
        self.separable1 = nn.Sequential(
            nn.Conv2d(D * F1, D * F1, (1, 16), padding=(0, 8), groups=D * F1, bias=False),
            nn.Conv2d(D * F1, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate),
        )
        self.separable2 = nn.Sequential(
            nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), groups=F2, bias=False),
            nn.Conv2d(F2, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate),
        )
        self.classifier = nn.Linear(F2 * (T // 256), 2, bias=True)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        x = windows.unsqueeze(1)  # [N, 1, C, T]
        x = self.firstconv(x)
        x = self.depthwise(x)
        x = self.separable1(x)
        x = self.separable2(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)


class DSModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        n_ch = config.n_channels
        self.eegnet = None
        if config.encoder == "eegnet":
            self.encoder = None
            self.eegnet = EEGNet(n_channels=n_ch, T=INPUT_SAMPLES)
            self.spectral = None
            self.spectral_net = None
            self.fusion_proj = None
            self.dropout = nn.Identity()
            self.head = None
            self.subject_attn = None
            return
        if config.encoder == "long_depthwise":
            self.encoder = LongDepthwiseTemporalEncoder(config.embedding_dim, n_channels=n_ch, width_mult=config.width_mult)
        elif config.encoder == "multi_scale":
            kstr = config.params.get("kernels")
            kernels = tuple(int(x) for x in str(kstr).split(",")) if kstr else (7, 15, 31)
            self.encoder = MultiScaleTemporalEncoder(
                config.embedding_dim, n_channels=n_ch, kernels=kernels,
                pooling=str(config.params.get("pooling", "mean_std")),
                depth=int(config.params.get("depth", 1)),
            )
        elif config.encoder == "dual_branch":
            self.encoder = DualBranchTemporalEncoder(
                config.embedding_dim, n_channels=n_ch,
                split=int(config.params.get("split", 3)),
                pooling=str(config.params.get("pooling", "mean_std")),
                depth=int(config.params.get("depth", 1)),
            )
        elif config.encoder == "hybrid":
            self.encoder = HybridTemporalEncoder(config.embedding_dim, n_channels=n_ch)
        else:
            raise ValueError(f"Unknown encoder {config.encoder}")

        if config.spectral == "bandpower":
            self.spectral = FixedBandPower(n_channels=n_ch)
            sp_dim = self.spectral.out_dim
            spectral_proj = nn.Linear(sp_dim, config.embedding_dim // 2, bias=False)
            self.spectral_net = nn.Sequential(spectral_proj, VectorGroupNorm(8, config.embedding_dim // 2), nn.GELU())
            fusion_in = config.embedding_dim + config.embedding_dim // 2
        elif config.spectral == "stft":
            self.spectral = STFTBranch(config.embedding_dim // 2, n_channels=n_ch)
            self.spectral_net = nn.Identity()
            fusion_in = config.embedding_dim + config.embedding_dim // 2
        elif config.spectral == "band_ratio":
            self.spectral = BandRatioBranch(n_channels=n_ch)
            sp_dim = self.spectral.out_dim
            self.spectral_net = nn.Sequential(
                nn.Linear(sp_dim, config.embedding_dim // 2, bias=False),
                VectorGroupNorm(8, config.embedding_dim // 2),
                nn.GELU(),
            )
            fusion_in = config.embedding_dim + config.embedding_dim // 2
        else:
            self.spectral = None
            self.spectral_net = None
            fusion_in = config.embedding_dim * (2 if config.encoder in ("dual_branch", "hybrid") else 1)

        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_in, config.embedding_dim, bias=False),
            VectorGroupNorm(8, config.embedding_dim),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(p=float(config.dropout))
        head_hidden = int(config.params.get("head_hidden", 0))
        if head_hidden > 0:
            self.head = nn.Sequential(
                nn.Linear(config.embedding_dim, head_hidden, bias=False),
                nn.GELU(),
                nn.Linear(head_hidden, 2, bias=True),
            )
        else:
            self.head = nn.Linear(config.embedding_dim, 2, bias=True)
        if config.params.get("subject_attn", False):
            self.subject_attn = SubjectAttentionHead(
                config.embedding_dim,
                hidden=int(config.params.get("attn_hidden", 32)),
                act=str(config.params.get("attn_act", "gelu")),
            )
        else:
            self.subject_attn = None

    def embed(self, windows: torch.Tensor) -> torch.Tensor:
        temporal = self.encoder(windows)
        if self.spectral is not None:
            spectral = self.spectral_net(self.spectral(windows))
            fused = torch.cat((temporal, spectral), dim=-1)
        else:
            fused = temporal
        return self.fusion_proj(fused)

    def classify(self, embedding: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.dropout(embedding))
        if logits.shape != (embedding.shape[0], 2):
            raise RuntimeError(f"Unexpected logits shape {tuple(logits.shape)}")
        return logits

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        if self.eegnet is not None:
            logits = self.eegnet(windows)
            if logits.shape != (windows.shape[0], 2):
                raise RuntimeError(f"Unexpected logits shape {tuple(logits.shape)}")
            return logits
        return self.classify(self.embed(windows))

    def embed_logits_weights(self, windows: torch.Tensor) -> tuple:
        """Attention-MIL outputs: (embedding, window logits, per-window weights)."""
        embedding = self.embed(windows)
        logits = self.classify(embedding)
        weights = self.subject_attn(embedding)
        return embedding, logits, weights


def trainable_parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ---------------------------------------------------------------------------
# Registered candidates
# ---------------------------------------------------------------------------

register("DS0", ModelConfig(name="DS0", description="T5-style long depthwise temporal encoder, window CE",
                            encoder="long_depthwise", embedding_dim=192))
register("DS0_DP", ModelConfig(name="DS0_DP", description="DS0 + head dropout 0.3",
                               encoder="long_depthwise", embedding_dim=192, dropout=0.3))
register("DS0_AUG", ModelConfig(name="DS0_AUG", description="DS0 + light augmentation",
                                encoder="long_depthwise", embedding_dim=192, augment="light"))
register("DS0_AUGHEAVY", ModelConfig(name="DS0_AUGHEAVY", description="DS0 + heavy augmentation",
                                     encoder="long_depthwise", embedding_dim=192, augment="heavy"))
register("DS0_Z", ModelConfig(name="DS0_Z", description="DS0 + per-window z-score preprocessing",
                              encoder="long_depthwise", embedding_dim=192, preprocess="window_zscore"))
register("DS0_Z_AUG", ModelConfig(name="DS0_Z_AUG", description="DS0 + z-score + light aug",
                                  encoder="long_depthwise", embedding_dim=192, preprocess="window_zscore", augment="light"))
register("DS0_LS", ModelConfig(name="DS0_LS", description="DS0 + label smoothing 0.1",
                               encoder="long_depthwise", embedding_dim=192, label_smoothing=0.1))
register("DS0_MIX", ModelConfig(name="DS0_MIX", description="DS0 + mixup alpha 0.2",
                                encoder="long_depthwise", embedding_dim=192, mixup_alpha=0.2))
register("DS0_BP", ModelConfig(name="DS0_BP", description="DS0 + fixed bandpower branch",
                               encoder="long_depthwise", embedding_dim=192, spectral="bandpower"))
register("DS0_STFT", ModelConfig(name="DS0_STFT", description="DS0 + learned log-STFT branch",
                                 encoder="long_depthwise", embedding_dim=192, spectral="stft"))
register("DS0_W2", ModelConfig(name="DS0_W2", description="DS0 with 2x width",
                               encoder="long_depthwise", embedding_dim=192, width_mult=2.0))
register("DS0_MS", ModelConfig(name="DS0_MS", description="compact multi-scale encoder (3 short kernels)",
                               encoder="multi_scale", embedding_dim=96))

# --- batch2: input transforms + best-of combos ---
register("DS1_MS_AUG", ModelConfig(name="DS1_MS_AUG", description="multi-scale + light aug",
                                   encoder="multi_scale", embedding_dim=96, augment="light"))
register("DS1_MS_AUG_LS", ModelConfig(name="DS1_MS_AUG_LS", description="multi-scale + light aug + LS 0.05",
                                      encoder="multi_scale", embedding_dim=96, augment="light", label_smoothing=0.05))
register("DS1_MS_Z_AUG", ModelConfig(name="DS1_MS_Z_AUG", description="multi-scale + zscore + light aug",
                                     encoder="multi_scale", embedding_dim=96, preprocess="window_zscore", augment="light"))
register("DS1_BI", ModelConfig(name="DS1_BI", description="multi-scale on bipolar 6ch",
                               encoder="multi_scale", embedding_dim=96, input_transform="bipolar"))
register("DS1_BI_AUG", ModelConfig(name="DS1_BI_AUG", description="multi-scale on bipolar 6ch + light aug",
                                   encoder="multi_scale", embedding_dim=96, input_transform="bipolar", augment="light"))
register("DS1_BAND", ModelConfig(name="DS1_BAND", description="multi-scale on 5-band 15ch",
                                 encoder="multi_scale", embedding_dim=96, input_transform="band"))
register("DS1_BAND_AUG", ModelConfig(name="DS1_BAND_AUG", description="multi-scale on 15ch bands + light aug",
                                     encoder="multi_scale", embedding_dim=96, input_transform="band", augment="light"))
register("DS1_BI_BAND", ModelConfig(name="DS1_BI_BAND", description="multi-scale on 30ch bipolar-bands",
                                    encoder="multi_scale", embedding_dim=96, input_transform="band_bipolar"))
register("DS1_MS_LD", ModelConfig(name="DS1_MS_LD", description="multi-scale + head dropout 0.3",
                                  encoder="multi_scale", embedding_dim=96, dropout=0.3))
register("DS1_MS_W", ModelConfig(name="DS1_MS_W", description="multi-scale with embedding 192",
                                 encoder="multi_scale", embedding_dim=192))

# --- 7-channel candidates (ds002778_7ch only for now) ---
register("DS2_7CH_MS", ModelConfig(name="DS2_7CH_MS", description="multi-scale on 7 channels",
                                   encoder="multi_scale", embedding_dim=96, params={"n_channels": 7}))
register("DS2_7CH_MS_AUG", ModelConfig(name="DS2_7CH_MS_AUG", description="multi-scale 7ch + light aug",
                                       encoder="multi_scale", embedding_dim=96, augment="light", params={"n_channels": 7}))
register("DS2_7CH_MS_Z", ModelConfig(name="DS2_7CH_MS_Z", description="multi-scale 7ch + zscore",
                                     encoder="multi_scale", embedding_dim=96, preprocess="window_zscore", params={"n_channels": 7}))
register("DS2_7CH_LD", ModelConfig(name="DS2_7CH_LD", description="long-depthwise on 7 channels",
                                   encoder="long_depthwise", embedding_dim=192, params={"n_channels": 7}))

# --- TTA variants (batch3) ---
register("DS1_MS_TTA", ModelConfig(name="DS1_MS_TTA", description="multi-scale + TTA 7 views",
                                   encoder="multi_scale", embedding_dim=96, params={"tta": 7}))
register("DS1_MS_AUG_TTA", ModelConfig(name="DS1_MS_AUG_TTA", description="multi-scale + light aug + TTA 7",
                                       encoder="multi_scale", embedding_dim=96, augment="light", params={"tta": 7}))
register("DS1_MS_Z_AUG_TTA", ModelConfig(name="DS1_MS_Z_AUG_TTA", description="multi-scale + zscore + aug + TTA 7",
                                         encoder="multi_scale", embedding_dim=96, preprocess="window_zscore", augment="light", params={"tta": 7}))
register("DS1_BAND_AUG_TTA", ModelConfig(name="DS1_BAND_AUG_TTA", description="band 15ch + aug + TTA 7",
                                         encoder="multi_scale", embedding_dim=96, input_transform="band", augment="light", params={"tta": 7}))
register("DS2_7CH_MS_AUG_TTA", ModelConfig(name="DS2_7CH_MS_AUG_TTA", description="7ch multi-scale + aug + TTA 7",
                                           encoder="multi_scale", embedding_dim=96, augment="light",
                                           params={"n_channels": 7, "tta": 7}))

# --- batch3: combine winners (MS_AUG_LS x BAND x TTA x 7ch) ---
register("DS3_BAND_AUG_LS", ModelConfig(name="DS3_BAND_AUG_LS", description="band 15ch + aug + LS 0.05",
                                        encoder="multi_scale", embedding_dim=96, input_transform="band",
                                        augment="light", label_smoothing=0.05))
register("DS3_MS_AUG_LS_TTA", ModelConfig(name="DS3_MS_AUG_LS_TTA", description="MS + aug + LS + TTA 7",
                                          encoder="multi_scale", embedding_dim=96, augment="light",
                                          label_smoothing=0.05, params={"tta": 7}))
register("DS3_BAND_AUG_TTA", ModelConfig(name="DS3_BAND_AUG_TTA", description="band 15ch + aug + TTA 7",
                                         encoder="multi_scale", embedding_dim=96, input_transform="band",
                                         augment="light", params={"tta": 7}))
register("DS3_BAND_AUG_LS_TTA", ModelConfig(name="DS3_BAND_AUG_LS_TTA", description="band 15ch + aug + LS + TTA 7",
                                            encoder="multi_scale", embedding_dim=96, input_transform="band",
                                            augment="light", label_smoothing=0.05, params={"tta": 7}))
register("DS3_7CH_MS_AUG_LS", ModelConfig(name="DS3_7CH_MS_AUG_LS", description="7ch MS + aug + LS 0.05",
                                          encoder="multi_scale", embedding_dim=96, augment="light",
                                          label_smoothing=0.05, params={"n_channels": 7}))
register("DS3_7CH_BAND", ModelConfig(name="DS3_7CH_BAND", description="7ch band 35ch input",
                                     encoder="multi_scale", embedding_dim=96, input_transform="band",
                                     params={"n_channels": 7}))
register("DS3_7CH_BAND_AUG_LS", ModelConfig(name="DS3_7CH_BAND_AUG_LS", description="7ch band 35ch + aug + LS",
                                            encoder="multi_scale", embedding_dim=96, input_transform="band",
                                            augment="light", label_smoothing=0.05, params={"n_channels": 7}))

# --- batch4: combine band+raw, wider/deeper/attn (7ch direction abandoned after batch3) ---
register("DS4_BR_MS", ModelConfig(name="DS4_BR_MS", description="band+raw 18ch multi-scale (baseline)",
                                  encoder="multi_scale", embedding_dim=96, input_transform="band_raw"))
register("DS4_BR_MS_AUG", ModelConfig(name="DS4_BR_MS_AUG", description="band+raw 18ch + aug",
                                      encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                      augment="light"))
register("DS4_BR_MS_AUG_LS", ModelConfig(name="DS4_BR_MS_AUG_LS", description="band+raw 18ch + aug + LS 0.05",
                                         encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                         augment="light", label_smoothing=0.05))

# --- Round-4: S-focused regularization/sampling candidates (08-10) ---
register("DS27_BR_MS_AUG_LS_SB", ModelConfig(name="DS27_BR_MS_AUG_LS_SB",
                                             description="DS4 + subject-balanced sampling (min-count per subject per epoch)",
                                             encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                             augment="light", label_smoothing=0.05,
                                             params={"subject_balanced": True}))
register("DS28_BR_MS_AUG_LS_WD3", ModelConfig(name="DS28_BR_MS_AUG_LS_WD3",
                                              description="DS4 + weight decay 1e-3",
                                              encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                              augment="light", label_smoothing=0.05,
                                              params={"wd": 1e-3}))

register("DS41_BR_MS_AUG_LS_WD3_E64", ModelConfig(name="DS41_BR_MS_AUG_LS_WD3_E64",
                                                    description="DS28 + embedding 64 (less capacity w/ wd 1e-3)",
                                                    encoder="multi_scale", embedding_dim=64, input_transform="band_raw",
                                                    augment="light", label_smoothing=0.05,
                                                    params={"wd": 1e-3}))

# --- Round-10: self-supervised pretraining on DS28 winner (08-12) ---
register("DS42_BR_MS_AUG_LS_WD3_SSL_MASK", ModelConfig(name="DS42_BR_MS_AUG_LS_WD3_SSL_MASK",
                                                       description="DS28 + masked-reconstruction SSL pretrain (MAE-style) + fine-tune",
                                                       encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                                       augment="light", label_smoothing=0.05,
                                                       params={"wd": 1e-3, "ssl": "masked", "ssl_epochs": 40,
                                                               "ssl_lr": 1e-3, "ssl_seg_len": 16, "ssl_mask_ratio": 0.5}))
register("DS43_BR_MS_AUG_LS_WD3_SSL_CON", ModelConfig(name="DS43_BR_MS_AUG_LS_WD3_SSL_CON",
                                                      description="DS28 + contrastive SSL pretrain (SimCLR-style) + fine-tune",
                                                      encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                                      augment="light", label_smoothing=0.05,
                                                      params={"wd": 1e-3, "ssl": "contrastive", "ssl_epochs": 40,
                                                              "ssl_lr": 1e-3, "ssl_temperature": 0.1}))

# --- Round-11 backup: per-channel z-score (cohort power-offset removal) ---
register("DS44_BR_MS_AUG_LS_CHZ_WD3", ModelConfig(name="DS44_BR_MS_AUG_LS_CHZ_WD3",
                                                  description="DS28 + per-channel window z-score (removes absolute band-power level)",
                                                  encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                                  preprocess="channel_zscore", augment="light", label_smoothing=0.05,
                                                  params={"wd": 1e-3}))

# --- Round-11: consistency regularization (Pi-model style) on DS28 winner ---
register("DS45_BR_MS_AUG_LS_WD3_CONSIST", ModelConfig(name="DS45_BR_MS_AUG_LS_WD3_CONSIST",
                                                      description="DS28 + Pi-model consistency reg (2nd light view, logit MSE, lambda 1.0)",
                                                      encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                                      augment="light", label_smoothing=0.05,
                                                      params={"wd": 1e-3, "consist_lambda": 1.0, "consist_mode": "logit"}))

# --- Round-12 prereg: dual-branch encoder under DS28 hyperparams ---
register("DS46_BR_DUAL_AUG_LS_WD3", ModelConfig(name="DS46_BR_DUAL_AUG_LS_WD3",
                                                description="DS28 hyperparams + dual-branch encoder (raw 3ch + band 15ch independent multi-scale, emb concat)",
                                                encoder="dual_branch", embedding_dim=96, input_transform="band_raw",
                                                augment="light", label_smoothing=0.05,
                                                params={"wd": 1e-3}))

register("DS47_BR_MS_AUGSHIFT_LS_WD3", ModelConfig(name="DS47_BR_MS_AUGSHIFT_LS_WD3",
                                                    description="DS28 + light augment + random circular time shift (+/-16 samples)",
                                                    encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                                    augment="light_shift", label_smoothing=0.05,
                                                    params={"wd": 1e-3}))

register("DS48_BR_MS_AUG_LS_WD3_K63", ModelConfig(name="DS48_BR_MS_AUG_LS_WD3_K63",
                                                    description="DS28 + kernels (7,15,31,63): extra slow-scale branch (alpha period ~25.6 samples, 63 covers 2.5 cycles)",
                                                    encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                                    augment="light", label_smoothing=0.05,
                                                    params={"wd": 1e-3, "kernels": "7,15,31,63"}))
register("DS49_BR_MS_AUG_LS_WD3_S2", ModelConfig(name="DS49_BR_MS_AUG_LS_WD3_S2",
                                                   description="DS28 + stronger light augment (noise 0.08, amp +-15%)",
                                                   encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                                   augment="light_strong", label_smoothing=0.05,
                                                   params={"wd": 1e-3}))

register("DS50_BR_MS_AUG_LS_WD3_SUBJ", ModelConfig(name="DS50_BR_MS_AUG_LS_WD3_SUBJ",
                                                    description="DS28 + per-subject per-channel mean removal (490-specific: subject channel-profile offset removal)",
                                                    encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                                    preprocess="subject_chmean", augment="light", label_smoothing=0.05,
                                                    params={"wd": 1e-3}))


register("DS29_BR_MS_AUG_LS_LS10", ModelConfig(name="DS29_BR_MS_AUG_LS_LS10",
                                               description="DS4 + label smoothing 0.10",
                                               encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                               augment="light", label_smoothing=0.10))
register("DS30_BR_MS_AUG_LS_SB_WD3", ModelConfig(name="DS30_BR_MS_AUG_LS_SB_WD3",
                                                 description="DS4 + subject-balanced sampling + weight decay 1e-3",
                                                 encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                                 augment="light", label_smoothing=0.05,
                                                 params={"subject_balanced": True, "wd": 1e-3}))
# --- Round-5: focal loss candidates (untried; targets hard subjects -> S bAcc) ---
# --- Round-5: on top of DS28 winner (wd=1e-3), 08-11 ---
# --- Round-6: wd refinement + LS scan on DS28 winner (08-11) ---
# --- Round-8: new mechanism dimensions on DS28 winner (08-11) ---
register("DS37_BR_MS_AUG_LS_WD3_K51121", ModelConfig(name="DS37_BR_MS_AUG_LS_WD3_K51121",
                                                     description="DS28 + kernels (5,11,21) (shorter scales)",
                                                     encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                                     augment="light", label_smoothing=0.05,
                                                     params={"wd": 1e-3, "kernels": "5,11,21"}))
register("DS38_BR_MS_AUG_LS_WD3_ADAMW", ModelConfig(name="DS38_BR_MS_AUG_LS_WD3_ADAMW",
                                                    description="DS28 + AdamW optimizer (decoupled wd)",
                                                    encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                                    augment="light", label_smoothing=0.05,
                                                    params={"wd": 1e-3, "optimizer": "adamw"}))

# --- Round-9: pending candidates on DS28 winner (register 08-12 00:15, sync after R8 verdict) ---
register("DS40_BR_MS_AUG_LS_WD3_E128", ModelConfig(name="DS40_BR_MS_AUG_LS_WD3_E128",
                                                    description="DS28 + embedding 128 (more capacity w/ wd 1e-3)",
                                                    encoder="multi_scale", embedding_dim=128, input_transform="band_raw",
                                                    augment="light", label_smoothing=0.05,
                                                    params={"wd": 1e-3}))
register("DS35_BR_MS_AUG_LS_WD5E4", ModelConfig(name="DS35_BR_MS_AUG_LS_WD5E4",
                                                description="DS4 + weight decay 5e-4 (wd optimum refinement)",
                                                encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                                augment="light", label_smoothing=0.05,
                                                params={"wd": 5e-4}))
register("DS36_BR_MS_AUG_LS_WD3_LS10", ModelConfig(name="DS36_BR_MS_AUG_LS_WD3_LS10",
                                                   description="DS28 (wd=1e-3) + label smoothing 0.10",
                                                   encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                                   augment="light", label_smoothing=0.10,
                                                   params={"wd": 1e-3}))
register("DS33_BR_MS_AUG_LS_WD3_FL2", ModelConfig(name="DS33_BR_MS_AUG_LS_WD3_FL2",
                                                  description="DS28 (wd=1e-3) + focal loss gamma 2.0",
                                                  encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                                  augment="light", label_smoothing=0.05,
                                                  params={"wd": 1e-3, "loss": "focal", "focal_gamma": 2.0}))
register("DS34_BR_MS_AUG_LS_WD33", ModelConfig(name="DS34_BR_MS_AUG_LS_WD33",
                                               description="DS28 (wd=1e-3) + weight decay 3e-3",
                                               encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                               augment="light", label_smoothing=0.05,
                                               params={"wd": 3e-3}))
register("DS31_BR_MS_AUG_LS_FL2", ModelConfig(name="DS31_BR_MS_AUG_LS_FL2",
                                              description="DS4 + focal loss gamma 2.0",
                                              encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                              augment="light", label_smoothing=0.05,
                                              params={"loss": "focal", "focal_gamma": 2.0}))
register("DS32_BR_MS_AUG_LS_FL1", ModelConfig(name="DS32_BR_MS_AUG_LS_FL1",
                                              description="DS4 + focal loss gamma 1.0",
                                              encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                              augment="light", label_smoothing=0.05,
                                              params={"loss": "focal", "focal_gamma": 1.0}))
register("DS20_BR_MS_AUG_LS_CR", ModelConfig(name="DS20_BR_MS_AUG_LS_CR", description="DS4 + CE + rank loss (lambda 0.5, margin 0.3)",
                                             encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                             augment="light", label_smoothing=0.05,
                                             params={"loss": "ce_rank", "rank_lambda": 0.5, "rank_margin": 0.3}))
register("DS21_BR_MS_AUG_LS_RANK", ModelConfig(name="DS21_BR_MS_AUG_LS_RANK", description="DS4 + pure pairwise rank loss (margin 0.3)",
                                               encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                               augment="light", label_smoothing=0.05,
                                               params={"loss": "rank", "rank_margin": 0.3}))
register("DS22_BR_MS_AUG_LS_CR1", ModelConfig(name="DS22_BR_MS_AUG_LS_CR1", description="DS4 + CE + rank loss (lambda 1.0, margin 0.5)",
                                              encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                              augment="light", label_smoothing=0.05,
                                              params={"loss": "ce_rank", "rank_lambda": 1.0, "rank_margin": 0.5}))
register("DS23_BR_MS_AUG_LS_SIG", ModelConfig(name="DS23_BR_MS_AUG_LS_SIG", description="DS4 + subject_agg=sig_mean",
                                              encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                              augment="light", label_smoothing=0.05,
                                              params={"subject_agg": "sig_mean"}))
register("DS24_BR_MS_AUG_LS_TRIM", ModelConfig(name="DS24_BR_MS_AUG_LS_TRIM", description="DS4 + subject_agg=trim15",
                                               encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                               augment="light", label_smoothing=0.05,
                                               params={"subject_agg": "trim15"}))
register("DS4_MS192_AUG_LS", ModelConfig(name="DS4_MS192_AUG_LS", description="multi-scale emb 192 + aug + LS",
                                         encoder="multi_scale", embedding_dim=192, augment="light",
                                         label_smoothing=0.05))
register("DS4_BAND192_AUG_LS", ModelConfig(name="DS4_BAND192_AUG_LS", description="band 15ch emb 192 + aug + LS",
                                           encoder="multi_scale", embedding_dim=192, input_transform="band",
                                           augment="light", label_smoothing=0.05))
register("DS4_MS_ATTN_AUG_LS", ModelConfig(name="DS4_MS_ATTN_AUG_LS", description="multi-scale attn pool + aug + LS",
                                           encoder="multi_scale", embedding_dim=96, augment="light",
                                           label_smoothing=0.05, params={"pooling": "attn"}))
register("DS4_MS_DEEP_AUG_LS", ModelConfig(name="DS4_MS_DEEP_AUG_LS", description="multi-scale depth2 + aug + LS",
                                           encoder="multi_scale", embedding_dim=96, augment="light",
                                           label_smoothing=0.05, params={"depth": 2}))
register("DS4_BR_MS_ATTN_AUG_LS", ModelConfig(name="DS4_BR_MS_ATTN_AUG_LS", description="band+raw attn + aug + LS",
                                              encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                              augment="light", label_smoothing=0.05, params={"pooling": "attn"}))

# --- batch5: dual-branch fusion (raw encoder + band encoder) on band_raw input ---
register("DS5_BR_FUSE", ModelConfig(name="DS5_BR_FUSE", description="dual-branch raw+band fusion (baseline)",
                                    encoder="dual_branch", embedding_dim=96, input_transform="band_raw"))
register("DS5_BR_FUSE_AUG_LS", ModelConfig(name="DS5_BR_FUSE_AUG_LS", description="dual-branch fusion + aug + LS",
                                           encoder="dual_branch", embedding_dim=96, input_transform="band_raw",
                                           augment="light", label_smoothing=0.05))
register("DS5_BR_FUSE_ATTN_AUG_LS", ModelConfig(name="DS5_BR_FUSE_ATTN_AUG_LS", description="dual-branch attn + aug + LS",
                                                encoder="dual_branch", embedding_dim=96, input_transform="band_raw",
                                                augment="light", label_smoothing=0.05, params={"pooling": "attn"}))

# --- batch6: 8-band variants (finer frequency resolution) ---
register("DS6_8BAND", ModelConfig(name="DS6_8BAND", description="8-band 24ch baseline",
                                  encoder="multi_scale", embedding_dim=96, input_transform="band", n_bands=8))
register("DS6_8BAND_AUG_LS", ModelConfig(name="DS6_8BAND_AUG_LS", description="8-band 24ch + aug + LS",
                                         encoder="multi_scale", embedding_dim=96, input_transform="band",
                                         n_bands=8, augment="light", label_smoothing=0.05))
register("DS6_8BR_AUG_LS", ModelConfig(name="DS6_8BR_AUG_LS", description="8-band+raw 27ch + aug + LS",
                                       encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                       n_bands=8, augment="light", label_smoothing=0.05))

# --- batch7: frequency-domain augmentation ---
register("DS7_BR_AUGF_LS", ModelConfig(name="DS7_BR_AUGF_LS", description="band+raw 18ch + freq-aug + LS",
                                       encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                       augment="freq", label_smoothing=0.05))

# --- DS10: adversarial subject de-identification (GRL) on top of DS7 ---
register("DS10_BR_AUGF_LS_ADA1", ModelConfig(name="DS10_BR_AUGF_LS_ADA1", description="DS7 + subject-adv GRL lambda=0.1",
                                             encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                             augment="freq", label_smoothing=0.05,
                                             params={"adv_subject": True, "adv_lambda": 0.1}))
register("DS10_BR_AUGF_LS_ADA5", ModelConfig(name="DS10_BR_AUGF_LS_ADA5", description="DS7 + subject-adv GRL lambda=0.5",
                                             encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                             augment="freq", label_smoothing=0.05,
                                             params={"adv_subject": True, "adv_lambda": 0.5}))
register("DS7_8BR_AUGF_LS", ModelConfig(name="DS7_8BR_AUGF_LS", description="8-band+raw 27ch + freq-aug + LS",
                                        encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                        n_bands=8, augment="freq", label_smoothing=0.05))
register("DS7_BAND_AUGF_LS", ModelConfig(name="DS7_BAND_AUGF_LS", description="band 15ch + freq-aug + LS",
                                         encoder="multi_scale", embedding_dim=96, input_transform="band",
                                         augment="freq", label_smoothing=0.05))

# --- batch8: lr-ensemble final prediction / head capacity variants ---
register("DS8_ENS", ModelConfig(name="DS8_ENS", description="DS4 + lr-ensemble final predict",
                                encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                augment="light", label_smoothing=0.05, params={"lr_ensemble": True}))
register("DS8_HEAD32", ModelConfig(name="DS8_HEAD32", description="DS4 + 32-dim head + dropout 0.1",
                                   encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                   augment="light", label_smoothing=0.05, dropout=0.1,
                                   params={"head_hidden": 32}))
register("DS8_8BR_ENS", ModelConfig(name="DS8_8BR_ENS", description="8-band+raw + aug + LS + lr-ensemble",
                                    encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                    n_bands=8, augment="light", label_smoothing=0.05,
                                    params={"lr_ensemble": True}))

# --- batch9: frequency-aug + capacity / spectral fusion variants ---
register("DS9_BR_AUGF_LS_STFT", ModelConfig(name="DS9_BR_AUGF_LS_STFT",
                                            description="band+raw + freq-aug + LS + log-STFT branch",
                                            encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                            augment="freq", label_smoothing=0.05, spectral="stft"))
register("DS9_BR_AUGF_LS_DP", ModelConfig(name="DS9_BR_AUGF_LS_DP",
                                          description="band+raw + freq-aug + LS + head dropout 0.15",
                                          encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                          augment="freq", label_smoothing=0.05, dropout=0.15))
register("DS9_BR_AUGF_LS_CAL", ModelConfig(name="DS9_BR_AUGF_LS_CAL",
                                           description="band+raw + freq-aug + LS (use --calibrate at formal)",
                                           encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                           augment="freq", label_smoothing=0.05))

# --- DS11: epoch-snapshot averaged final prediction (variance/calibration stabilizer) ---
register("DS11_BR_AUGF_LS_EA10", ModelConfig(name="DS11_BR_AUGF_LS_EA10",
                                             description="DS7 + epoch-avg(10) final predict",
                                             encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                             augment="freq", label_smoothing=0.05,
                                             params={"epoch_avg": 10}))
register("DS11_BR_AUGF_LS_EA20", ModelConfig(name="DS11_BR_AUGF_LS_EA20",
                                             description="DS7 + epoch-avg(20) final predict",
                                             encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                             augment="freq", label_smoothing=0.05,
                                             params={"epoch_avg": 20}))
register("DS11_BR_MS_AUG_LS_EA10", ModelConfig(name="DS11_BR_MS_AUG_LS_EA10",
                                               description="DS4 + epoch-avg(10) final predict",
                                               encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                               augment="light", label_smoothing=0.05,
                                               params={"epoch_avg": 10}))

# --- DS12: head-capacity / dropout combos on DS7 (bAcc direction, follow-up of DS8_HEAD32) ---
register("DS12_BR_AUGF_LS_HEAD32", ModelConfig(name="DS12_BR_AUGF_LS_HEAD32",
                                               description="DS7 + 32-dim head + dropout 0.1",
                                               encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                               augment="freq", label_smoothing=0.05, dropout=0.1,
                                               params={"head_hidden": 32}))
register("DS12_BR_AUGF_LS_DP01", ModelConfig(name="DS12_BR_AUGF_LS_DP01",
                                             description="DS7 + head dropout 0.1 (no head_hidden)",
                                             encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                             augment="freq", label_smoothing=0.05, dropout=0.1))
register("DS12_BR_AUGF_LS_HEAD32_EA10", ModelConfig(name="DS12_BR_AUGF_LS_HEAD32_EA10",
                                                    description="DS7 + head32 + dropout0.1 + epoch-avg(10)",
                                                    encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                                    augment="freq", label_smoothing=0.05, dropout=0.1,
                                                    params={"head_hidden": 32, "epoch_avg": 10}))

# --- DS13: lr-ensemble on DS7 / combos (follow-up of DS8_ENS 10-fold win) ---
register("DS13_BR_AUGF_LS_ENS", ModelConfig(name="DS13_BR_AUGF_LS_ENS",
                                            description="DS7 + lr-ensemble final predict",
                                            encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                            augment="freq", label_smoothing=0.05,
                                            params={"lr_ensemble": True}))
register("DS13_BR_AUGF_LS_ENS_EA10", ModelConfig(name="DS13_BR_AUGF_LS_ENS_EA10",
                                                 description="DS7 + lr-ensemble + epoch-avg(10)",
                                                 encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                                 augment="freq", label_smoothing=0.05,
                                                 params={"lr_ensemble": True, "epoch_avg": 10}))
register("DS13_BR_MS_AUG_LS_ENS_EA10", ModelConfig(name="DS13_BR_MS_AUG_LS_ENS_EA10",
                                                   description="DS4 + lr-ensemble + epoch-avg(10)",
                                                   encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                                   augment="light", label_smoothing=0.05,
                                                   params={"lr_ensemble": True, "epoch_avg": 10}))

# --- DS14: SWA (last-K weight averaging) + lr-ensemble tolerance filtering ---
register("DS14_BR_AUGF_LS_SWA10", ModelConfig(name="DS14_BR_AUGF_LS_SWA10",
                                              description="DS7 + SWA(10) final predict",
                                              encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                              augment="freq", label_smoothing=0.05,
                                              params={"swa_epochs": 10}))
register("DS14_BR_AUGF_LS_SWA20", ModelConfig(name="DS14_BR_AUGF_LS_SWA20",
                                              description="DS7 + SWA(20) final predict",
                                              encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                              augment="freq", label_smoothing=0.05,
                                              params={"swa_epochs": 20}))
register("DS14_BR_AUGF_LS_ENS_TOL", ModelConfig(name="DS14_BR_AUGF_LS_ENS_TOL",
                                                description="DS7 + lr-ensemble tol0.01 (validation-filtered)",
                                                encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                                augment="freq", label_smoothing=0.05,
                                                params={"lr_ensemble": True, "lr_ensemble_tol": 0.01}))
register("DS14_BR_AUGF_LS_ENS_TOL_EA10", ModelConfig(name="DS14_BR_AUGF_LS_ENS_TOL_EA10",
                                                     description="DS7 + lr-ensemble tol0.01 + epoch-avg(10)",
                                                     encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                                     augment="freq", label_smoothing=0.05,
                                                     params={"lr_ensemble": True, "lr_ensemble_tol": 0.01,
                                                             "epoch_avg": 10}))

# --- DS15: cross-channel coherence input (band_raw_coh 33ch) ---
register("DS15_BRCOH_MS_AUG_LS", ModelConfig(name="DS15_BRCOH_MS_AUG_LS",
                                             description="band+raw+coh 33ch + light aug + LS",
                                             encoder="multi_scale", embedding_dim=96, input_transform="band_raw_coh",
                                             augment="light", label_smoothing=0.05))
register("DS15_BRCOH_MS_AUGF_LS", ModelConfig(name="DS15_BRCOH_MS_AUGF_LS",
                                              description="band+raw+coh 33ch + freq-aug + LS",
                                              encoder="multi_scale", embedding_dim=96, input_transform="band_raw_coh",
                                              augment="freq", label_smoothing=0.05))
register("DS15_BRCOH_MS_AUGF_LS_SWA10", ModelConfig(name="DS15_BRCOH_MS_AUGF_LS_SWA10",
                                                    description="DS15 freq-aug + SWA(10)",
                                                    encoder="multi_scale", embedding_dim=96, input_transform="band_raw_coh",
                                                    augment="freq", label_smoothing=0.05,
                                                    params={"swa_epochs": 10}))

# --- DS16: SWA on the incumbent DS4 base (light aug) ---
register("DS16_BR_MS_AUG_LS_SWA10", ModelConfig(name="DS16_BR_MS_AUG_LS_SWA10",
                                                description="DS4 + SWA(10) final predict",
                                                encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                                augment="light", label_smoothing=0.05,
                                                params={"swa_epochs": 10}))
register("DS16_BR_MS_AUG_LS_SWA20", ModelConfig(name="DS16_BR_MS_AUG_LS_SWA20",
                                                description="DS4 + SWA(20) final predict",
                                                encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                                augment="light", label_smoothing=0.05,
                                                params={"swa_epochs": 20}))
register("DS16_BR_MS_AUG_LS_EA10_SWA10", ModelConfig(name="DS16_BR_MS_AUG_LS_EA10_SWA10",
                                                     description="DS4 + EA10 + SWA10 final predict",
                                                     encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                                     augment="light", label_smoothing=0.05,
                                                     params={"epoch_avg": 10, "swa_epochs": 10}))

# --- DS18: EA20 on the incumbent DS4 base ---
register("DS18_BR_MS_AUG_LS_EA20", ModelConfig(name="DS18_BR_MS_AUG_LS_EA20",
                                               description="DS4 + epoch-avg(20) final predict",
                                               encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                               augment="light", label_smoothing=0.05,
                                               params={"epoch_avg": 20}))

# --- DS19: mild frequency augmentation on DS4 (DS7 freq-aug was too aggressive on 31-fold) ---
register("DS19_BR_MS_AUGF_LS", ModelConfig(name="DS19_BR_MS_AUGF_LS",
                                           description="DS4 + freq_light aug (1-6 bin masks, weak noise)",
                                           encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                           augment="freq_light", label_smoothing=0.05))
# --- Round-16: ds003490-specific one-shot hypotheses (08-13) ---
# 490 = 14847 windows / 50 subjects vs 778 = 3022 windows / 31 subjects.
# H1 (capacity): 5x more windows -> emb96 (tuned on 778) may be under-capacity.
# H2 (regularization): 6x more windows per subject -> stronger identity signal -> wd 3e-3 may be needed.
register("DS51_BR_MS_AUG_LS_WD3_E128", ModelConfig(name="DS51_BR_MS_AUG_LS_WD3_E128",
                                                    description="DS28 + embedding 128 on ds003490 from scratch (capacity hypothesis: 5x data)",
                                                    encoder="multi_scale", embedding_dim=128, input_transform="band_raw",
                                                    augment="light", label_smoothing=0.05,
                                                    params={"wd": 1e-3}))
register("DS52_BR_MS_AUG_LS_WD33", ModelConfig(name="DS52_BR_MS_AUG_LS_WD33",
                                                description="DS28 + weight decay 3e-3 on ds003490 from scratch (regularization hypothesis: 6x windows/subject)",
                                                encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                                augment="light", label_smoothing=0.05,
                                                params={"wd": 3e-3}))
# --- Round-19: learned attention subject aggregation (MIL) on ds003490 (08-14) ---
# External evidence (parallel project E45, implementation reviewed): attention-MIL
# subject aggregation improved 490 AUC (+0.0528) with subject-primary loss
# (1.0 subj CE + 0.25 window CE), tanh attention 96->48->1, 32-window train bags.
# Our independent variants (pre-registered, frozen at launch):
#   DS53: window-primary hybrid (window CE + 1.0 subj CE), GELU attention, full-subject bags
#   DS54: E45-style subject-primary (1.0 subj + 0.25 window CE), tanh attention, full-subject bags
# Both use CONSISTENT attention aggregation for inner selection and final test
# (deliberate deviation from E45's window-mean inner selection, per review advice).
register("DS53_BR_MS_AUG_LS_WD3_ATTN", ModelConfig(name="DS53_BR_MS_AUG_LS_WD3_ATTN",
                                                    description="DS28 + attention MIL (window CE + 1.0 subject CE, GELU attn 96->32->1)",
                                                    encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                                    augment="light", label_smoothing=0.05,
                                                    params={"wd": 1e-3, "subject_attn": True, "subject_attn_lambda": 1.0,
                                                            "window_ce_weight": 1.0, "attn_hidden": 32, "attn_act": "gelu"}))
register("DS54_BR_MS_AUG_LS_WD3_ATTNE", ModelConfig(name="DS54_BR_MS_AUG_LS_WD3_ATTNE",
                                                     description="E45-style attention MIL (1.0 subject CE + 0.25 window CE, tanh attn 96->48->1)",
                                                     encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                                     augment="light", label_smoothing=0.05,
                                                     params={"wd": 1e-3, "subject_attn": True, "subject_attn_lambda": 1.0,
                                                             "window_ce_weight": 0.25, "attn_hidden": 48, "attn_act": "tanh"}))
# --- Round-20: relative band-power ratio branch (cohort absolute-power offset removal) ---
# Motivation: audit showed cohort absolute power offset between 778 and 490; the
# two offset-removal normalizations (channel_zscore/subject_chmean) failed because
# they discard threshold-side absolute info. Band RATIOS keep spectral SHAPE only.
register("DS55_BR_MS_AUG_LS_WD3_RATIO", ModelConfig(name="DS55_BR_MS_AUG_LS_WD3_RATIO",
                                                     description="DS28 + relative band-power ratio branch (log10 theta/delta, alpha/delta, beta/theta, gamma/beta per channel)",
                                                     encoder="multi_scale", embedding_dim=96, input_transform="band_raw",
                                                     augment="light", label_smoothing=0.05,
                                                     spectral="band_ratio",
                                                     params={"wd": 1e-3}))
# --- Round-21: long-depthwise (T5-style) architecture x DS28 frozen hyperparams (08-17) ---
# Motivation: frozen E32 T5 predictions lead ds003490 bAcc (5-seed mean 0.74 vs DS28 0.71-0.72).
# This candidate trains the long-temporal architecture inside OUR frozen protocol
# (band_raw + wd 1e-3 + LS 0.05 + light aug) from scratch on ds003490.
register("DS56_BR_LD_AUG_LS_WD3", ModelConfig(name="DS56_BR_LD_AUG_LS_WD3",
                                               description="T5-style long-depthwise encoder (kernels 15/31/63/127, emb 192) + DS28 frozen hyperparams",
                                               encoder="long_depthwise", embedding_dim=192, input_transform="band_raw",
                                               augment="light", label_smoothing=0.05,
                                               params={"wd": 1e-3}))
# --- Round-22: hybrid encoder = multi-scale short kernels + long depthwise kernels ---
# Single-model embodiment of the cross-architecture complementarity proven by the
# fixed fusion (0.7936/0.7600 on 490). Paper-protagonist candidate: must first pass
# the 778 S gate (2 seeds vs DS28 same-seed), then 5 seeds + W + 490.
register("DS57_BR_HYB_AUG_LS_WD3", ModelConfig(name="DS57_BR_HYB_AUG_LS_WD3",
                                                description="hybrid encoder: multi-scale (7/15/31) + long-depthwise (15/31/63/127), emb 96+96 fused to 96",
                                                encoder="hybrid", embedding_dim=96, input_transform="band_raw",
                                                augment="light", label_smoothing=0.05,
                                                params={"wd": 1e-3}))
# --- Round-23: EEGNet fair baseline (Lawhern 2018), raw 3ch, frozen training protocol ---
register("DS58_EEGNET", ModelConfig(name="DS58_EEGNET",
                                     description="EEGNet baseline (F1=8 D=2 F2=16, raw C3/Cz/C4) under the frozen training protocol",
                                     encoder="eegnet", input_transform="raw",
                                     augment="light", label_smoothing=0.05,
                                     params={"wd": 1e-3}))
