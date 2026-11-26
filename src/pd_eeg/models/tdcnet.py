"""TDCNet representation and contrastive objectives (Xu et al., 2026)."""

from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass
from typing import Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


MODEL_CONFIG = {
    "input_channels": 3,
    "sequence_length": 512,
    "embedding_dim": 128,
    "residual_dilations": (1, 2, 4, 8, 16),
    "temporal_expert_kernels": (1, 2, 4, 8, 16, 32, 64, 128),
    "normalization_groups": 8,
    "queue_size": 256,
    "temperature": 0.07,
    "momentum": 0.999,
    "noise_fraction_min": 0.01,
    "noise_fraction_max": 0.10,
    "subjects_per_batch": 8,
    "windows_per_subject": 4,
}

TRAINING_CONFIG = {
    "epochs": 50,
    "batch_size": 32,
    "optimizer": "SGD",
    "learning_rate": 0.01,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "gradient_clip_norm": 1.0,
    "temporal_loss_weight": 0.35,
    "frequency_loss_weight": 0.50,
    "subject_loss_weight": 0.15,
}

SVM_C_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, np.inf)


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


class ResidualDilatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, groups: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=int(dilation),
            dilation=int(dilation),
        )
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=int(dilation),
            dilation=int(dilation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.gelu(self.norm1(x)))
        x = self.conv2(F.gelu(self.norm2(x)))
        return residual + x


class TemporalEvolutionUnit(nn.Module):
    def __init__(self, channels: int, kernels: Sequence[int]):
        super().__init__()
        self.kernels = tuple(int(value) for value in kernels)
        self.experts = nn.ModuleList(
            [
                nn.Conv1d(channels, channels, kernel_size=kernel, padding=0)
                for kernel in self.kernels
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = None
        for kernel, expert in zip(self.kernels, self.experts):
            current = expert(F.pad(x, (kernel - 1, 0)))
            total = current if total is None else total + current
        return total / float(len(self.experts))


class LearnableFourierLayer(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.frequency = nn.Parameter(torch.linspace(0.0, np.pi, embedding_dim))
        self.phase = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("Fourier input must be [batch, time, embedding].")
        time = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)
        argument = (
            time[None, :, None] * self.frequency[None, None, :]
            + self.phase[None, None, :]
        )
        return x * torch.cos(argument), x * torch.sin(argument)


class TDCEncoder(nn.Module):
    def __init__(self, config: dict = MODEL_CONFIG):
        super().__init__()
        channels = int(config["embedding_dim"])
        self.input_projection = nn.Linear(int(config["input_channels"]), channels)
        self.residual_blocks = nn.ModuleList(
            [
                ResidualDilatedBlock(
                    channels,
                    int(dilation),
                    int(config["normalization_groups"]),
                )
                for dilation in config["residual_dilations"]
            ]
        )
        self.temporal_unit = TemporalEvolutionUnit(
            channels,
            config["temporal_expert_kernels"],
        )
        self.frequency_unit = LearnableFourierLayer(channels)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("encoder input must be [batch, channel, time].")
        hidden = self.input_projection(x.transpose(1, 2)).transpose(1, 2)
        for block in self.residual_blocks:
            hidden = block(hidden)
        temporal = self.temporal_unit(hidden).transpose(1, 2)
        spectral = hidden.transpose(1, 2)
        real, imag = self.frequency_unit(spectral)
        psd = real.square() + imag.square()
        phase = torch.atan2(imag, real + 1e-8)
        fused = torch.cat([temporal, psd, phase], dim=-1)
        return temporal, psd, phase, fused


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TDCContrastiveModel(nn.Module):
    def __init__(self, config: dict = MODEL_CONFIG):
        super().__init__()
        self.config = dict(config)
        embedding_dim = int(config["embedding_dim"])
        self.online_encoder = TDCEncoder(config)
        self.momentum_encoder = copy.deepcopy(self.online_encoder)
        self.online_temporal_head = ProjectionHead(embedding_dim)
        self.online_psd_head = ProjectionHead(embedding_dim)
        self.online_phase_head = ProjectionHead(embedding_dim)
        self.momentum_temporal_head = copy.deepcopy(self.online_temporal_head)
        self.momentum_psd_head = copy.deepcopy(self.online_psd_head)
        self.momentum_phase_head = copy.deepcopy(self.online_phase_head)
        self._freeze_momentum_parameters()
        queue = torch.randn(embedding_dim, int(config["queue_size"]))
        self.register_buffer("temporal_queue", F.normalize(queue, dim=0))
        self.register_buffer("queue_pointer", torch.zeros(1, dtype=torch.long))

    def _freeze_momentum_parameters(self) -> None:
        modules = (
            self.momentum_encoder,
            self.momentum_temporal_head,
            self.momentum_psd_head,
            self.momentum_phase_head,
        )
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = False

    def online_trainable_parameters(self):
        modules = (
            self.online_encoder,
            self.online_temporal_head,
            self.online_psd_head,
            self.online_phase_head,
        )
        for module in modules:
            yield from module.parameters()

    @torch.no_grad()
    def momentum_update(self) -> None:
        coefficient = float(self.config["momentum"])
        pairs = (
            (self.online_encoder, self.momentum_encoder),
            (self.online_temporal_head, self.momentum_temporal_head),
            (self.online_psd_head, self.momentum_psd_head),
            (self.online_phase_head, self.momentum_phase_head),
        )
        for online, momentum in pairs:
            for source, target in zip(online.parameters(), momentum.parameters()):
                target.mul_(coefficient).add_(source, alpha=1.0 - coefficient)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        keys = F.normalize(keys.detach(), dim=1)
        queue_size = int(self.config["queue_size"])
        batch_size = int(keys.shape[0])
        pointer = int(self.queue_pointer.item())
        first = min(batch_size, queue_size - pointer)
        self.temporal_queue[:, pointer : pointer + first] = keys[:first].T
        remaining = batch_size - first
        if remaining:
            self.temporal_queue[:, :remaining] = keys[first:].T
        self.queue_pointer[0] = (pointer + batch_size) % queue_size

    def clean_embedding(self, x: torch.Tensor) -> torch.Tensor:
        _, _, _, fused = self.online_encoder(x)
        return fused.mean(dim=1)


def _augment_view(
    x: torch.Tensor,
    generator: torch.Generator,
    config: dict,
) -> torch.Tensor:
    fraction = torch.empty((), device=x.device).uniform_(
        float(config["noise_fraction_min"]),
        float(config["noise_fraction_max"]),
        generator=generator,
    )
    scale = x.std(dim=2, keepdim=True, unbiased=False).clamp_min(1e-6)
    noise = torch.randn(
        x.shape,
        dtype=x.dtype,
        device=x.device,
        generator=generator,
    )
    return x + noise * scale * fraction


def _moco_loss(
    query: torch.Tensor,
    key: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    query = F.normalize(query, dim=1)
    key = F.normalize(key.detach(), dim=1)
    positive = torch.sum(query * key, dim=1, keepdim=True)
    negative = query @ negatives.detach()
    logits = torch.cat([positive, negative], dim=1) / float(temperature)
    labels = torch.zeros(len(query), dtype=torch.long, device=query.device)
    return F.cross_entropy(logits, labels)


def _symmetric_batch_contrastive_loss(
    query: torch.Tensor,
    key: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    query = F.normalize(query, dim=1)
    key = F.normalize(key.detach(), dim=1)
    logits = query @ key.T / float(temperature)
    labels = torch.arange(len(query), dtype=torch.long, device=query.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.T, labels)
    )


def subject_structured_loss(
    embeddings: torch.Tensor,
    subject_codes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    unique_codes = torch.unique(subject_codes, sorted=True)
    within_values = []
    centroids = []
    for code in unique_codes:
        current = embeddings[subject_codes == code]
        if len(current) < 2:
            continue
        distances = torch.pdist(current, p=2)
        within_values.append(distances.mean())
        centroids.append(current.mean(dim=0))
    if len(within_values) < 2:
        zero = embeddings.sum() * 0.0
        return zero, zero, zero
    within = torch.stack(within_values).mean()
    between = torch.pdist(torch.stack(centroids), p=2).mean()
    return F.relu(within - between), within, between


def contrastive_loss_components(
    model: TDCContrastiveModel,
    batch: torch.Tensor,
    subject_codes: torch.Tensor,
    generator: torch.Generator,
    training_config: dict = TRAINING_CONFIG,
) -> dict[str, torch.Tensor]:
    view_online = _augment_view(batch, generator, model.config)
    view_momentum = _augment_view(batch, generator, model.config)
    q_temporal, q_psd, q_phase, q_fused = model.online_encoder(view_online)
    with torch.no_grad():
        model.momentum_update()
        k_temporal, k_psd, k_phase, _ = model.momentum_encoder(view_momentum)
    q_temporal = model.online_temporal_head(q_temporal.mean(dim=1))
    q_psd = model.online_psd_head(q_psd.mean(dim=1))
    q_phase = model.online_phase_head(q_phase.mean(dim=1))
    with torch.no_grad():
        k_temporal = model.momentum_temporal_head(k_temporal.mean(dim=1))
        k_psd = model.momentum_psd_head(k_psd.mean(dim=1))
        k_phase = model.momentum_phase_head(k_phase.mean(dim=1))
    temporal_loss = _moco_loss(
        q_temporal,
        k_temporal,
        model.temporal_queue.clone(),
        float(model.config["temperature"]),
    )
    psd_loss = _symmetric_batch_contrastive_loss(
        q_psd, k_psd, float(model.config["temperature"])
    )
    phase_loss = _symmetric_batch_contrastive_loss(
        q_phase, k_phase, float(model.config["temperature"])
    )
    orthogonality = 0.5 * (
        F.cosine_similarity(q_psd, q_phase, dim=1).abs().mean()
        + F.cosine_similarity(k_psd, k_phase, dim=1).abs().mean()
    )
    frequency_loss = 0.5 * (psd_loss + phase_loss) + orthogonality
    subject_loss, within_distance, between_distance = subject_structured_loss(
        q_fused.mean(dim=1),
        subject_codes,
    )
    total = (
        float(training_config["temporal_loss_weight"]) * temporal_loss
        + float(training_config["frequency_loss_weight"]) * frequency_loss
        + float(training_config["subject_loss_weight"]) * subject_loss
    )
    model.dequeue_and_enqueue(k_temporal)
    return {
        "total": total,
        "temporal": temporal_loss,
        "frequency": frequency_loss,
        "subject": subject_loss,
        "orthogonality": orthogonality,
        "within_distance": within_distance,
        "between_distance": between_distance,
    }


@dataclass(frozen=True)
class SubjectBatchPlan:
    batch_indices: np.ndarray
    batch_subject_codes: np.ndarray


def build_subject_aware_epoch_batches(
    groups: Sequence,
    epoch: int,
    seed: int,
    config: dict = MODEL_CONFIG,
) -> list[SubjectBatchPlan]:
    groups = np.asarray(groups).astype(str)
    unique_subjects = np.array(sorted(np.unique(groups).tolist()), dtype=object)
    subjects_per_batch = min(
        int(config["subjects_per_batch"]),
        len(unique_subjects),
    )
    windows_per_subject = int(config["windows_per_subject"])
    if subjects_per_batch < 2 or windows_per_subject < 2:
        raise ValueError("subject-aware batching requires at least two subjects and windows.")
    rng = np.random.default_rng(int(seed) + 1009 * int(epoch))
    subject_to_indices = {
        subject: np.flatnonzero(groups == subject).astype(np.int64)
        for subject in unique_subjects
    }
    n_batches = max(
        1,
        len(groups) // (subjects_per_batch * windows_per_subject),
    )
    plans = []
    for _ in range(n_batches):
        selected_subjects = rng.choice(
            unique_subjects,
            size=subjects_per_batch,
            replace=False,
        )
        indices = []
        codes = []
        for code, subject in enumerate(selected_subjects):
            pool = subject_to_indices[str(subject)]
            chosen = rng.choice(
                pool,
                size=windows_per_subject,
                replace=len(pool) < windows_per_subject,
            )
            indices.extend(chosen.tolist())
            codes.extend([code] * windows_per_subject)
        order = rng.permutation(len(indices))
        plans.append(
            SubjectBatchPlan(
                batch_indices=np.asarray(indices, dtype=np.int64)[order],
                batch_subject_codes=np.asarray(codes, dtype=np.int64)[order],
            )
        )
    return plans


def _assert_finite_model(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(f"model parameter is non-finite: {name}.")


def train_tdcnet_representation(
    train_windows: np.ndarray,
    train_groups: Sequence,
    device: torch.device,
    epochs: int,
    seed: int,
    model_config: dict = MODEL_CONFIG,
    training_config: dict = TRAINING_CONFIG,
) -> tuple[TDCContrastiveModel, list[dict]]:
    windows = np.asarray(train_windows, dtype=np.float32)
    groups = np.asarray(train_groups).astype(str)
    if windows.ndim != 3 or windows.shape[1:] != (
        int(model_config["input_channels"]),
        int(model_config["sequence_length"]),
    ):
        raise ValueError("training windows have an invalid shape.")
    if len(windows) != len(groups) or len(np.unique(groups)) < 2:
        raise ValueError("training subject metadata are invalid.")
    if not np.isfinite(windows).all():
        raise ValueError("training windows contain NaN or Inf.")
    set_deterministic_seed(seed)
    model = TDCContrastiveModel(model_config).to(device)
    trainable_parameters = list(model.online_trainable_parameters())
    optimizer = torch.optim.SGD(
        trainable_parameters,
        lr=float(training_config["learning_rate"]),
        momentum=float(training_config["momentum"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    tensor = torch.as_tensor(windows, dtype=torch.float32, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    history = []
    component_names = (
        "total",
        "temporal",
        "frequency",
        "subject",
        "orthogonality",
        "within_distance",
        "between_distance",
    )
    for epoch in range(1, int(epochs) + 1):
        model.train()
        sums = {name: 0.0 for name in component_names}
        gradient_norm_sum = 0.0
        count = 0
        plans = build_subject_aware_epoch_batches(
            groups,
            epoch=epoch,
            seed=seed,
            config=model_config,
        )
        for plan in plans:
            batch_indices = torch.as_tensor(
                plan.batch_indices,
                dtype=torch.long,
                device=device,
            )
            subject_codes = torch.as_tensor(
                plan.batch_subject_codes,
                dtype=torch.long,
                device=device,
            )
            batch = tensor.index_select(0, batch_indices)
            optimizer.zero_grad(set_to_none=True)
            components = contrastive_loss_components(
                model,
                batch,
                subject_codes,
                generator,
                training_config,
            )
            if not torch.isfinite(components["total"]):
                values = {
                    name: float(components[name].detach().item())
                    for name in component_names
                }
                raise FloatingPointError(
                    f"contrastive loss is NaN or Inf at epoch {epoch}, "
                    f"batch {count + 1}: {values}."
                )
            components["total"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=float(training_config["gradient_clip_norm"]),
                error_if_nonfinite=True,
            )
            optimizer.step()
            for name in component_names:
                sums[name] += float(components[name].detach().item())
            gradient_norm_sum += float(gradient_norm.detach().item())
            count += 1
        if count != len(plans) or count <= 0:
            raise RuntimeError("epoch batch accounting failed.")
        _assert_finite_model(model)
        row = {"epoch": int(epoch), "n_batches": int(count)}
        row.update({f"{name}_loss": sums[name] / count for name in component_names})
        row["preclip_gradient_norm"] = gradient_norm_sum / count
        if not np.isfinite(np.asarray(list(row.values()), dtype=float)).all():
            raise FloatingPointError("epoch history contains NaN or Inf.")
        history.append(row)
    return model, history


@torch.no_grad()
def encode_tdcnet_embeddings(
    model: TDCContrastiveModel,
    windows: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    array = np.asarray(windows, dtype=np.float32)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError("embedding input is invalid.")
    model.eval()
    outputs = []
    for start in range(0, len(array), int(batch_size)):
        batch = torch.as_tensor(
            array[start : start + int(batch_size)],
            dtype=torch.float32,
            device=device,
        )
        outputs.append(model.clean_embedding(batch).cpu().numpy())
    result = np.concatenate(outputs, axis=0).astype(np.float32, copy=False)
    expected_dim = int(model.config["embedding_dim"]) * 3
    if result.shape != (len(array), expected_dim) or not np.isfinite(result).all():
        raise RuntimeError("encoded embeddings are invalid.")
    return result


def fit_rbf_svm(
    train_embeddings: np.ndarray,
    train_labels: Sequence,
    c_value: float,
    seed: int,
) -> Pipeline:
    embeddings = np.asarray(train_embeddings, dtype=np.float64)
    labels = np.asarray(train_labels, dtype=np.int64).reshape(-1)
    if len(embeddings) != len(labels) or set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("SVM training data are invalid.")
    if not np.isfinite(embeddings).all():
        raise ValueError("SVM embeddings contain NaN or Inf.")
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    C=float(c_value),
                    kernel="rbf",
                    gamma="scale",
                    class_weight=None,
                    probability=True,
                    random_state=int(seed),
                    cache_size=1024,
                    max_iter=10_000_000,
                ),
            ),
        ]
    )
    model.fit(embeddings, labels)
    return model


def positive_class_probabilities(
    model: Pipeline,
    embeddings: np.ndarray,
) -> np.ndarray:
    probabilities = model.predict_proba(
        np.asarray(embeddings, dtype=np.float64)
    )
    classes = np.asarray(model.named_steps["svc"].classes_)
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1:
        raise RuntimeError("SVM does not expose exactly one PD probability.")
    result = probabilities[:, int(matches[0])].astype(np.float64, copy=False)
    if not np.isfinite(result).all() or np.any((result < 0.0) | (result > 1.0)):
        raise RuntimeError("SVM probabilities are invalid.")
    return result


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def parameter_counts(model: TDCContrastiveModel) -> tuple[int, int]:
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    return int(trainable), int(total)
