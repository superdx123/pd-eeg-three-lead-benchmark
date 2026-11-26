"""Welch-PSD autoencoder and RBF classifier (Jibon et al., 2024)."""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits


PSD_CONFIG = {
    "sampling_rate_hz": 256.0,
    "nperseg": 256,
    "noverlap": 128,
    "nfft": 256,
    "fmin_hz": 1.0,
    "fmax_hz": 40.0,
    "detrend": "constant",
    "window": "periodic_hann",
    "normalization": "per_window_per_channel_max",
}

AE_CONFIG = {
    "input_units": 120,
    "encoder_units": [400, 200, 100, 50],
    "decoder_units": [100, 100, 200, 400, 120],
    "batch_normalization": True,
    "hidden_activation": "relu",
    "output_activation": "relu",
}

RBF_CONFIG = {
    "input_units": 50,
    "n_centers": 100,
    "n_classes": 2,
    "activation": "gaussian",
    "center_initialization": "kmeans",
    "kmeans_n_init": 10,
    "kmeans_algorithm": "lloyd",
    "log_sigma_min": -8.0,
    "log_sigma_max": 8.0,
}


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


def max_normalized_welch_psd_features(
    windows: np.ndarray,
    sfreq: float,
    channel_names: Sequence[str],
) -> tuple[np.ndarray, list[str], dict]:
    X = np.asarray(windows, dtype=np.float64)
    channels = tuple(str(value) for value in channel_names)
    if X.ndim != 3 or X.shape[0] == 0:
        raise ValueError(f"expects non-empty [N, C, T] windows, found {X.shape}.")
    if X.shape[1] != len(channels):
        raise ValueError("PSD channel-name count does not match the data.")
    if not np.isfinite(X).all():
        raise ValueError("PSD input contains NaN or Inf.")
    if not np.isclose(float(sfreq), float(PSD_CONFIG["sampling_rate_hz"])):
        raise ValueError(f"requires sampling rate 256 Hz, found {sfreq}.")
    nperseg = int(PSD_CONFIG["nperseg"])
    noverlap = int(PSD_CONFIG["noverlap"])
    nfft = int(PSD_CONFIG["nfft"])
    if X.shape[2] != 512 or nperseg > X.shape[2] or not 0 <= noverlap < nperseg:
        raise ValueError("PSD input length or Welch configuration is invalid.")
    step = nperseg - noverlap
    starts = list(range(0, X.shape[2] - nperseg + 1, step))
    if starts != [0, 128, 256]:
        raise ValueError(f"expected Welch starts [0, 128, 256], found {starts}.")

    sample_axis = np.arange(nperseg, dtype=np.float64)
    taper = 0.5 - 0.5 * np.cos(2.0 * np.pi * sample_axis / nperseg)
    scale = float(sfreq) * float(np.square(taper).sum())
    accumulated = np.zeros((X.shape[0], X.shape[1], nfft // 2 + 1), dtype=np.float64)
    for start in starts:
        segment = X[:, :, start : start + nperseg]
        segment = segment - segment.mean(axis=-1, keepdims=True)
        spectrum = np.fft.rfft(segment * taper.reshape(1, 1, -1), n=nfft, axis=-1)
        density = np.square(np.abs(spectrum)) / scale
        density[..., 1:-1] *= 2.0
        accumulated += density
    density = accumulated / float(len(starts))
    frequencies = np.fft.rfftfreq(nfft, d=1.0 / float(sfreq))
    mask = (
        (frequencies >= float(PSD_CONFIG["fmin_hz"]))
        & (frequencies <= float(PSD_CONFIG["fmax_hz"]))
    )
    selected_frequencies = frequencies[mask]
    if not np.array_equal(selected_frequencies, np.arange(1.0, 41.0, dtype=np.float64)):
        raise ValueError("PSD frequency bins are inconsistent with the configured frequency range.")
    selected = density[..., mask]
    maxima = selected.max(axis=-1, keepdims=True)
    if not np.isfinite(maxima).all() or np.any(maxima <= 0.0):
        raise ValueError("PSD contains a non-positive per-channel maximum.")
    normalized = selected / maxima
    features = np.asarray(normalized.reshape(X.shape[0], -1), dtype=np.float32)
    if features.shape[1] != int(AE_CONFIG["input_units"]):
        raise RuntimeError(f"expected 120 PSD features, found {features.shape[1]}.")
    if (
        not np.isfinite(features).all()
        or np.any(features < 0.0)
        or np.any(features > 1.0 + 1e-7)
    ):
        raise ValueError("normalized PSD features are invalid.")
    names = [
        f"{channel}_max_normalized_welch_psd_{frequency:.0f}Hz"
        for channel in channels
        for frequency in selected_frequencies
    ]
    metadata = {
        **PSD_CONFIG,
        "channel_names": list(channels),
        "segment_starts": starts,
        "selected_frequencies_hz": [float(value) for value in selected_frequencies],
        "n_features": int(features.shape[1]),
    }
    return features, names, metadata


def feature_payload_sha256(features: np.ndarray, metadata: dict) -> str:
    array = np.ascontiguousarray(np.asarray(features, dtype="<f4"))
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "ascii"
        )
    )
    return digest.hexdigest()


class PSDAutoencoder(nn.Module):
    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg = dict(AE_CONFIG if config is None else config)
        encoder_units = [int(value) for value in cfg["encoder_units"]]
        decoder_units = [int(value) for value in cfg["decoder_units"]]
        encoder_layers: list[nn.Module] = []
        previous = int(cfg["input_units"])
        for units in encoder_units:
            encoder_layers.extend(
                [nn.Linear(previous, units), nn.BatchNorm1d(units), nn.ReLU()]
            )
            previous = units
        decoder_layers: list[nn.Module] = []
        for index, units in enumerate(decoder_units):
            decoder_layers.append(nn.Linear(previous, units))
            if index < len(decoder_units) - 1:
                decoder_layers.extend([nn.BatchNorm1d(units), nn.ReLU()])
            else:
                decoder_layers.append(nn.ReLU())
            previous = units
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(features))


class TrainableRBFNN(nn.Module):
    def __init__(self, initial_centers: np.ndarray, initial_sigma: float):
        super().__init__()
        centers = np.asarray(initial_centers, dtype=np.float32)
        if centers.shape != (
            int(RBF_CONFIG["n_centers"]),
            int(RBF_CONFIG["input_units"]),
        ):
            raise ValueError(f"RBF center shape is invalid: {centers.shape}.")
        if not np.isfinite(centers).all():
            raise ValueError("RBF centers contain NaN or Inf.")
        if not np.isfinite(initial_sigma) or float(initial_sigma) <= 0.0:
            raise ValueError("RBF initial sigma must be positive and finite.")
        self.centers = nn.Parameter(torch.as_tensor(centers, dtype=torch.float32))
        self.log_sigma = nn.Parameter(
            torch.full(
                (int(RBF_CONFIG["n_centers"]),),
                float(np.log(float(initial_sigma))),
                dtype=torch.float32,
            )
        )
        self.classifier = nn.Linear(
            int(RBF_CONFIG["n_centers"]), int(RBF_CONFIG["n_classes"])
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        differences = latent.unsqueeze(1) - self.centers.unsqueeze(0)
        squared_distance = differences.square().sum(dim=2)
        sigma = torch.exp(
            torch.clamp(
                self.log_sigma,
                min=float(RBF_CONFIG["log_sigma_min"]),
                max=float(RBF_CONFIG["log_sigma_max"]),
            )
        )
        activations = torch.exp(-squared_distance / (2.0 * sigma.square().unsqueeze(0)))
        return self.classifier(activations)


def _epoch_batches(
    n_samples: int,
    batch_size: int,
    generator: torch.Generator,
    require_minimum_two: bool,
) -> list[torch.Tensor]:
    if n_samples < 2:
        raise ValueError("training requires at least two samples.")
    order = torch.randperm(int(n_samples), generator=generator, device="cpu")
    batches = [order[start : start + int(batch_size)] for start in range(0, n_samples, batch_size)]
    if require_minimum_two and len(batches) > 1 and len(batches[-1]) == 1:
        final_value = batches[-1]
        moved_value = batches[-2][-1:]
        batches[-2] = batches[-2][:-1]
        batches[-1] = torch.cat([moved_value, final_value])
        if len(batches[-2]) == 0:
            batches = batches[:-2] + [batches[-1]]
    flattened = torch.cat(batches)
    if len(flattened) != n_samples or len(torch.unique(flattened)) != n_samples:
        raise RuntimeError("batch construction lost or duplicated samples.")
    if require_minimum_two and any(len(batch) < 2 for batch in batches):
        raise RuntimeError("BatchNorm received a singleton training batch.")
    return batches


def _assert_finite_model(model: nn.Module, context: str) -> None:
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(f"{context} parameter {name} contains NaN or Inf.")


def train_autoencoder(
    train_features: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
    learning_rate: float = 0.05,
) -> tuple[PSDAutoencoder, list[float]]:
    features = np.asarray(train_features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != int(AE_CONFIG["input_units"]):
        raise ValueError("AE training features have an invalid shape.")
    if not np.isfinite(features).all():
        raise ValueError("AE training features contain NaN or Inf.")
    set_deterministic_seed(seed)
    model = PSDAutoencoder(AE_CONFIG).to(device)
    tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=float(learning_rate))
    criterion = nn.MSELoss()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    losses: list[float] = []
    for _ in range(int(epochs)):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch_indices in _epoch_batches(
            len(tensor), batch_size, generator, require_minimum_two=True
        ):
            batch = tensor.index_select(0, batch_indices.to(device))
            optimizer.zero_grad(set_to_none=True)
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("AE loss is NaN or Inf.")
            loss.backward()
            optimizer.step()
            count = int(len(batch))
            total_loss += float(loss.detach().item()) * count
            total_count += count
        _assert_finite_model(model, "AE")
        value = total_loss / float(total_count)
        if total_count != len(tensor) or not np.isfinite(value):
            raise RuntimeError("AE epoch output is invalid.")
        losses.append(float(value))
    return model, losses


@torch.no_grad()
def encode_features(
    model: PSDAutoencoder,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    model.eval()
    encoded = []
    for start in range(0, len(array), int(batch_size)):
        batch = torch.as_tensor(array[start : start + int(batch_size)], device=device)
        encoded.append(model.encode(batch).detach().cpu().numpy())
    result = np.concatenate(encoded).astype(np.float32, copy=False)
    if result.shape != (len(array), int(RBF_CONFIG["input_units"])):
        raise RuntimeError(f"AE latent shape is invalid: {result.shape}.")
    if not np.isfinite(result).all():
        raise RuntimeError("AE latent contains NaN or Inf.")
    return result


def initialize_rbf_centers(
    train_latent: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, float]:
    latent = np.asarray(train_latent, dtype=np.float32)
    if latent.ndim != 2 or latent.shape[1] != int(RBF_CONFIG["input_units"]):
        raise ValueError("KMeans latent input has an invalid shape.")
    if len(latent) < int(RBF_CONFIG["n_centers"]):
        raise ValueError("has fewer training samples than fixed RBF centers.")
    if not np.isfinite(latent).all():
        raise ValueError("KMeans latent input contains NaN or Inf.")
    estimator = KMeans(
        n_clusters=int(RBF_CONFIG["n_centers"]),
        random_state=int(seed),
        n_init=int(RBF_CONFIG["kmeans_n_init"]),
        algorithm=str(RBF_CONFIG["kmeans_algorithm"]),
    )
    with threadpool_limits(limits=1):
        centers = np.asarray(estimator.fit(latent).cluster_centers_, dtype=np.float32)
    differences = centers.astype(np.float64)[:, None, :] - centers.astype(np.float64)[None, :, :]
    distances = np.sqrt(np.square(differences).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    sigma = float(np.median(nearest))
    if not np.isfinite(centers).all() or not np.isfinite(sigma) or sigma <= 0.0:
        raise RuntimeError("KMeans produced invalid centers or sigma.")
    return centers, sigma


def train_rbfnn(
    train_latent: np.ndarray,
    train_labels: np.ndarray,
    initial_centers: np.ndarray,
    initial_sigma: float,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
    learning_rate: float,
) -> tuple[TrainableRBFNN, list[float]]:
    latent = np.asarray(train_latent, dtype=np.float32)
    labels = np.asarray(train_labels, dtype=np.int64).reshape(-1)
    if len(latent) != len(labels) or set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("RBF training labels are invalid.")
    set_deterministic_seed(seed)
    model = TrainableRBFNN(initial_centers, initial_sigma).to(device)
    features = torch.as_tensor(latent, dtype=torch.float32, device=device)
    targets = torch.as_tensor(labels, dtype=torch.int64, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    losses: list[float] = []
    for _ in range(int(epochs)):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch_indices in _epoch_batches(
            len(targets), batch_size, generator, require_minimum_two=False
        ):
            indices = batch_indices.to(device)
            batch_features = features.index_select(0, indices)
            batch_targets = targets.index_select(0, indices)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("RBFNN loss is NaN or Inf.")
            loss.backward()
            optimizer.step()
            count = int(len(batch_targets))
            total_loss += float(loss.detach().item()) * count
            total_count += count
        _assert_finite_model(model, "RBFNN")
        value = total_loss / float(total_count)
        if total_count != len(targets) or not np.isfinite(value):
            raise RuntimeError("RBFNN epoch output is invalid.")
        losses.append(float(value))
    return model, losses


@torch.no_grad()
def infer_rbfnn_probabilities(
    model: TrainableRBFNN,
    latent: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    array = np.asarray(latent, dtype=np.float32)
    model.eval()
    probabilities = []
    for start in range(0, len(array), int(batch_size)):
        batch = torch.as_tensor(array[start : start + int(batch_size)], device=device)
        probabilities.append(torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy())
    result = np.concatenate(probabilities).astype(np.float64, copy=False)
    if len(result) != len(array) or not np.isfinite(result).all():
        raise RuntimeError("RBFNN probabilities are invalid.")
    if np.any((result < 0.0) | (result > 1.0)):
        raise RuntimeError("RBFNN probabilities fall outside [0, 1].")
    return result


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


class PSDAERBFNN(nn.Module):
    """Compose the PSD encoder and an RBF classifier initialized on training data.

    Input consists of 120 PSD features, not waveform samples. Centers and widths
    come from ``initialize_rbf_centers`` after fitting the autoencoder.
    """

    def __init__(self, initial_centers: np.ndarray, initial_sigma: float,
                 autoencoder: PSDAutoencoder | None = None):
        super().__init__()
        self.autoencoder = PSDAutoencoder() if autoencoder is None else autoencoder
        self.classifier = TrainableRBFNN(initial_centers, initial_sigma)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.autoencoder.encode(features))
