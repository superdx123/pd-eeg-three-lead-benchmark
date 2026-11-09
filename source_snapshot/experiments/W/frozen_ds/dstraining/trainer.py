"""Deterministic training utilities: preprocessing, augmentation, early stopping,
candidate (learning-rate) selection, and fixed-epoch refit + evaluation.
"""
from __future__ import annotations

import os
import math
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from dsdata.metrics import aggregate_true_label_subject_probabilities, binary_probability_metrics_full

DEFAULT_BATCH_SIZE = 256
MAX_EPOCHS = 80
PATIENCE = 12
LEARNING_RATES = (3e-4, 1e-3)
WEIGHT_DECAY = 1e-4
THRESHOLD = 0.5
METRIC_TIE_TOLERANCE = 1e-12


def configure_device(deterministic: bool = True) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    return device


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class GradientReversalFunction(torch.autograd.Function):
    """Reverse gradients w.r.t. the encoder for the subject discriminator loss."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


def is_adv_subject(config) -> bool:
    return bool(config.params.get("adv_subject", False))


def adv_lambda(config) -> float:
    return float(config.params.get("adv_lambda", 0.1))


def build_subject_discriminator(config, n_subjects: int, device: torch.device) -> nn.Module:
    """Small MLP discriminator over subject identity (built per fold: n_subjects varies)."""
    if n_subjects < 2:
        raise ValueError("Subject discriminator needs >= 2 subjects")
    return nn.Sequential(
        nn.Linear(int(config.embedding_dim), 64, bias=False),
        nn.ReLU(),
        nn.Linear(64, int(n_subjects), bias=True),
    ).to(device)


def make_subject_labels(groups, train_indices) -> tuple:
    """Map train subject ids to contiguous ints; return (labels_tensor, n_subjects)."""
    subjects = np.sort(np.unique(np.asarray(groups)[train_indices]))
    lookup = {str(v): i for i, v in enumerate(subjects)}
    labels = np.asarray([lookup[str(v)] for v in np.asarray(groups)[train_indices]], dtype=np.int64)
    return torch.as_tensor(labels, dtype=torch.int64), int(len(subjects))


# ---------------------------------------------------------------------------
# Preprocessing (fit on train only, except per-window transforms)
# ---------------------------------------------------------------------------

def fit_preprocess(X_train: np.ndarray, mode: str, input_transform: str = "raw", n_bands: int = 5,
                    groups: np.ndarray | None = None):
    """Return (transform_fn, meta). transform_fn(X, groups=None) -> standardized X."""
    from dsmodel.transforms import transform_input
    X_train = transform_input(np.asarray(X_train, dtype=np.float32), input_transform, n_bands=n_bands)
    if mode == "global_scalar":
        mean = float(np.mean(X_train, dtype=np.float64))
        std = float(np.std(X_train, dtype=np.float64) + 1e-8)

        def transform(X, groups=None):
            X = transform_input(np.asarray(X, dtype=np.float32), input_transform, n_bands=n_bands)
            return np.asarray((X - mean) / std, dtype=np.float32)

        return transform, {"mean": mean, "std": std}
    elif mode == "window_zscore":
        def transform(X, groups=None):
            X = transform_input(np.asarray(X, dtype=np.float32), input_transform, n_bands=n_bands)
            mu = X.mean(axis=(-2, -1), keepdims=True)
            sd = X.std(axis=(-2, -1), keepdims=True) + 1e-6
            return np.asarray((X - mu) / sd, dtype=np.float32)

        return transform, {"mode": "window_zscore"}
    elif mode == "channel_zscore":
        # Per-channel per-window standardization (time axis only). Removes absolute
        # band-power level differences across cohorts/channels while preserving
        # within-band waveform shape. Round-11 candidate for cross-cohort transfer.
        def transform(X, groups=None):
            X = transform_input(np.asarray(X, dtype=np.float32), input_transform, n_bands=n_bands)
            mu = X.mean(axis=-1, keepdims=True)
            sd = X.std(axis=-1, keepdims=True) + 1e-6
            return np.asarray((X - mu) / sd, dtype=np.float32)

        return transform, {"mode": "channel_zscore"}
    elif mode == "subject_chmean":
        # Round-15 (490-specific): per-subject per-channel mean removal (subject
        # identity channel-amplitude profile). Self-normalizing: each subject's
        # windows are centered by that subject's own channel means (train subjects
        # from their train windows, eval subjects from their own windows). Preserves
        # within-subject waveform amplitude (unlike channel_zscore).
        def transform(X, groups=None):
            X = transform_input(np.asarray(X, dtype=np.float32), input_transform, n_bands=n_bands)
            X = np.asarray(X, dtype=np.float32)
            if groups is None:
                raise ValueError("subject_chmean requires groups (subject ids per row)")
            g = np.asarray(groups)
            out = np.empty_like(X)
            for s in np.unique(g):
                mask = g == s
                mu = X[mask].mean(axis=(0, -1), keepdims=True)
                out[mask] = X[mask] - mu
            return out

        return transform, {"mode": "subject_chmean"}
    else:
        raise ValueError(f"Unknown preprocess mode {mode!r}")


def apply_preprocess(X: np.ndarray, meta: dict, mode: str, input_transform: str = "raw", n_bands: int = 5,
                    groups: np.ndarray | None = None) -> np.ndarray:
    from dsmodel.transforms import transform_input
    X = transform_input(np.asarray(X, dtype=np.float32), input_transform, n_bands=n_bands)
    if mode == "global_scalar":
        return np.asarray((X - float(meta["mean"])) / float(meta["std"]), dtype=np.float32)
    if mode == "window_zscore":
        X = np.asarray(X, dtype=np.float32)
        mu = X.mean(axis=(-2, -1), keepdims=True)
        sd = X.std(axis=(-2, -1), keepdims=True) + 1e-6
        return np.asarray((X - mu) / sd, dtype=np.float32)
    if mode == "channel_zscore":
        X = np.asarray(X, dtype=np.float32)
        mu = X.mean(axis=-1, keepdims=True)
        sd = X.std(axis=-1, keepdims=True) + 1e-6
        return np.asarray((X - mu) / sd, dtype=np.float32)
    if mode == "subject_chmean":
        X = np.asarray(X, dtype=np.float32)
        if groups is None:
            raise ValueError("subject_chmean requires groups")
        g = np.asarray(groups)
        out = np.empty_like(X)
        for s in np.unique(g):
            mask = g == s
            mu = X[mask].mean(axis=(0, -1), keepdims=True)
            out[mask] = X[mask] - mu
        return out
    raise ValueError(f"Unknown preprocess mode {mode!r}")


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment_batch(x: torch.Tensor, mode: str, generator: torch.Generator, std_scale: float = 1.0) -> torch.Tensor:
    if mode == "none":
        return x
    # x: [B, 3, 512], approximately standardized. Random values are drawn on CPU
    # (CPU generator) and moved to the tensor device for deterministic streams.
    device = x.device
    n = x.shape[0]
    if mode == "light":
        noise = torch.randn(x.shape, generator=generator, device="cpu").to(device) * (0.05 * std_scale)
        scale = 1.0 + (torch.rand(n, 1, 1, generator=generator, device="cpu") - 0.5).to(device) * 0.2
        return x * scale + noise
    if mode == "light_shift":
        # Round-12: light (noise+scale) + random circular time shift (+/-16 samples).
        noise = torch.randn(x.shape, generator=generator, device="cpu").to(device) * (0.05 * std_scale)
        scale = 1.0 + (torch.rand(n, 1, 1, generator=generator, device="cpu") - 0.5).to(device) * 0.2
        out = x * scale + noise
        shift = torch.randint(-16, 17, (n,), generator=generator, device="cpu")
        idx = (torch.arange(x.shape[-1], device="cpu")[None, :] - shift[:, None]) % x.shape[-1]
        return out.gather(-1, idx[:, None, :].to(device).expand(n, x.shape[1], x.shape[-1]))
    if mode == "light_strong":
        # Round-13: stronger time-domain light augment (noise 0.08, amp +-15%).
        noise = torch.randn(x.shape, generator=generator, device="cpu").to(device) * (0.08 * std_scale)
        scale = 1.0 + (torch.rand(n, 1, 1, generator=generator, device="cpu") - 0.5).to(device) * 0.3
        return x * scale + noise
    if mode in ("freq", "freq_light"):
        # SpecAugment-style: random contiguous frequency masking + light noise/scale.
        # freq_light: much smaller masks (1-6 bins) + weaker noise.
        spec = torch.fft.rfft(x, dim=-1)
        n_freq = spec.shape[-1]
        if mode == "freq_light":
            mask_len = torch.randint(1, 7, (n,), generator=generator, device="cpu")
        else:
            mask_len = torch.randint(6, 36, (n,), generator=generator, device="cpu")
        start = torch.randint(0, n_freq, (n,), generator=generator, device="cpu")
        freq_mask = torch.ones(n, n_freq, dtype=x.dtype)
        for i in range(n):
            length = min(int(mask_len[i].item()), n_freq)
            s = min(int(start[i].item()), n_freq - length)
            freq_mask[i, s:s + length] = 0.0
        spec = spec * freq_mask.unsqueeze(1).to(device)
        out = torch.fft.irfft(spec, n=x.shape[-1], dim=-1)
        noise_std = 0.01 if mode == "freq_light" else 0.03
        noise = torch.randn(x.shape, generator=generator, device="cpu").to(device) * (noise_std * std_scale)
        scale = 1.0 + (torch.rand(n, 1, 1, generator=generator, device="cpu") - 0.5).to(device) * 0.2
        return out * scale + noise
    if mode == "heavy":
        noise = torch.randn(x.shape, generator=generator, device="cpu").to(device) * (0.08 * std_scale)
        scale = 1.0 + (torch.rand(n, 1, 1, generator=generator, device="cpu") - 0.5).to(device) * 0.4
        out = x * scale + noise
        drop = torch.rand(n, generator=generator, device="cpu") < 0.2
        if drop.any():
            drop_idx = torch.nonzero(drop, as_tuple=True)[0]
            ch = torch.randint(0, x.shape[1], (len(drop_idx),), generator=generator, device="cpu")
            out[drop_idx.to(device), ch.to(device)] = 0.0
        shift = torch.randint(-24, 25, (n,), generator=generator, device="cpu")
        out = torch.stack([torch.roll(out[i], int(shift[i].item()), dims=-1) for i in range(n)], dim=0)
        mask_len = int(torch.randint(32, 97, (1,), generator=generator, device="cpu").item())
        start = int(torch.randint(0, 512 - mask_len + 1, (1,), generator=generator, device="cpu").item())
        out[..., start:start + mask_len] = 0.0
        flip = torch.rand(n, 1, 1, generator=generator, device="cpu") < 0.1
        out = torch.where(flip.to(device), -out, out)
        return out
    raise ValueError(f"Unknown augmentation mode {mode!r}")


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float, generator: torch.Generator) -> tuple:
    if alpha <= 0.0:
        return x, y
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.shape[0], generator=generator)
    x_mix = lam * x + (1 - lam) * x[perm]
    return x_mix, (y, y[perm], lam)


def ranking_loss(logits: torch.Tensor, targets: torch.Tensor, margin: float = 0.3,
                 max_pairs: int = 40000) -> torch.Tensor:
    """Pairwise margin ranking loss on window scores (surrogate for subject AUC).

    score = logit_pd - logit_hc; enforces score_pos > score_neg + margin.
    Returns a scalar tensor; falls back to zero if a batch has one class only.
    """
    score = logits[:, 1] - logits[:, 0]
    pos = score[targets == 1]
    neg = score[targets == 0]
    if len(pos) == 0 or len(neg) == 0:
        # No cross-class pairs in this batch: return a differentiable zero
        # (logits are part of the autograd graph).
        return (logits * 0.0).sum()
    diff = pos[:, None] - neg[None, :]
    n_pairs = diff.numel()
    if n_pairs > max_pairs:
        flat = diff.reshape(-1)
        idx = torch.randperm(n_pairs, device=logits.device)[:max_pairs]
        diff = flat[idx]
    return torch.relu(margin - diff).mean()


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def balanced_class_weights(labels, device: torch.device) -> torch.Tensor:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    counts = np.bincount(y, minlength=2)
    if len(y) == 0 or np.any(counts <= 0):
        raise ValueError("Training labels must contain both classes")
    weights = len(y) / (2.0 * counts.astype(np.float64))
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


class FocalLoss(nn.Module):
    """Focal loss (Lin et al. 2017) with optional class weights and label
    smoothing. Modulates per-sample CE by (1 - p_t)^gamma to up-weight hard
    windows; targets hard subjects that sit near the 0.5 decision boundary."""

    def __init__(self, weight=None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer("weight", weight.clone() if weight is not None else None)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        pt_log = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        if self.label_smoothing > 0.0:
            n_cls = logits.shape[-1]
            per_sample = -(1.0 - self.label_smoothing) * pt_log \
                - self.label_smoothing * log_probs.mean(dim=-1)
        else:
            per_sample = -pt_log
        with torch.no_grad():
            p = torch.softmax(logits, dim=-1)
            pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        per_sample = per_sample * (1.0 - pt) ** self.gamma
        if self.weight is not None:
            per_sample = per_sample * self.weight[targets]
        return per_sample.mean()


def build_optimizer(config, optimizer_params, learning_rate: float) -> torch.optim.Optimizer:
    """Adam (default) or AdamW (decoupled weight decay) per config.params.optimizer."""
    wd = float(config.params.get("wd", WEIGHT_DECAY))
    if str(config.params.get("optimizer", "adam")) == "adamw":
        return torch.optim.AdamW(optimizer_params, lr=float(learning_rate), weight_decay=wd)
    return torch.optim.Adam(optimizer_params, lr=float(learning_rate), weight_decay=wd)


def build_criterion(config, labels, device: torch.device) -> nn.Module:
    """CE (default) or focal-loss criterion, with class weights + LS."""
    weight = balanced_class_weights(labels, device)
    if str(config.params.get("loss", "ce")) == "focal":
        return FocalLoss(weight=weight, gamma=float(config.params.get("focal_gamma", 2.0)),
                         label_smoothing=float(config.label_smoothing))
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=float(config.label_smoothing))


def balanced_epoch_order(groups, generator: torch.Generator) -> torch.Tensor:
    """Subject-balanced epoch order: every subject contributes the same number of
    windows (min per-subject count), sampled without replacement; extras rotate
    across epochs. Returns CPU index tensor for the training set."""
    g = np.asarray(groups)
    subj_ids = np.unique(g)
    counts = np.bincount(np.searchsorted(np.sort(subj_ids), g))
    n_target = int(counts.min())
    chosen = []
    for sid in subj_ids:
        idx = np.flatnonzero(g == sid)
        perm = torch.randperm(len(idx), generator=generator, device="cpu")
        chosen.append(idx[perm[:n_target]])
    order = torch.cat([torch.as_tensor(c, dtype=torch.long) for c in chosen])
    order = order[torch.randperm(len(order), generator=generator, device="cpu")]
    return order


def _subject_block_batches(train_groups, generator: torch.Generator, batch_size: int) -> list:
    """Round-19 MIL: subject-grouped batches packing complete subjects.

    Shuffles subject order (generator-driven, deterministic), keeps each subject's
    windows contiguous, packs whole subjects into batches up to batch_size windows.
    Oversized subjects are split into batch_size chunks (later chunks get window
    CE only that epoch; the subject-level term covers complete subjects in a batch).
    """
    g = np.asarray(train_groups)
    subj_ids = np.unique(g)
    perm = torch.randperm(len(subj_ids), generator=generator, device="cpu")
    batches = []
    current = []
    for i in perm.tolist():
        idx = np.flatnonzero(g == subj_ids[int(i)])
        if len(idx) >= batch_size:
            if current:
                batches.append(np.concatenate(current))
                current = []
            for s in range(0, len(idx), batch_size):
                batches.append(idx[s:s + batch_size])
        else:
            if current and sum(len(c) for c in current) + len(idx) > batch_size:
                batches.append(np.concatenate(current))
                current = []
            current.append(idx)
    if current:
        batches.append(np.concatenate(current))
    return batches


def _train_one_epoch_subject_attn(
    model, optimizer, criterion, x: torch.Tensor, y: torch.Tensor,
    batch_size: int, generator: torch.Generator, config, data_std: float,
    train_groups,
) -> float:
    """Round-19 MIL training: window CE + lambda * attention-aggregated subject CE.

    Window supervision unchanged; a second term supervises the learned
    per-window attention via subject-level CE (softmax weights over a subject's
    windows present in the batch). Deterministic given the generator stream.
    """
    model.train()
    if train_groups is None:
        raise ValueError("subject_attn requires train_groups")
    attn_lambda = float(config.params.get("subject_attn_lambda", 1.0))
    batches = _subject_block_batches(train_groups, generator, batch_size)
    total_loss = 0.0
    total_count = 0
    for pos in batches:
        batch_indices = torch.as_tensor(pos, dtype=torch.long).to(x.device)
        inputs = x.index_select(0, batch_indices)
        targets = y.index_select(0, batch_indices)
        subj = np.asarray(train_groups)[pos]
        if config.augment != "none":
            inputs = augment_batch(inputs, config.augment, generator, std_scale=data_std)
        optimizer.zero_grad(set_to_none=True)
        embedding = model.embed(inputs)
        logits = model.classify(embedding)
        weights = model.subject_attn(embedding)
        win_loss = criterion(logits, targets)
        subj_losses = []
        for sid in np.unique(subj):
            mask = torch.as_tensor(subj == sid).to(x.device)
            if int(mask.sum().item()) < 2:
                continue
            a = torch.softmax(weights[mask].squeeze(-1), dim=0)
            s_logit = (a.unsqueeze(-1) * logits[mask]).sum(dim=0, keepdim=True)
            t = targets[mask][0].view(1)
            subj_losses.append(criterion(s_logit, t))
        if subj_losses:
            subj_loss = torch.stack(subj_losses).mean()
        else:
            subj_loss = (logits * 0.0).sum()
        window_ce_weight = float(config.params.get("window_ce_weight", 1.0))
        loss = window_ce_weight * win_loss + attn_lambda * subj_loss
        loss.backward()
        optimizer.step()
        count = int(len(targets))
        total_loss += float(loss.detach().item()) * count
        total_count += count
    if total_count != len(y):
        raise RuntimeError("Epoch did not consume every training window")
    return total_loss / max(total_count, 1)


def train_one_epoch(
    model, optimizer, criterion, x: torch.Tensor, y: torch.Tensor,
    batch_size: int, generator: torch.Generator, config, data_std: float,
    subject_labels=None, discriminator=None, train_groups=None,
) -> float:
    model.train()
    adv = discriminator is not None and subject_labels is not None
    if adv and config.mixup_alpha > 0.0:
        raise ValueError("adv_subject cannot be combined with mixup")
    if config.params.get("subject_attn", False):
        if config.params.get("subject_balanced", False):
            raise ValueError("subject_attn cannot be combined with subject_balanced")
        return _train_one_epoch_subject_attn(
            model, optimizer, criterion, x, y, batch_size, generator, config, data_std, train_groups)
    if config.params.get("subject_balanced", False):
        if train_groups is None:
            raise ValueError("subject_balanced requires train_groups")
        order = balanced_epoch_order(train_groups, generator)
    else:
        order = torch.randperm(len(y), generator=generator, device="cpu")
    total_loss = 0.0
    total_count = 0
    for start in range(0, len(order), batch_size):
        batch_indices = order[start:start + batch_size].to(x.device)
        inputs = x.index_select(0, batch_indices)
        targets = y.index_select(0, batch_indices)
        if config.augment != "none":
            inputs = augment_batch(inputs, config.augment, generator, std_scale=data_std)
        optimizer.zero_grad(set_to_none=True)
        if adv:
            embedding = model.embed(inputs)
            logits = model.classify(embedding)
            loss = criterion(logits, targets)
            subj_batch = subject_labels.index_select(0, batch_indices)
            reversed_embedding = GradientReversalFunction.apply(embedding, adv_lambda(config))
            disc_logits = discriminator(reversed_embedding)
            disc_criterion = nn.CrossEntropyLoss()
            loss = loss + disc_criterion(disc_logits, subj_batch)
        elif config.mixup_alpha > 0.0 and model.training:
            inputs, mix = mixup_batch(inputs, targets, config.mixup_alpha, generator)
            y0, y1, lam = mix
            logits = model(inputs)
            loss = (1 - lam) * criterion(logits, y0) + lam * criterion(logits, y1)
        else:
            logits = model(inputs)
            loss_mode = str(config.params.get("loss", "ce"))
            if loss_mode == "rank":
                loss = ranking_loss(logits, targets, margin=float(config.params.get("rank_margin", 0.3)))
            elif loss_mode == "ce_rank":
                loss = criterion(logits, targets) + float(config.params.get("rank_lambda", 0.5)) * ranking_loss(
                    logits, targets, margin=float(config.params.get("rank_margin", 0.3)))
            else:
                loss = criterion(logits, targets)
            # Consistency regularization (Round-11, Pi-model style): a second
            # light-augmented view of the same window must give similar output.
            consist_lambda = float(config.params.get("consist_lambda", 0.0) or 0.0)
            if consist_lambda > 0.0:
                inputs2 = augment_batch(
                    x.index_select(0, batch_indices), config.augment, generator, std_scale=data_std)
                logits2 = model(inputs2)
                if str(config.params.get("consist_mode", "logit")) == "prob":
                    p1 = torch.softmax(logits, dim=-1)
                    p2 = torch.softmax(logits2, dim=-1)
                    consist = ((p1 - p2) ** 2).mean()
                else:
                    consist = ((logits - logits2) ** 2).mean()
                loss = loss + consist_lambda * consist
        loss.backward()
        optimizer.step()
        count = int(len(targets))
        total_loss += float(loss.detach().item()) * count
        total_count += count
    if config.params.get("subject_balanced", False):
        # subject-balanced epochs intentionally subsample per subject (min-count)
        pass
    elif total_count != len(y):
        raise RuntimeError("Epoch did not consume every training window")
    return total_loss / max(total_count, 1)


@torch.no_grad()
def infer_probabilities(model, x: torch.Tensor, batch_size: int, tta: int = 0, config=None) -> np.ndarray:
    model.eval()
    probs = []
    for start in range(0, len(x), batch_size):
        batch = x[start:start + batch_size]
        if tta > 0 and config is not None and config.augment != "none":
            views = [batch]
            for v in range(tta):
                gen = torch.Generator(device="cpu").manual_seed(100000 + v)
                views.append(augment_batch(batch, config.augment, gen, std_scale=1.0))
            stacked = torch.cat(views, dim=0)
            logits = model(stacked)
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise RuntimeError(f"Unexpected logits shape {tuple(logits.shape)}")
            p_all = torch.softmax(logits, dim=1)[:, 1].reshape(len(views), len(batch)).mean(dim=0)
            probs.append(p_all.detach().cpu().numpy())
        else:
            logits = model(batch)
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise RuntimeError(f"Unexpected logits shape {tuple(logits.shape)}")
            probs.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
    result = np.concatenate(probs).astype(np.float64, copy=False)
    if len(result) != len(x) or not np.isfinite(result).all():
        raise RuntimeError("Invalid inference probabilities")
    return result


def cpu_state_dict(model: nn.Module) -> dict:
    return {name: t.detach().cpu().clone() for name, t in model.state_dict().items()}


@torch.no_grad()
def infer_probabilities_weights(model, x: torch.Tensor, batch_size: int) -> tuple:
    """Round-19 MIL inference: (window head probs, pd logit-difference, attention weights).

    Subject aggregation = sigmoid(sum(softmax(weights) * pd_logit)) — logit space,
    consistent with the training-time subject CE (subject logit = weighted-mean
    of window logits under a linear head).
    """
    model.eval()
    probs = []
    pd_logits = []
    weights = []
    for start in range(0, len(x), batch_size):
        batch = x[start:start + batch_size]
        embedding = model.embed(batch)
        logits = model.classify(embedding)
        w = model.subject_attn(embedding)
        probs.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
        pd_logits.append((logits[:, 1] - logits[:, 0]).detach().cpu().numpy())
        weights.append(w.squeeze(-1).detach().cpu().numpy())
    p = np.concatenate(probs).astype(np.float64, copy=False)
    d = np.concatenate(pd_logits).astype(np.float64, copy=False)
    w = np.concatenate(weights).astype(np.float64, copy=False)
    if len(p) != len(x) or not np.isfinite(p).all() or not np.isfinite(d).all() or not np.isfinite(w).all():
        raise RuntimeError("Invalid attention inference output")
    return p, d, w


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# ---------------------------------------------------------------------------
# Curve / selection
# ---------------------------------------------------------------------------

def train_validation_curve(
    config, learning_rate: float, X, y, groups,
    train_indices, validation_indices, device, seed: int,
    max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    selection_metric: str = "bacc",
) -> tuple[dict, list, list]:
    """Train with early stopping on validation subject-level metric (bacc or auc).

    Returns (candidate dict, epoch history, retained validation rows).
    """
    transform, meta = fit_preprocess(X[train_indices], config.preprocess, config.input_transform, n_bands=config.n_bands,
                                    groups=groups[train_indices])
    X_train = transform(X[train_indices], groups[train_indices])
    X_validation = transform(X[validation_indices], groups[validation_indices])
    data_std = float(np.std(X_train, dtype=np.float64) + 1e-8) if config.augment != "none" else 1.0

    set_seed(seed)
    from dsmodel.models import build_model
    model = build_model(config, device)
    if str(config.params.get("ssl", "") or "") and dataset_id is not None:
        from dsmodel.ssl import apply_pretrained
        apply_pretrained(model, config, dataset_id, protocol, repeat_index, outer_fold)
    discriminator = None
    subject_labels = None
    if is_adv_subject(config):
        subject_labels, n_subjects = make_subject_labels(groups, train_indices)
        subject_labels = subject_labels.to(device)
        discriminator = build_subject_discriminator(config, n_subjects, device)
    criterion = build_criterion(config, y[train_indices], device)
    optimizer_params = list(model.parameters())
    if discriminator is not None:
        optimizer_params = optimizer_params + list(discriminator.parameters())
    optimizer = build_optimizer(config, optimizer_params, learning_rate)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    train_features = torch.as_tensor(X_train, device=device)
    train_labels = torch.as_tensor(y[train_indices], dtype=torch.int64, device=device)
    validation_features = torch.as_tensor(X_validation, device=device)

    history = []
    validation_rows = []
    best_epoch = 0
    best_bacc = -np.inf
    best_auc = -np.inf
    patience_counter = 0
    stop_epoch = max_epochs

    for epoch in range(1, int(max_epochs) + 1):
        epoch_gen = torch.Generator(device="cpu").manual_seed(seed * 100003 + epoch)
        loss_value = train_one_epoch(
            model, optimizer, criterion, train_features, train_labels,
            batch_size, epoch_gen, config, data_std,
            subject_labels=subject_labels, discriminator=discriminator,
            train_groups=groups[train_indices],
        )
        probabilities = infer_probabilities(model, validation_features, batch_size, tta=int(config.params.get("tta", 0)), config=config)
        subjects = aggregate_true_label_subject_probabilities(
            groups[validation_indices], y[validation_indices], probabilities, threshold=THRESHOLD
        )
        metrics = binary_probability_metrics_full(subjects["true_label"], subjects["prob_pd"], threshold=THRESHOLD)
        bacc = float(metrics["balanced_accuracy"])
        auc = float(metrics["auc"])
        if selection_metric == "auc":
            improved = auc > best_auc + METRIC_TIE_TOLERANCE
            if math.isclose(auc, best_auc, rel_tol=0.0, abs_tol=METRIC_TIE_TOLERANCE) and bacc > best_bacc + METRIC_TIE_TOLERANCE:
                improved = True
        else:
            improved = bacc > best_bacc + METRIC_TIE_TOLERANCE
            if math.isclose(bacc, best_bacc, rel_tol=0.0, abs_tol=METRIC_TIE_TOLERANCE) and auc > best_auc + METRIC_TIE_TOLERANCE:
                improved = True
        if improved:
            best_epoch = epoch
            best_bacc = bacc
            best_auc = auc
            patience_counter = 0
        else:
            patience_counter += 1
        history.append({
            "epoch": epoch, "training_loss": float(loss_value),
            "validation_subject_balanced_accuracy": bacc, "validation_subject_auc": auc,
            "improved": bool(improved), "patience_counter": int(patience_counter),
        })
        for row in subjects.itertuples(index=False):
            validation_rows.append({
                "epoch": epoch, "subject_id": str(row.subject_id), "true_label": int(row.true_label),
                "prob_pd": float(row.prob_pd), "n_windows": int(row.n_windows),
                "predicted_label": int(row.predicted_label),
            })
        if patience_counter >= int(patience):
            stop_epoch = epoch
            break
    if best_epoch < 1:
        raise RuntimeError("Epoch selection failed")
    for row in history:
        row["selection_stopped_here"] = bool(int(row["epoch"]) == stop_epoch)
    candidate = {
        "learning_rate": float(learning_rate), "selected_epoch": int(best_epoch),
        "best_validation_subject_balanced_accuracy": float(best_bacc),
        "best_validation_subject_auc": float(best_auc), "selection_stop_epoch": int(stop_epoch),
    }
    retained = [row for row in validation_rows if int(row["epoch"]) <= stop_epoch]
    del model, optimizer, criterion, train_features, train_labels, validation_features
    if discriminator is not None:
        del discriminator, subject_labels
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return candidate, history, retained


def choose_candidate(candidates: list[dict], metric: str = "bacc") -> dict:
    """Fixed rule: subject bAcc primary (AUC tie-break) or AUC primary (bAcc tie-break)."""
    if metric not in ("bacc", "auc"):
        raise ValueError(f"Unknown selection metric {metric!r}")
    primary_key = "best_validation_subject_balanced_accuracy" if metric == "bacc" else "best_validation_subject_auc"
    secondary_key = "best_validation_subject_auc" if metric == "bacc" else "best_validation_subject_balanced_accuracy"
    if len(candidates) == 0:
        raise RuntimeError("Empty candidate set")
    best = None
    for candidate in sorted(candidates, key=lambda row: float(row["learning_rate"])):
        if best is None:
            best = candidate
            continue
        primary = float(candidate[primary_key])
        best_primary = float(best[primary_key])
        secondary = float(candidate[secondary_key])
        best_secondary = float(best[secondary_key])
        replace = primary > best_primary + METRIC_TIE_TOLERANCE
        if math.isclose(primary, best_primary, rel_tol=0.0, abs_tol=METRIC_TIE_TOLERANCE):
            if secondary > best_secondary + METRIC_TIE_TOLERANCE:
                replace = True
            elif math.isclose(secondary, best_secondary, rel_tol=0.0, abs_tol=METRIC_TIE_TOLERANCE):
                if float(candidate["learning_rate"]) < float(best["learning_rate"]):
                    replace = True
                elif math.isclose(float(candidate["learning_rate"]), float(best["learning_rate"]), rel_tol=0.0, abs_tol=1e-15) and int(candidate["selected_epoch"]) < int(best["selected_epoch"]):
                    replace = True
        if replace:
            best = candidate
    return dict(best)


def train_final_and_predict(
    config, learning_rate: float, X, y, groups,
    train_indices, test_indices, selected_epoch: int, device, seed: int,
    batch_size: int = DEFAULT_BATCH_SIZE, return_state: bool = False,
    dataset_id=None, protocol=None, repeat_index=None, outer_fold=None,
):
    """Refit on train(+validation) indices for selected_epoch, predict test windows."""
    transform, meta = fit_preprocess(X[train_indices], config.preprocess, config.input_transform, n_bands=config.n_bands,
                                    groups=groups[train_indices])
    X_train = transform(X[train_indices], groups[train_indices])
    X_test = transform(X[test_indices], groups[test_indices])
    data_std = float(np.std(X_train, dtype=np.float64) + 1e-8) if config.augment != "none" else 1.0
    set_seed(seed)
    from dsmodel.models import build_model
    model = build_model(config, device)
    discriminator = None
    subject_labels = None
    if is_adv_subject(config):
        subject_labels, n_subjects = make_subject_labels(groups, train_indices)
        subject_labels = subject_labels.to(device)
        discriminator = build_subject_discriminator(config, n_subjects, device)
    criterion = build_criterion(config, y[train_indices], device)
    optimizer_params = list(model.parameters())
    if discriminator is not None:
        optimizer_params = optimizer_params + list(discriminator.parameters())
    optimizer = build_optimizer(config, optimizer_params, learning_rate)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    train_features = torch.as_tensor(X_train, device=device)
    train_labels = torch.as_tensor(y[train_indices], dtype=torch.int64, device=device)
    test_features = torch.as_tensor(X_test, device=device)
    losses = []
    epoch_avg = int(config.params.get("epoch_avg", 0) or 0)
    swa_epochs = int(config.params.get("swa_epochs", 0) or 0)
    prob_buffer = []
    weight_buffer = []
    for ep in range(1, int(selected_epoch) + 1):
        epoch_gen = torch.Generator(device="cpu").manual_seed(seed * 100003 + ep)
        losses.append(train_one_epoch(
            model, optimizer, criterion, train_features, train_labels,
            batch_size, epoch_gen, config, data_std,
            subject_labels=subject_labels, discriminator=discriminator,
            train_groups=groups[train_indices],
        ))
        if epoch_avg > 0:
            prob_buffer.append(infer_probabilities(
                model, test_features, batch_size,
                tta=int(config.params.get("tta", 0)), config=config,
            ))
            if len(prob_buffer) > int(epoch_avg):
                prob_buffer.pop(0)
        if swa_epochs > 0:
            weight_buffer.append(cpu_state_dict(model))
            if len(weight_buffer) > int(swa_epochs):
                weight_buffer.pop(0)
    if swa_epochs > 0 and weight_buffer:
        averaged = {k: sum(w[k] for w in weight_buffer) / len(weight_buffer) for k in weight_buffer[0]}
        model.load_state_dict(averaged)
    if epoch_avg > 0 and prob_buffer:
        probabilities = np.mean(np.stack(prob_buffer, axis=0), axis=0)
    else:
        probabilities = infer_probabilities(model, test_features, batch_size, tta=int(config.params.get("tta", 0)), config=config)
    attn_outputs = None
    if config.params.get("subject_attn", False):
        if epoch_avg > 0 or swa_epochs > 0:
            raise ValueError("subject_attn cannot be combined with epoch_avg/swa")
        probabilities, pd_logits, weights = infer_probabilities_weights(model, test_features, batch_size)
        attn_outputs = (pd_logits, weights)
    state = cpu_state_dict(model) if return_state else None
    del model, optimizer, criterion, train_features, train_labels, test_features
    if discriminator is not None:
        del discriminator, subject_labels
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probabilities, state, meta, attn_outputs


def train_final_and_predict_cross(
    config, learning_rate: float, X_src, y_src, X_tgt, selected_epoch: int,
    device, seed: int, batch_size: int = DEFAULT_BATCH_SIZE,
):
    """Round-23 direct transfer: refit on ALL source windows for selected_epoch,
    predict target windows. Preprocessing is fit on the SOURCE cohort only and
    applied to the target (no target statistics are used anywhere).

    Returns (target_window_probabilities, attn_outputs_or_None, preprocess_meta).
    """
    if str(config.preprocess) == "subject_chmean":
        raise ValueError("subject_chmean requires per-subject statistics and is not "
                         "allowed in cross-cohort transfer (target leakage risk)")
    transform, meta = fit_preprocess(X_src, config.preprocess, config.input_transform,
                                     n_bands=config.n_bands)
    X_train = transform(X_src)
    X_test = transform(X_tgt)
    data_std = float(np.std(X_train, dtype=np.float64) + 1e-8) if config.augment != "none" else 1.0
    set_seed(seed)
    from dsmodel.models import build_model
    model = build_model(config, device)
    criterion = build_criterion(config, y_src, device)
    optimizer = build_optimizer(config, list(model.parameters()), learning_rate)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    train_features = torch.as_tensor(X_train, device=device)
    train_labels = torch.as_tensor(y_src, dtype=torch.int64, device=device)
    test_features = torch.as_tensor(X_test, device=device)
    for ep in range(1, int(selected_epoch) + 1):
        epoch_gen = torch.Generator(device="cpu").manual_seed(seed * 100003 + ep)
        train_one_epoch(model, optimizer, criterion, train_features, train_labels,
                        batch_size, epoch_gen, config, data_std, train_groups=None)
    attn_outputs = None
    if config.params.get("subject_attn", False):
        probabilities, pd_logits, weights = infer_probabilities_weights(model, test_features, batch_size)
        attn_outputs = (pd_logits, weights)
    else:
        probabilities = infer_probabilities(model, test_features, batch_size,
                                            tta=int(config.params.get("tta", 0)), config=config)
    del model, optimizer, criterion, train_features, train_labels, test_features
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probabilities, attn_outputs, meta
