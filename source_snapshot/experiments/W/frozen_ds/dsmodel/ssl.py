"""Self-supervised pretraining for the DS model family (Round-10 candidates).

Two pretext tasks, both run inside each outer fold on the outer-train windows
only (the outer-test subject is NEVER touched):
  1. "masked"     - MAE-style masked reconstruction: mask random time segments,
                    reconstruct masked values with a light decoder, MSE on masked
                    positions only.
  2. "contrastive"- SimCLR-style: two light-augmented views per window, InfoNCE
                    (NT-Xent) on projected embeddings.

Pretraining is label-free and only initializes the feature extractor
(encoder / spectral / fusion_proj); the supervised head is always trained from
scratch inside the frozen inner-selection protocol. The window_zscore transform
is per-window, so features seen in pretraining are identical to fine-tuning.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dstraining.trainer import fit_preprocess, set_seed

FEATURE_PREFIXES = ("encoder.", "spectral.", "spectral_net.", "fusion_proj.")


class MaskedReconDecoder(nn.Module):
    def __init__(self, embed_dim: int, n_channels: int, n_times: int, hidden: int = 256):
        super().__init__()
        self.n_channels = int(n_channels)
        self.n_times = int(n_times)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, self.n_channels * self.n_times, bias=True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1, self.n_channels, self.n_times)


class SimCLRProjection(nn.Module):
    def __init__(self, embed_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim, bias=True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def feature_state_dict(model: nn.Module) -> dict:
    """Subset of model state dict that defines the feature extractor (no head)."""
    return {k: v for k, v in model.state_dict().items() if k.startswith(FEATURE_PREFIXES)}


def load_feature_state(model: nn.Module, state: dict) -> None:
    """Load pretrained feature weights into a freshly built model (strict=False on head)."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in pretrained state: {sorted(unexpected)}")
    bad = [k for k in missing if k.startswith(FEATURE_PREFIXES)]
    if bad:
        raise RuntimeError(f"Missing feature keys after pretrained load: {bad}")


def _masked_inputs(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero out masked time segments across all channels. mask: (N, n_seg) bool (True=keep)."""
    n_seg = mask.shape[1]
    seg_len = x.shape[-1] // n_seg
    xm = x.reshape(x.shape[0], x.shape[1], n_seg, seg_len)
    xm = xm * mask.to(x.device).unsqueeze(1).unsqueeze(-1).to(x.dtype)
    return xm.reshape_as(x)


def _masked_recon_loss(rec: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE only on masked (False) positions."""
    n_seg = mask.shape[1]
    seg_len = x.shape[-1] // n_seg
    rec4 = rec.reshape(x.shape[0], x.shape[1], n_seg, seg_len)
    x4 = x.reshape_as(rec4)
    m = (~mask).to(x.device).unsqueeze(1).unsqueeze(-1).to(x.dtype)
    diff = (rec4 - x4) ** 2
    return (diff * m).sum() / m.sum().clamp(min=1.0)


def _light_view(x: torch.Tensor, generator: torch.Generator, std_scale: float = 1.0) -> torch.Tensor:
    """One augmented view: light noise + amplitude (same as 'light' augment) + small time shift."""
    n = x.shape[0]
    device = x.device
    v = x
    noise = torch.randn(x.shape, generator=generator, device="cpu").to(device) * (0.05 * std_scale)
    scale = 1.0 + (torch.rand(n, 1, 1, generator=generator, device="cpu") - 0.5).to(device) * 0.2
    v = v * scale + noise
    shift = torch.randint(-16, 17, (n,), generator=generator, device="cpu")
    for i in range(n):
        s = int(shift[i].item())
        if s != 0:
            v[i] = torch.roll(v[i], s, dims=-1)
    # channel dropout: zero a random channel for a fraction of windows
    drop = torch.rand(n, generator=generator, device="cpu") < 0.1
    if drop.any():
        ch = torch.randint(0, x.shape[1], (n,), generator=generator, device="cpu")
        idx = torch.nonzero(drop).flatten()
        v[idx.to(device), ch[idx].to(device)] = 0.0
    return v


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=-1)
    sim = z @ z.T / temperature
    n = z1.shape[0]
    device = z.device
    sim.masked_fill_(torch.eye(2 * n, dtype=torch.bool, device=device), -1e9)
    labels = torch.cat([torch.arange(n, device=device) + n, torch.arange(n, device=device)])
    return F.cross_entropy(sim, labels)


def pretrain_masked(config, x_pre: torch.Tensor, device, seed: int,
                    n_epochs: int = 40, batch_size: int = 256, lr: float = 1e-3,
                    wd: float = 1e-3, seg_len: int = 16, mask_ratio: float = 0.5) -> dict:
    """MAE-style masked reconstruction pretraining. Returns feature state dict."""
    from dsmodel.models import build_model
    set_seed(seed)
    model = build_model(config, device)
    decoder = MaskedReconDecoder(config.embedding_dim, x_pre.shape[1], x_pre.shape[2]).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=lr, weight_decay=wd)
    n_seg = x_pre.shape[-1] // seg_len
    n_masked = max(1, int(round(n_seg * mask_ratio)))
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed * 7919 + 7)
    n = x_pre.shape[0]
    model.train()
    decoder.train()
    for _ep in range(1, int(n_epochs) + 1):
        perm = torch.randperm(n, generator=rng)
        for start in range(0, n, int(batch_size)):
            idx = perm[start:start + int(batch_size)]
            xb = x_pre[idx]
            # vectorized per-sample mask: keep n_seg-n_masked segments
            ranks = torch.argsort(torch.rand(xb.shape[0], n_seg, generator=rng), dim=-1)
            mask = (ranks >= n_masked).to(device)  # True = keep
            xm = _masked_inputs(xb, mask)
            rec = decoder(model.embed(xm))
            loss = _masked_recon_loss(rec, xb, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return feature_state_dict(model)


def pretrain_contrastive(config, x_pre: torch.Tensor, device, seed: int,
                         n_epochs: int = 40, batch_size: int = 256, lr: float = 1e-3,
                         wd: float = 1e-3, temperature: float = 0.1) -> dict:
    """SimCLR-style contrastive pretraining. Returns feature state dict."""
    from dsmodel.models import build_model
    set_seed(seed)
    model = build_model(config, device)
    proj = SimCLRProjection(config.embedding_dim).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(proj.parameters()), lr=lr, weight_decay=wd)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed * 7919 + 13)
    n = x_pre.shape[0]
    model.train()
    proj.train()
    for _ep in range(1, int(n_epochs) + 1):
        perm = torch.randperm(n, generator=rng)
        for start in range(0, n, int(batch_size)):
            idx = perm[start:start + int(batch_size)]
            xb = x_pre[idx]
            z1 = proj(model.embed(_light_view(xb, rng)))
            z2 = proj(model.embed(_light_view(xb, rng)))
            loss = nt_xent_loss(z1, z2, temperature=temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return feature_state_dict(model)


PRETRAIN_CACHE: dict = {}


def ssl_mode(config) -> str:
    return str(config.params.get("ssl", "") or "")


def pretrain_key(config, dataset_id, protocol, repeat_index, outer_fold) -> str | None:
    mode = ssl_mode(config)
    if not mode:
        return None
    return f"{dataset_id}:{protocol}:{repeat_index}:{outer_fold}:{mode}"


def ensure_pretrained(config, dataset_id, X, y, groups, train_idx, device, seed,
                      protocol, repeat_index, outer_fold, batch_size=256):
    """Pretrain once per (fold) on the given train windows; cache by fold key.

    Unsupervised, uses only train-fold windows (outer-test never touched).
    Returns the cached feature state dict (or None when ssl disabled).
    """
    if not ssl_mode(config) or dataset_id is None:
        return None
    key = pretrain_key(config, dataset_id, protocol, repeat_index, outer_fold)
    if key in PRETRAIN_CACHE:
        return PRETRAIN_CACHE[key]
    transform, _meta = fit_preprocess(X[train_idx], config.preprocess, config.input_transform, n_bands=config.n_bands)
    x_pre = torch.as_tensor(transform(X[train_idx]), device=device)
    state = pretrain(config, x_pre, device, seed, batch_size=batch_size)
    PRETRAIN_CACHE[key] = state
    return state


def apply_pretrained(model, config, dataset_id, protocol, repeat_index, outer_fold) -> bool:
    """Load the cached pretrained feature weights into `model` (head stays fresh)."""
    if not ssl_mode(config) or dataset_id is None:
        return False
    key = pretrain_key(config, dataset_id, protocol, repeat_index, outer_fold)
    if key is None or key not in PRETRAIN_CACHE:
        return False
    load_feature_state(model, PRETRAIN_CACHE[key])
    return True


def pretrain(config, x_pre: torch.Tensor, device, seed: int, batch_size: int = 256) -> dict:

    """Dispatch on config.params['ssl']."""
    mode = str(config.params.get("ssl", ""))
    kw = dict(
        n_epochs=int(config.params.get("ssl_epochs", 40)),
        batch_size=batch_size,
        lr=float(config.params.get("ssl_lr", 1e-3)),
        wd=float(config.params.get("wd", 1e-3)),
    )
    if mode == "masked":
        return pretrain_masked(config, x_pre, device, seed,
                               seg_len=int(config.params.get("ssl_seg_len", 16)),
                               mask_ratio=float(config.params.get("ssl_mask_ratio", 0.5)), **kw)
    if mode == "contrastive":
        return pretrain_contrastive(config, x_pre, device, seed,
                                    temperature=float(config.params.get("ssl_temperature", 0.1)), **kw)
    raise ValueError(f"Unknown ssl mode: {mode!r}")
