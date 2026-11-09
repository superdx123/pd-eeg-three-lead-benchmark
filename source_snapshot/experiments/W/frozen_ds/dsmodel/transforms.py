"""Deterministic channel-input transforms derived from the fixed 3 channels.

All transforms are fixed functions of the input signal (no learned params, no
labels) so they can be applied identically to train/validation/test.
"""
from __future__ import annotations

import numpy as np
import torch


def bipolar_channels(x: np.ndarray) -> np.ndarray:
    """Append C3-C4, C3-Cz, Cz-C4 difference signals -> 6 channels."""
    c3, cz, c4 = x[:, 0], x[:, 1], x[:, 2]
    extra = np.stack([c3 - c4, c3 - cz, cz - c4], axis=1).astype(np.float32)
    return np.concatenate([x, extra], axis=1)


BANDS = ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 40.0))
BANDS_8 = ((0.5, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 11.0), (11.0, 14.0),
           (14.0, 20.0), (20.0, 30.0), (30.0, 45.0))
_SFREQ = 256.0
_N = 512
_FREQS = None


def _band_mask(lo: float, hi: float) -> np.ndarray:
    global _FREQS
    if _FREQS is None:
        _FREQS = np.fft.rfftfreq(_N, d=1.0 / _SFREQ)
    return ((_FREQS >= lo) & (_FREQS <= hi)).astype(np.float32)


def bandpass_channels(x: np.ndarray, bands=BANDS) -> np.ndarray:
    """Decompose each channel into fixed bands (zero-phase FFT); default 5 bands."""
    n = x.shape[0]
    out = np.empty((n, len(bands) * x.shape[1], _N), dtype=np.float32)
    spectrum = np.fft.rfft(x.astype(np.float64), axis=-1)
    for b, (lo, hi) in enumerate(bands):
        mask = _band_mask(lo, hi)
        band = np.fft.irfft(spectrum * mask, n=_N, axis=-1).astype(np.float32)
        out[:, b * x.shape[1]:(b + 1) * x.shape[1], :] = band
    return out


def band_raw_channels(x: np.ndarray, bands=BANDS) -> np.ndarray:
    """Concatenate raw channels with band-decomposed channels (raw + bands)."""
    return np.concatenate([x, bandpass_channels(x, bands=bands)], axis=1)


COH_PAIRS = ((0, 1), (0, 2), (1, 2))  # C3-Cz, C3-C4, Cz-C4


def band_coh_channels(x: np.ndarray, bands=BANDS) -> np.ndarray:
    """Band-limited cross-channel correlation time series per pair.

    For each pair (i, j) and band, returns irfft(S_i(f) * conj(S_j(f)) * mask(f)),
    i.e. the zero-lag-centered cross-correlation restricted to the band. Captures
    phase/coherence structure between C3/Cz/C4 that per-channel band power misses.
    """
    n = x.shape[0]
    out = np.empty((n, len(bands) * len(COH_PAIRS), _N), dtype=np.float32)
    spectrum = np.fft.rfft(x.astype(np.float64), axis=-1)
    for b, (lo, hi) in enumerate(bands):
        mask = _band_mask(lo, hi)
        for p, (i, j) in enumerate(COH_PAIRS):
            cross = spectrum[:, i, :] * np.conj(spectrum[:, j, :]) * mask
            out[:, b * len(COH_PAIRS) + p, :] = np.fft.irfft(cross, n=_N, axis=-1).astype(np.float32)
    return out


def band_raw_coh_channels(x: np.ndarray, bands=BANDS) -> np.ndarray:
    """Raw + band-decomposed + band-limited cross-channel coherence (3+15+15=33ch)."""
    return np.concatenate([
        x,
        bandpass_channels(x, bands=bands),
        band_coh_channels(x, bands=bands),
    ], axis=1)


def _bands_for(n_bands: int):
    if int(n_bands) == 5:
        return BANDS
    if int(n_bands) == 8:
        return BANDS_8
    raise ValueError(f"Unsupported n_bands {n_bands!r}")


def transform_input(x: np.ndarray, mode: str, n_bands: int = 5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    bands = _bands_for(n_bands)
    if mode == "raw":
        return x
    if mode == "bipolar":
        return bipolar_channels(x)
    if mode == "band":
        return bandpass_channels(x, bands=bands)
    if mode == "band_bipolar":
        return bandpass_channels(bipolar_channels(x), bands=bands)
    if mode == "band_raw":
        return band_raw_channels(x, bands=bands)
    if mode == "band_raw_coh":
        return band_raw_coh_channels(x, bands=bands)
    raise ValueError(f"Unknown input transform {mode!r}")


def transform_input_torch(x: torch.Tensor, mode: str, n_bands: int = 5) -> torch.Tensor:
    if mode == "raw":
        return x
    return torch.as_tensor(transform_input(x.detach().cpu().numpy(), mode, n_bands=n_bands), device=x.device, dtype=x.dtype)


def n_input_channels(mode: str, base: int = 3, n_bands: int = 5) -> int:
    if mode == "raw":
        return base
    if mode == "bipolar":
        return base * 2
    if mode == "band":
        return base * int(n_bands)
    if mode == "band_bipolar":
        return base * int(n_bands) * 2
    if mode == "band_raw":
        return base * (1 + int(n_bands))
    if mode == "band_raw_coh":
        return base * (1 + int(n_bands)) + len(COH_PAIRS) * int(n_bands)
    raise ValueError(f"Unknown input transform {mode!r}")
