"""Fixed waveform representations for C3, Cz and C4 at 256 Hz."""

from __future__ import annotations

import numpy as np

SAMPLING_RATE = 256.0
WINDOW_SAMPLES = 512
BANDS = ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 40.0))


def _windows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (3, WINDOW_SAMPLES):
        raise ValueError("Expected windows shaped [N, 3, 512].")
    if not np.isfinite(values).all():
        raise ValueError("Windows contain non-finite values.")
    return values


def bandpass_channels(values: np.ndarray) -> np.ndarray:
    """Return five groups of three band-limited waveforms, in band-major order.

    Masks include both boundaries, so adjacent bands share their boundary bin.
    This is an FFT mask operation, not a causal or learned bandpass filter.
    """
    values = _windows(values)
    frequencies = np.fft.rfftfreq(WINDOW_SAMPLES, d=1.0 / SAMPLING_RATE)
    spectrum = np.fft.rfft(values.astype(np.float64), axis=-1)
    output = np.empty((len(values), len(BANDS) * 3, WINDOW_SAMPLES), dtype=np.float32)
    for band, (low, high) in enumerate(BANDS):
        mask = ((frequencies >= low) & (frequencies <= high)).astype(np.float32)
        output[:, band * 3:(band + 1) * 3] = np.fft.irfft(
            spectrum * mask, n=WINDOW_SAMPLES, axis=-1
        ).astype(np.float32)
    return output


def expand_bands(values: np.ndarray) -> np.ndarray:
    """Concatenate the original triplet and its five band-limited triplets."""
    values = _windows(values)
    return np.concatenate((values, bandpass_channels(values)), axis=1)


def repeat_raw(values: np.ndarray) -> np.ndarray:
    """Concatenate six raw triplets along the channel axis for the R18 control."""
    return np.concatenate([_windows(values)] * 6, axis=1)


def prepare_input(values: np.ndarray, variant: str = "full") -> np.ndarray:
    """Apply the representation only; training-fitted scaling is separate."""
    if variant == "raw_only":
        return _windows(values)
    if variant == "r18":
        return repeat_raw(values)
    if variant in {"full", "single_scale", "mean_only"}:
        return expand_bands(values)
    raise ValueError(f"Unknown BE-MSNet variant: {variant}")
