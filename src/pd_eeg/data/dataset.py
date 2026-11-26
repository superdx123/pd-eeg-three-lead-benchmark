"""Validated, non-pickled NPZ input format shared by all experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np


CHANNELS = ("C3", "Cz", "C4")
STORED_778_CHANNELS = ("FC1", "FC2", "C3", "Cz", "C4", "CP1", "CP2")
PROCESSED_SHA256 = {
    "ds002778": "f00315271235e23cff51c28604c20e2ec151ca30508953cef44cc4dc1deaf06b",
    "ds003490": "87e44b301c4651335d6576561108861c2b11611ef288d0cbd629465c652c1d8a",
}


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    recording_ids: np.ndarray
    channel_names: tuple[str, ...]
    sfreq: float
    window_sec: float
    step_sec: float = 2.0


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_output_path(path: str | Path, workspace: str | Path = ".") -> Path:
    """Resolve an output below the selected workspace, rejecting escapes."""
    base = Path(workspace).resolve()
    value = Path(path)
    target = (value if value.is_absolute() else base / value).resolve()
    if not target.is_relative_to(base):
        raise ValueError("Output path must remain inside the selected workspace")
    return target


def _scalar(payload, key: str, default=None) -> float:
    if key not in payload:
        if default is None:
            raise ValueError(f"Missing scalar field: {key}")
        return float(default)
    value = np.asarray(payload[key])
    if value.size != 1:
        raise ValueError(f"{key} must contain one scalar")
    result = float(value.reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError(f"{key} is not finite")
    return result


def validate_dataset(dataset: Dataset, *, cohort: str | None = None) -> None:
    X, y = dataset.X, dataset.y
    if X.dtype != np.float32 or X.ndim != 3 or X.shape[1:] != (3, 512):
        raise ValueError("X must be float32 with shape (windows, 3, 512)")
    if not len(X) or not np.isfinite(X).all():
        raise ValueError("X must be nonempty and finite")
    if y.dtype != np.int64 or y.shape != (len(X),) or set(np.unique(y)) != {0, 1}:
        raise ValueError("y must be int64 binary labels with one value per window")
    if dataset.channel_names != CHANNELS or dataset.sfreq != 256.0:
        raise ValueError("Input channels must be C3/Cz/C4 at 256 Hz")
    if dataset.window_sec != 2.0 or dataset.step_sec != 2.0:
        raise ValueError("The benchmark uses non-overlapping 2 s windows")
    for name, values in (("groups", dataset.groups), ("recording_ids", dataset.recording_ids)):
        if values.shape != (len(X),) or values.dtype.kind not in "US":
            raise ValueError(f"{name} must be a string vector with one value per window")
        if any(not str(value).strip() for value in values):
            raise ValueError(f"{name} contains a blank value")
    subjects = np.unique(dataset.groups)
    counts = {0: 0, 1: 0}
    for subject in subjects:
        selected = dataset.groups == subject
        labels = np.unique(y[selected])
        if len(labels) != 1 or len(np.unique(dataset.recording_ids[selected])) != 1:
            raise ValueError("Each subject must have one label and one selected recording")
        counts[int(labels[0])] += 1
    expected = {"ds002778": (3022, 31, {0: 16, 1: 15}), "ds003490": (1348, 50, {0: 25, 1: 25})}
    if cohort is not None:
        if cohort not in expected:
            raise ValueError("Supported cohorts are ds002778 and ds003490")
        windows, people, classes = expected[cohort]
        if len(X) != windows or len(subjects) != people or counts != classes:
            raise ValueError(f"Dataset does not match the {cohort} study cohort")


def load_dataset(path: str | Path, *, expected_sha256: str | None = None,
                 cohort: str | None = None) -> Dataset:
    """Load prepared windows and select the classifier channels without reprocessing."""
    path = Path(path)
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise ValueError("Dataset SHA-256 does not match the specified input")
    with np.load(path, allow_pickle=False) as payload:
        required = {"X", "y", "groups", "channel_names", "sfreq", "window_sec"}
        if required - set(payload.files):
            raise ValueError(f"Missing NPZ fields: {sorted(required - set(payload.files))}")
        names = tuple(str(value) for value in payload["channel_names"].reshape(-1))
        if names not in (CHANNELS, STORED_778_CHANNELS):
            raise ValueError("Unsupported stored channel order")
        stored = payload["X"]
        if stored.ndim != 3 or stored.shape[1] != len(names):
            raise ValueError("Stored X and channel_names disagree")
        X = stored[:, [names.index(value) for value in CHANNELS], :].copy()
        y = payload["y"].copy()
        groups = payload["groups"].astype(str)
        # A single-recording-per-subject file may omit a redundant recording vector.
        recordings = payload["recording_ids"].astype(str) if "recording_ids" in payload else groups.copy()
        dataset = Dataset(X, y, groups, recordings, CHANNELS, _scalar(payload, "sfreq"),
                          _scalar(payload, "window_sec"), _scalar(payload, "step_sec", 2.0))
    validate_dataset(dataset, cohort=cohort)
    return dataset


def save_dataset(dataset: Dataset, destination: str | Path, *, extra_fields: dict | None = None) -> Path:
    """Write a new dataset atomically; an existing file is never overwritten."""
    validate_dataset(dataset)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(X=dataset.X, y=dataset.y, groups=dataset.groups,
                   recording_ids=dataset.recording_ids, channel_names=np.asarray(dataset.channel_names),
                   sfreq=np.asarray(dataset.sfreq), window_sec=np.asarray(dataset.window_sec),
                   step_sec=np.asarray(dataset.step_sec))
    extra = extra_fields or {}
    if set(payload) & set(extra):
        raise ValueError("Extra fields cannot replace required dataset fields")
    payload.update(extra)
    if any(np.asarray(value).dtype.kind == "O" for value in payload.values()):
        raise ValueError("Dataset fields cannot contain Python object arrays")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists():
            raise FileExistsError(destination)
        os.replace(temporary, destination)
        return destination
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def import_preprocessed(source: str | Path, destination: str | Path, *, cohort: str,
                        expected_sha256: str | None = None) -> Dataset:
    """Import existing standardized inputs by indexing channels, without another z-score."""
    dataset = load_dataset(source, expected_sha256=expected_sha256, cohort=cohort)
    save_dataset(dataset, destination)
    return dataset
