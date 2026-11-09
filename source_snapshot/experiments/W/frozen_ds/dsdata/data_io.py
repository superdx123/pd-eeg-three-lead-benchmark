"""Data and manifest loading for the frozen ds002778 / ds003490 experiments.

Reads only (never writes) from the shared pd-eeg-off-classification project and
verifies fixed hashes from each dataset's manifest_config.json before use.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Absolute read-only paths on the shared server (no modification ever).
# DS_DATA_ROOT env var overrides the root for secondary compute servers
# (e.g. AutoDL 2080Ti) where the same frozen npz/manifests are mirrored.
PROJECT_ROOT = Path(os.environ.get("DS_DATA_ROOT", "/home/jerry/dx/pd-eeg-off-classification"))
MAIN_WINDOWS_NPZ = PROJECT_ROOT / "outputs/01_data_prep/02_build_main_window_dataset/main_windows_7ch_2s_nonoverlap.npz"
DS003490_WINDOWS_NPZ = PROJECT_ROOT / "outputs/12_second_cohort_replication/01_build_ds003490_off_window_dataset/ds003490_off_windows_3ch_2s.npz"
MANIFEST_DIR_2778 = PROJECT_ROOT / "outputs/11_dual_protocol_subject_generalization/01_build_dual_protocol_manifests"
MANIFEST_DIR_3490 = PROJECT_ROOT / "outputs/12_second_cohort_replication/02_build_ds003490_dual_protocol_manifests"
# Round-25 one-shot external cohort (built by code/run/prep_ds004584.py, frozen hashes).
DS004584_NPZ = PROJECT_ROOT / "ds004584/ds004584_windows_3ch_2s.npz"
DS004584_DIR = PROJECT_ROOT / "ds004584"

TARGET_CHANNELS = ("C3", "Cz", "C4")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _scalar(npz_obj, key: str):
    values = np.asarray(npz_obj[key]).reshape(-1)
    if len(values) != 1:
        raise ValueError(f"Expected scalar {key}, got {np.asarray(npz_obj[key]).shape}")
    return values[0].item()


@dataclass(frozen=True)
class Dataset:
    dataset_id: str
    X: np.ndarray            # float32 [N, 3, 512]
    y: np.ndarray            # int64 [N]
    groups: np.ndarray       # str [N]
    recording_ids: np.ndarray
    channel_names: tuple
    sfreq: float
    config: dict
    manifest_dir: Path
    extra_train_idx: np.ndarray = None  # row indices usable only as extra training data


def _load_combined_2778_plus_490() -> Dataset:
    """ds002778 evaluation data + ds003490 rows appended as extra training data.

    Groups from ds003490 are prefixed ('Q2:sub-001') to avoid any collision with
    ds002778 subject ids. Manifests are ds002778's frozen ones; ds003490 rows are
    only reachable through Dataset.extra_train_idx (never used for validation/test).
    """
    import hashlib
    base = load_dataset("ds002778")
    extra = load_dataset("ds003490")
    X = np.concatenate([base.X, extra.X], axis=0).astype(np.float32)
    y = np.concatenate([base.y, extra.y], axis=0).astype(np.int64)
    groups = np.concatenate([base.groups, np.asarray([f"Q2:{g}" for g in extra.groups])]).astype(str)
    recording_ids = np.concatenate([base.recording_ids, extra.recording_ids]).astype(str)
    extra_train_idx = np.arange(len(base.X), len(base.X) + len(extra.X), dtype=np.int64)
    return Dataset(
        dataset_id="ds002778_p490", X=X, y=y, groups=groups,
        recording_ids=recording_ids, channel_names=base.channel_names,
        sfreq=base.sfreq, config=base.config, manifest_dir=base.manifest_dir,
        extra_train_idx=extra_train_idx,
    )


def load_dataset(dataset_id: str) -> Dataset:
    dataset_id = str(dataset_id).lower()
    if dataset_id in ("ds002778", "2778", "778"):
        npz_path = MAIN_WINDOWS_NPZ
        manifest_dir = MANIFEST_DIR_2778
        dataset_id = "ds002778"
        use_7ch = False
    elif dataset_id in ("ds002778_7ch", "2778_7ch", "778_7ch"):
        npz_path = MAIN_WINDOWS_NPZ
        manifest_dir = MANIFEST_DIR_2778
        dataset_id = "ds002778_7ch"
        use_7ch = True
    elif dataset_id in ("ds003490", "3490", "490"):
        npz_path = DS003490_WINDOWS_NPZ
        manifest_dir = MANIFEST_DIR_3490
        dataset_id = "ds003490"
    elif dataset_id in ("ds002778_p490", "778_p490"):
        # Experimental multi-cohort training: evaluate on ds002778 manifests,
        # extra training rows come from ds003490 (never test subjects here).
        return _load_combined_2778_plus_490()
    elif dataset_id in ("ds004584", "584"):
        return _load_ds004584()
    else:
        raise ValueError(f"Unknown dataset {dataset_id!r}")

    config = json.loads((manifest_dir / "manifest_config.json").read_text(encoding="utf-8"))

    with np.load(npz_path, allow_pickle=False) as npz:
        X_all = np.asarray(npz["X"], dtype=np.float32)
        y = np.asarray(npz["y"], dtype=np.int64).reshape(-1)
        groups = np.asarray(npz["groups"]).astype(str)
        recording_ids = np.asarray(npz["recording_ids"]).astype(str)
        channel_names = tuple(str(v) for v in npz["channel_names"])
        sfreq = float(_scalar(npz, "sfreq"))

    if dataset_id == "ds002778":
        indices = [channel_names.index(ch) for ch in TARGET_CHANNELS]
        X = np.asarray(X_all[:, indices, :], dtype=np.float32)
        expected_shape = tuple(config["input_shape"])
        if X_all.shape != expected_shape:
            raise ValueError(f"ds002778 npz shape {X_all.shape} != frozen {expected_shape}")
    elif dataset_id == "ds002778_7ch":
        X = X_all
        if tuple(X.shape) != tuple(config["input_shape"]):
            raise ValueError(f"ds002778 7ch npz shape {X.shape} != frozen {tuple(config['input_shape'])}")
    else:
        X = X_all
        if tuple(X.shape) != tuple(config["input_shape"]):
            raise ValueError(f"ds003490 npz shape {X.shape} != frozen {tuple(config['input_shape'])}")

    # Verify file + content hashes against frozen manifest config.
    actual_file = file_sha256(npz_path)
    if actual_file != str(config["input_npz_file_sha256"]):
        raise ValueError(f"{dataset_id} NPZ file hash mismatch: {actual_file}")
    if not np.isfinite(X).all():
        raise ValueError("Selected channels contain NaN/Inf")
    if len(np.unique(y).tolist()) != 2 or set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("Labels must be exactly {0, 1}")
    channels = TARGET_CHANNELS if dataset_id == "ds002778" else tuple(channel_names)
    return Dataset(
        dataset_id=dataset_id, X=X, y=y, groups=groups, recording_ids=recording_ids,
        channel_names=channels, sfreq=sfreq, config=config, manifest_dir=manifest_dir,
    )


def _load_ds004584() -> Dataset:
    """Round-25 one-shot external cohort (target only; never used for selection)."""
    config = json.loads((DS004584_DIR / "manifest_config.json").read_text(encoding="utf-8"))
    if file_sha256(DS004584_NPZ) != str(config["npz_file_sha256"]):
        raise ValueError("ds004584 NPZ file hash mismatch")
    with np.load(DS004584_NPZ, allow_pickle=False) as npz:
        X = np.asarray(npz["X"], dtype=np.float32)
        y = np.asarray(npz["y"], dtype=np.int64).reshape(-1)
        groups = np.asarray(npz["groups"]).astype(str)
        channel_names = tuple(str(v) for v in npz["channel_names"])
        sfreq = float(_scalar(npz, "sfreq"))
    if not np.isfinite(X).all():
        raise ValueError("ds004584 contains NaN/Inf")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("ds004584 labels must be exactly {0, 1}")
    return Dataset(
        dataset_id="ds004584", X=X, y=y, groups=groups,
        recording_ids=groups.copy(), channel_names=channel_names, sfreq=sfreq,
        config=config, manifest_dir=DS004584_DIR,
    )


def load_manifests(dataset: Dataset):
    d = dataset.manifest_dir
    w_manifest = pd.read_csv(d / "protocol_w_window_manifest.csv", encoding="utf-8-sig")
    s_outer = pd.read_csv(d / "protocol_s_outer_subject_manifest.csv", encoding="utf-8-sig")
    s_inner = pd.read_csv(d / "protocol_s_inner_subject_manifest.csv", encoding="utf-8-sig")
    audit = pd.read_csv(d / "dataset_subject_audit.csv", encoding="utf-8-sig")
    # Verify manifest hashes.
    for name, path in (("protocol_w_window_manifest.csv", d / "protocol_w_window_manifest.csv"),
                       ("protocol_s_outer_subject_manifest.csv", d / "protocol_s_outer_subject_manifest.csv"),
                       ("protocol_s_inner_subject_manifest.csv", d / "protocol_s_inner_subject_manifest.csv"),
                       ("dataset_subject_audit.csv", d / "dataset_subject_audit.csv")):
        expected = dataset.config["manifest_sha256"].get(name)
        if expected and file_sha256(path) != str(expected):
            raise ValueError(f"{dataset.dataset_id} manifest hash mismatch: {name}")
    return w_manifest, s_outer, s_inner, audit
