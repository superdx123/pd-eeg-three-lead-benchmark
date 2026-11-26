"""Reconstruct the ds003490 eyes-open, medication-OFF analysis windows."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import CHANNELS, Dataset, file_sha256, save_dataset, validate_dataset


PARTICIPANTS_SHA256 = "e435a5d9b6906ad7f422847f38cdc3e7d381899a84d451d94828d649a51c3130"
EO_EVENT = "Eyes Open: Every 1000 ms"
EXCLUDED_EVENTS = {"Eyes Closed: Every 1000 ms", "Standard Tone", "Target Tone", "Novel Tone"}


def eo_interval(events: pd.DataFrame) -> tuple[float, float]:
    """Require one uninterrupted sequence of at least 55 eyes-open markers."""
    if not {"onset", "trial_type"}.issubset(events.columns):
        raise ValueError("Events require onset and trial_type columns")
    onsets = pd.to_numeric(events.loc[events.trial_type == EO_EVENT, "onset"], errors="raise").to_numpy(float)
    if len(onsets) < 55 or not np.isfinite(onsets).all():
        raise ValueError("Insufficient finite eyes-open event markers")
    gaps = np.diff(onsets)
    if not np.all((gaps > 0.90) & (gaps < 1.10)):
        raise ValueError("Eyes-open markers must form one continuous sequence")
    lo, hi = float(onsets[0]), float(onsets[-1])
    bad = events.trial_type.isin(EXCLUDED_EVENTS)
    times = pd.to_numeric(events.onset, errors="raise")
    if np.any(bad & times.between(lo, hi)):
        raise ValueError("Closed-eye or auditory events occur inside the eyes-open interval")
    return lo, hi


def inward_crop_bounds(lo: float, hi: float, sfreq: float = 500.0) -> tuple[float, float]:
    if not np.isfinite([lo, hi, sfreq]).all() or lo < 0 or hi <= lo or sfreq <= 0:
        raise ValueError("Invalid crop limits or sampling rate")
    start = math.ceil(lo * sfreq - 1e-7) / sfreq
    stop = math.floor(hi * sfreq + 1e-7) / sfreq
    if stop <= start:
        raise ValueError("Empty inward-rounded crop")
    return start, stop


def window_and_standardize(signal_uv: np.ndarray, crop_start: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Discard 2 s at each end, then apply non-overlapping 2 s QC and z-score."""
    if signal_uv.ndim != 2 or signal_uv.shape[0] != 3 or signal_uv.shape[1] <= 1024:
        raise ValueError("Expected a three-channel 256 Hz segment longer than 4 s")
    if not np.isfinite(signal_uv).all():
        raise ValueError("Non-finite signal")
    usable = signal_uv[:, 512:-512]
    count = usable.shape[1] // 512
    if count == 0:
        raise ValueError("No complete windows remain after edge exclusion")
    windows = usable[:, :count * 512].reshape(3, count, 512).transpose(1, 0, 2)
    peaks = np.abs(windows).max(axis=(1, 2))
    mean_std = windows.std(axis=2).mean(axis=1)
    keep = (peaks <= 150.0) & (mean_std >= 0.5) & (mean_std <= 80.0)
    if not keep.any():
        raise ValueError("No windows passed the fixed QC criteria")
    values = windows[keep]
    X = ((values - values.mean(axis=2, keepdims=True)) /
         (values.std(axis=2, keepdims=True) + 1e-8)).astype(np.float32)
    return X, {
        "window_start_seconds": crop_start + 2.0 + np.flatnonzero(keep) * 2.0,
        "window_peak_abs_uv": peaks[keep],
        "window_mean_channel_std_uv": mean_std[keep],
    }


def _rest_file(root: Path, subject: str, session: str) -> Path:
    folder = root / subject / session / "eeg"
    values = sorted(folder.glob("*task-Rest*_eeg.set"))
    if len(values) != 1:
        raise ValueError("Each selected session must have exactly one Rest EEG file")
    path = values[0].resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("EEG path escapes the dataset root")
    return path


def select_off_records(root: str | Path, *, selection_file: str | Path | None = None) -> pd.DataFrame:
    """Select the unique PD OFF visit and the unique HC Rest visit from BIDS metadata."""
    root = Path(root).resolve()
    people = pd.read_csv(root / "participants.tsv", sep="\t", encoding="utf-8-sig")
    required = {"participant_id", "Group", "sess1_Med", "sess2_Med"}
    if required - set(people.columns):
        raise ValueError(f"participants.tsv is missing {sorted(required - set(people.columns))}")
    if people.participant_id.astype(str).duplicated().any():
        raise ValueError("Duplicate participant rows")
    explicit = None
    if selection_file is not None:
        explicit = pd.read_csv(selection_file)
        if not {"participant_id", "session", "label"}.issubset(explicit.columns):
            raise ValueError("Selection CSV requires participant_id, session and label")
        if explicit.participant_id.astype(str).duplicated().any():
            raise ValueError("Selection CSV duplicates a participant")
        explicit["label"] = pd.to_numeric(explicit.label, errors="raise")
        if not explicit.label.isin([0, 1]).all():
            raise ValueError("Selection labels must be 0 or 1")
        if set(explicit.participant_id.astype(str)) != set(people.participant_id.astype(str)):
            raise ValueError("Selection CSV must cover the same participants as the BIDS table")
        explicit = explicit.set_index("participant_id")
    rows = []
    for _, person in people.sort_values("participant_id").iterrows():
        subject = str(person.participant_id).strip()
        if not subject.startswith("sub-") or Path(subject).name != subject or "/" in subject or "\\" in subject:
            raise ValueError("Invalid BIDS participant identifier")
        group = str(person.Group).strip().upper()
        if group == "PD":
            sessions = [name for name, value in (("ses-01", person.sess1_Med), ("ses-02", person.sess2_Med))
                        if str(value).strip().upper() == "OFF"]
            label = 1
        elif group == "CTL":
            sessions = []
            for folder in sorted((root / subject).glob("ses-*")):
                if len(list((folder / "eeg").glob("*task-Rest*_eeg.set"))) == 1:
                    sessions.append(folder.name)
            label = 0
        else:
            raise ValueError("Expected only PD and CTL groups in ds003490")
        if explicit is not None:
            selected = explicit.loc[subject]
            session = str(selected.session)
            if session not in sessions or int(selected.label) != label:
                raise ValueError("Explicit selection disagrees with group or medication metadata")
        else:
            if len(sessions) != 1:
                raise ValueError("Session selection is ambiguous; provide a metadata-consistent selection CSV")
            session = sessions[0]
        rows.append(dict(participant_id=subject, session=session, label=label,
                         eeg_path=_rest_file(root, subject, session).relative_to(root).as_posix()))
    frame = pd.DataFrame(rows)
    if len(frame) != 50 or frame.label.value_counts().sort_index().to_dict() != {0: 25, 1: 25}:
        raise ValueError("The study cohort requires 25 controls and 25 PD-OFF participants")
    return frame


def build_eo_dataset(root: str | Path, destination: str | Path, *, selection_file: str | Path | None = None,
                     expected_participants_sha256: str = PARTICIPANTS_SHA256) -> Dataset:
    """Prepare EO windows directly from the public BIDS files, with no project-specific inputs."""
    import mne

    root, destination = Path(root).resolve(), Path(destination)
    receipt = destination.with_suffix(".sources.json")
    if destination.exists() or receipt.exists():
        raise FileExistsError("Prepared data or its source manifest already exists")
    if file_sha256(root / "participants.tsv") != expected_participants_sha256:
        raise ValueError("participants.tsv differs from the specified ds003490 snapshot")
    selected = select_off_records(root, selection_file=selection_file)
    X_parts, labels, subjects, recordings = [], [], [], []
    extras = {name: [] for name in ("window_start_seconds", "window_peak_abs_uv", "window_mean_channel_std_uv")}
    sources = {"participants.tsv": file_sha256(root / "participants.tsv")}
    for row in selected.itertuples(index=False):
        path = root / row.eeg_path
        base = path.name.removesuffix("_eeg.set")
        events_path = path.parent / f"{base}_events.tsv"
        json_path = path.parent / f"{base}_eeg.json"
        channels_path = path.parent / f"{base}_channels.tsv"
        for source in (path, path.with_suffix(".fdt"), events_path, json_path, channels_path):
            sources[source.relative_to(root).as_posix()] = file_sha256(source)
        events = pd.read_csv(events_path, sep="\t")
        lo, hi = eo_interval(events)
        raw = mne.io.read_raw_eeglab(str(path), preload=False, verbose="ERROR")
        if float(raw.info["sfreq"]) != 500.0 or raw.first_samp != 0:
            raise ValueError("Expected the original 500 Hz recording with first_samp=0")
        start, stop = inward_crop_bounds(lo, hi, 500.0)
        raw.crop(tmin=start, tmax=stop, include_tmax=False).load_data()
        reference = [name for name in raw.ch_names if not (
            any(term in name.lower() for term in ("eog", "emg", "ecg"))
            or name.lower() in {"x", "y", "z", "status"} or name.lower().startswith("exg"))]
        if not set(CHANNELS).issubset(reference):
            raise ValueError("A classifier channel is absent from the scalp-channel selection")
        raw.pick(reference)
        raw.filter(1.0, 40.0, fir_design="firwin", verbose="ERROR")
        raw.resample(256.0, verbose="ERROR")
        raw.set_eeg_reference(ref_channels="average", verbose="ERROR")
        raw.pick(list(CHANNELS))
        if tuple(raw.ch_names) != CHANNELS:
            raise ValueError("Unexpected final channel order")
        signal = raw.get_data().astype(np.float64) * 1e6
        windows, metadata = window_and_standardize(signal, start)
        times = metadata["window_start_seconds"]
        if times.min() < lo + 2.0 - 1e-6 or (times + 2.0).max() > hi - 2.0 + 1 / 256:
            raise ValueError("Window boundary extends beyond the guarded EO interval")
        X_parts.append(windows)
        labels.extend([int(row.label)] * len(windows))
        subjects.extend([row.participant_id] * len(windows))
        recordings.extend([row.participant_id + "_" + row.session] * len(windows))
        for name, values in metadata.items():
            extras[name].append(values)
    for relative, digest in sources.items():
        if file_sha256(root / relative) != digest:
            raise ValueError("A raw source changed during preprocessing")
    dataset = Dataset(np.concatenate(X_parts), np.asarray(labels, dtype=np.int64), np.asarray(subjects),
                      np.asarray(recordings), CHANNELS, 256.0, 2.0, 2.0)
    validate_dataset(dataset, cohort="ds003490")
    counts = [int(np.sum(dataset.groups == subject)) for subject in np.unique(dataset.groups)]
    if min(counts) < 26 or max(counts) > 27:
        raise ValueError("Prepared study data must retain 26--27 windows per subject")
    save_dataset(dataset, destination, extra_fields={name: np.concatenate(values) for name, values in extras.items()})
    record = {"dataset_sha256": file_sha256(destination), "subjects": 50, "windows": len(dataset.y),
              "channels": list(CHANNELS), "sfreq": 256.0, "window_sec": 2.0,
              "raw_sources_sha256": sources}
    receipt.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return dataset
