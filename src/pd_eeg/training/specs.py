"""Fixed configurations used in the manuscript."""
from __future__ import annotations

from dataclasses import dataclass

SEEDS = (11, 22, 33, 44, 55)
MATCHED_SEEDS = (11, 22, 33)
MAIN_MODELS = ("BE-MSNet", "EEGConformer", "EEGNeX", "ATCNet", "FBCNet", "EEGNet",
               "4L-CNN", "Lee-CRNN", "PSD-AE-RBFNN", "RBF-SVM", "Spectral-LR")
BRAINDDECODE_MODELS = frozenset(("EEGConformer", "EEGNeX", "ATCNet", "FBCNet"))
ALIASES = {"DS28": "BE-MSNet", "PrimaryCNN": "4L-CNN", "LeeCRNN": "Lee-CRNN",
           "PSDAE": "PSD-AE-RBFNN", "RBFSVM": "RBF-SVM", "LR": "Spectral-LR",
           "HarmonizedLR": "Harmonized-LR", "TDCNet-strict": "TDCNet",
           "TDCNet-dagger": "TDCNet-reference"}


def model_name(name: str) -> str:
    result = ALIASES.get(name, name)
    if result not in (*MAIN_MODELS, "Harmonized-LR", "TDCNet", "TDCNet-reference"):
        raise ValueError(f"Unknown model: {name}")
    return result


@dataclass(frozen=True)
class NeuralPolicy:
    batch_size: int
    max_epochs: int
    patience: int
    learning_rates: tuple[float, ...]
    weight_decay: float
    weighted: bool = True
    label_smoothing: float = 0.0
    augmentation: bool = False
    online_stopping: bool = True


def neural_policy(name: str) -> NeuralPolicy:
    name = model_name(name)
    if name == "BE-MSNet":
        return NeuralPolicy(256, 80, 12, (3e-4, 1e-3), 1e-3,
                            label_smoothing=.05, augmentation=True)
    if name in BRAINDDECODE_MODELS:
        return NeuralPolicy(64, 80, 12, (3e-4, 1e-3), 1e-4)
    if name in ("EEGNet", "4L-CNN"):
        return NeuralPolicy(64, 80, 12, (1e-3,), 1e-4, online_stopping=False)
    if name == "Lee-CRNN":
        return NeuralPolicy(64, 100, 12, (1e-3,), 0., weighted=False, online_stopping=False)
    raise ValueError(f"{name} does not use supervised neural epoch selection")


def preferred(candidate: dict, current: dict | None, parameter: str = "learning_rate") -> bool:
    """bAcc, AUC, smaller parameter, earlier epoch; metric ties use 1e-12."""
    if current is None:
        return True
    for key in ("balanced_accuracy", "auc"):
        diff = float(candidate[key]) - float(current[key])
        if abs(diff) > 1e-12:
            return diff > 0
    for key in (parameter, "epoch"):
        a, b = float(candidate.get(key, 0)), float(current.get(key, 0))
        if a != b:
            return a < b
    return False


