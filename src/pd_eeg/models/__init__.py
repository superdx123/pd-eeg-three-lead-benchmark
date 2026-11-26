"""Model constructors for the manuscript's neural configurations."""

from __future__ import annotations

import torch
from torch import nn

from .bands import BANDS, expand_bands, prepare_input, repeat_raw
from .be_msnet import BEMSNet, parameter_count

SUPPORTED_MODELS = (
    "BE-MSNet", "4L-CNN", "EEGNet", "EEGConformer", "ATCNet", "EEGNeX",
    "FBCNet", "Lee-CRNN", "PSD-AE-RBFNN", "TDCNet",
)


def build_model(name: str, *, device: str | torch.device = "cpu", variant: str = "full",
                **kwargs) -> nn.Module:
    """Construct a network without fitting or reading data.

    BE-MSNet expects prepared 18-channel input (3 channels for ``raw_only``).
    ``DS28`` is accepted as the legacy alias for BE-MSNet.
    For PSD-AE-RBFNN, use ``stage='autoencoder'`` before fitting, or supply
    ``initial_centers`` and ``initial_sigma`` from fitted training embeddings.
    TDCNet returns a contrastive representation model; its SVM is fitted later.
    """
    key = name.lower().replace("-", "").replace("_", "")
    if key in {"bemsnet", "ds28", "ds28brmsauglswd3"}:
        if kwargs:
            raise TypeError(f"Unexpected BE-MSNet arguments: {sorted(kwargs)}")
        return BEMSNet(variant).to(device)
    if variant != "full":
        raise ValueError("The variant argument applies only to BE-MSNet.")
    if key in {"4lcnn", "fourlayercnn"}:
        from .cnn import FourLayerCNN
        model = FourLayerCNN(**kwargs)
    elif key == "eegnet":
        from .eegnet import EEGNet
        model = EEGNet(**kwargs)
    elif key == "leecrnn":
        from .lee_crnn import LeeCRNN
        model = LeeCRNN(**kwargs)
    elif key in {"psdaerbfnn", "psdae"}:
        from .psd_ae import PSDAutoencoder, PSDAERBFNN, TrainableRBFNN
        stage = kwargs.pop("stage", "combined")
        if stage == "autoencoder":
            model = PSDAutoencoder(**kwargs)
        elif stage == "rbf":
            model = TrainableRBFNN(**kwargs)
        elif stage == "combined":
            if "initial_centers" not in kwargs or "initial_sigma" not in kwargs:
                raise ValueError("PSD-AE-RBFNN requires centers and sigma fitted on training embeddings.")
            model = PSDAERBFNN(**kwargs)
        else:
            raise ValueError(f"Unknown PSD-AE-RBFNN stage: {stage}")
    elif key == "tdcnet":
        from .tdcnet import TDCContrastiveModel
        model = TDCContrastiveModel(**kwargs)
    elif key in {"eegconformer", "atcnet", "eegnex", "fbcnet"}:
        if kwargs:
            raise TypeError(f"Unexpected Braindecode arguments: {sorted(kwargs)}")
        from .braindecode import build_model as build_braindecode
        canonical = {value.lower(): value for value in SUPPORTED_MODELS}
        return build_braindecode(canonical[key], torch.device(device))
    else:
        raise ValueError(f"Unknown neural model: {name}. Choose from {SUPPORTED_MODELS}.")
    return model.to(device)


__all__ = [
    "BANDS", "BEMSNet", "SUPPORTED_MODELS", "build_model", "expand_bands",
    "parameter_count", "prepare_input", "repeat_raw",
]
