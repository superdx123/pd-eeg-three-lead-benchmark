"""Window datasets, preprocessing and subject-aware data splits."""

from .dataset import Dataset, file_sha256, import_preprocessed, load_dataset, resolve_output_path, save_dataset
from .splits import (
    DEFAULT_W_SEEDS,
    build_nested_loso_manifests,
    build_protocol_w_manifest,
    build_time_blocks,
    blocked_roles,
    representation_folds,
    s_inner_splits,
    s_outer_indices,
    validate_s_manifests,
    validate_w_manifest,
    w_role_indices,
)

__all__ = [
    "Dataset", "file_sha256", "import_preprocessed", "load_dataset", "save_dataset",
    "resolve_output_path", "DEFAULT_W_SEEDS", "build_nested_loso_manifests",
    "build_protocol_w_manifest", "s_outer_indices", "s_inner_splits",
    "w_role_indices", "validate_s_manifests", "validate_w_manifest",
    "build_time_blocks", "blocked_roles", "representation_folds",
]
