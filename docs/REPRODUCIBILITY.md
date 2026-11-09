# Reproducing the archived experiment

## 1. Identify the exact archive

The file `archive.lock.json` records the accepted v3 archive's SHA-256 and size.
No weights, EEG or individual records are stored in this Git repository.

```sh
python -B tools/prepare_source_snapshot.py --verify --archive "/path/to/ds28_778_LOCAL_REVIEW_CANDIDATE_v3.zip"
```

Obtain the archive through the channel authorized by the owner. A public URL and
fixed release identifier will be added only when actually created. Extract the
verified archive into a separate directory, preserving all its relative paths.
Do not combine the reduced Git source subset with a different input or model archive.

## 2. Select the intended operation

| Operation | Entry point inside the full archive | Requirements |
| --- | --- | --- |
| Verify saved aggregate scores | `verify_scores.py` | Standard library and saved prediction tables |
| Check reduced checkpoint structure | `predict_saved_weights.py --check-architecture-only` | PyTorch and archived model code; no real-EEG forward pass |
| Predict from a reduced checkpoint | `predict_saved_weights.py --execute` | Original pinned CUDA environment and complete archived inputs |
| Perform a separate new training run | `experiments/S/scripts/run_fold.py` or `experiments/W/scripts/run_fold.py` | Original pinned environment, fixed inputs and explicit task arguments |
| Validate a newly trained full checkpoint | Corresponding `scripts/validate_fold.py` | Same native environment and full checkpoint metadata |

The source root's `--verify` command only hashes files and parses Python; it does
not call any of these scientific entry points. Use normal Python mode (`python -B`),
not `python -O`; the archived scripts retain some assertion-based guards. Legacy
helper defaults are described in `ENTRYPOINTS_AND_ATTRIBUTION.md`.

## 3. Execution environments

Integrity tools and tests use Python 3.10+ and the standard library. The native
training and full-checkpoint validation used Python 3.10.8, PyTorch 2.5.1+cu121,
CUDA 12.1, NumPy 2.2.6, pandas 2.3.3 and scikit-learn 1.7.2 on an RTX 3080 Ti.
Use `source_snapshot/ENVIRONMENT.md` and the unchanged protocol JSON files for the
recorded details. Do not relax the archived environment checks to claim exact replay.

## Verification scope

- All 155 S and 125 W original full checkpoints completed native independent
  prediction verification. Their original full files remain in the internal archive.
- Each reduced model copy preserves all parameter tensors exactly, with unnecessary
  machine/environment and individual-index metadata removed.
- The reduced files were reserialized using CPU PyTorch 2.10.0; the adapter passed
  S/W CPU structure checks. A new real-EEG forward replay of these reduced copies
  has not been performed during release preparation.
- Saved-prediction aggregate-score verification passed. It is distinct from model
  inference and does not restore missing historical preprocessing records.
- The supplied S/W configuration and accepted results concern ds002778. The full
  two-cohort/all-baseline manuscript pipeline is outside this first package.

The full archive's versioned manifests bind its files. The source repository's
manifest inventories only this Git subset and must not replace those bindings.
