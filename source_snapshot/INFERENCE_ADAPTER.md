# Reduced-checkpoint adapter: exact scope

`predict_saved_weights.py` uses the archived DS28 architecture and original
training-only preprocessing scalars. It never fits scalers, changes thresholds,
selects epochs, trains a network or modifies the saved input/weights.

Only the following structural checks were performed during candidate preparation:

```text
python -B predict_saved_weights.py --protocol S --seed 11 --fold 1 --check-architecture-only
python -B predict_saved_weights.py --protocol W --seed 11 --repeat 1 --fold 1 --check-architecture-only
```

They load CPU model tensors and verify 67,190 parameters and strict state keys and
shapes. They do not load EEG or execute a forward pass. Saved-prediction metric
verification by `verify_scores.py` is a different check; neither of these checks
is a new forward-prediction replay of reserialized weights on another runtime.

For a later explicit inference-only check in the pinned CUDA runtime:

```text
python -B predict_saved_weights.py --protocol S --seed 11 --fold 1 --execute
python -B predict_saved_weights.py --protocol W --seed 11 --repeat 1 --fold 1 --execute
```

Outputs go outside the payload to `replay_outputs/`, with refusal to overwrite.
The adapter compares subject-mean probabilities to the corresponding saved
seed/fold rows at 1e-7 and preserves a failing result rather than relaxing the
tolerance. Its scope is subject-mean comparison: it does not replace the original
native full-checkpoint per-window replay receipts. Repeated inference on a
different GPU may not meet the strict tolerance; such a result must be reported.

P-number IDs are deterministic recodings of the source identifiers, not a claim
of irreversible anonymization; source pseudonyms are also present in the licensed
NPZ and training splits. Original labels and probabilities are not reversed or
selected based on performance.

The current full payload is inventoried by `release_manifest_v2.json`. The earlier
`release_manifest.json` belongs to the preserved v1 archive, before adding this
adapter, and is retained as a versioned provenance artifact.
