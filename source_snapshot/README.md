# DS28 ds002778 S/W: local reproducibility release candidate

**Status: prepared locally, not uploaded, not yet a publicly licensed repository.**

This candidate contains the exact archived DS28 model/training/validation code,
fixed five-seed protocol, unchanged stored processed data and split definitions,
reduced-metadata model files, and pseudonymized prediction tables for the newly
completed ds002778 experiments. It is not recovery of missing early checkpoints.
It is the new ds002778 S/W reproducibility core, not a claim that every model,
baseline, cohort, paper statistic and figure has been integrated end-to-end.

- `experiments/S/`: nested subject-level LOSO, 155 final models in the original run.
- `experiments/W/`: five repeated five-fold within-record splits, 125 final models.
- `weights/`: 280 metadata-reduced model copies; every parameter tensor is bitwise
  unchanged. Do not substitute them for the full original validator checkpoints.
- `metadata/`: prediction/split tables with P-number IDs; all labels, probabilities,
  indices and remaining cells preserved. Training roots use the original public
  dataset pseudonyms to match NPZ groups. The two ID namespaces must not be mixed.
- `provenance/`: original code/data/FROZEN hashes for historical reference.
- `primary_metrics.json`: exact accepted aggregate S/W summaries and source hashes.
- `ENVIRONMENT.md` and `LICENSE_STATUS.md`: execution boundaries and release review.

## Result verification without training

Run `python -B verify_scores.py`. It only reads saved pseudonymized prediction
tables and independently recomputes S AUC/bAcc and W mean-of-25-fold AUC/bAcc.
It does not read EEG, load weights, fit a model or optimize a threshold.

Expected five-seed results, AUC/bAcc: S=0.8458/0.7417; W=0.9913/0.9378.
The W estimator is the mean of 25 test-fold subject metrics, not a pooled OOF score.

## Training source and bindings

To perform a separate new reproduction, establish the pinned CUDA runtime in
`ENVIRONMENT.md`, then use the explicit argument interface of
`experiments/S/scripts/run_fold.py` or `experiments/W/scripts/run_fold.py`, followed
by the matching `validate_fold.py`. They refuse existing output directories.
No such training was started when preparing this candidate. The packaging-only
new FROZEN manifests pass file hash checks; no new native CUDA preflight or
cross-environment prediction replay is claimed.

The stored processed input is distributed only after the official ds002778
version metadata's CC0 declaration was checked. This does not restore missing
historical preprocessing provenance. Preserve appropriate dataset/study citations.
Do not infer a blanket project-code license or clinical validity from CC0 data.
