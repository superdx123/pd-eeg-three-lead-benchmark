# Experiments and evaluation

`configs/paper.json` binds the study inputs and exact S/W manifests by
SHA-256. All paths are relative to the selected project root. GPU
selection is explicit via `--device`. A completed run directory is never
overwritten.

| Experiment | Design |
| --- | --- |
| `benchmark-s` | Nested LOSO, both cohorts, 11 primary configurations |
| `benchmark-w` | 5 repeats of 5 within-recording window folds, both cohorts |
| `transfer` | BE-MSNet, ATCNet and fixed spectral LR in both directions |
| `ablation` | Three matched seeds; full, raw-only, single-scale, mean-only |
| `strict-tdc` | Three matched seeds; representation learning excludes inner validation |
| `tdc-reference` | Representation learning includes inner validation; separately reported |
| `harmonized-lr` | Within-cohort S and bidirectional transfer with common spectral features |
| `time-blocked` | Contiguous test blocks with 2 s purging; BE-MSNet and harmonized LR |
| `representation-control` | Fixed five subject folds, 50 epochs, full versus R18 |

Select any experiment with `--experiment`, list its jobs, and add `--run`
to execute. Use the cohort, model, seed and fold filters to run jobs in
batches. `--smoke` saves reduced-epoch runs separately; they are not accepted
for benchmark scoring.
Frozen transfer selects on source data only; scoring attaches target
labels after predictions have been saved.

```sh
python scripts/validate_models.py --runs outputs/benchmark-s/ds002778/S/BE-MSNet --device cuda
python scripts/collect_predictions.py --runs outputs/benchmark-s --experiment benchmark-s --output outputs/s-predictions.csv
python scripts/analyze.py --predictions outputs/s-predictions.csv --output outputs/s-statistics
```

The collector checks saved file hashes, exact job coverage and the held-out
window indices from the study splits. For a single model or cohort, point
`--runs` at that subset and specify the matching `--model` and/or `--cohort`.
`--allow-partial` is for exploratory analysis only. Directly analyzing an
arbitrary CSV does not check it against the study plan.

Verify checkpoints from your own runs using the same fitting device and
software versions. The validator checks maximum absolute prediction error
against 1e-7. The example verifies PyTorch state-only checkpoints. To verify
locally created joblib models, also pass `--trust-sklearn`. Do not load
checkpoints from an untrusted source.

The default bootstrap seed is 20260930 for primary comparisons and the
matched representation control, and 20261018 for harmonized LR and time
blocks. Subjects are sampled
within diagnoses, with the same resample used for paired predictions and
across W folds. Models are matched by subject and seed ensembles average
probabilities before calculating AUC or bAcc. The fixed threshold is 0.5.
The analysis also reports fixed-bin calibration for S and transfer. With
both cohorts' complete five-seed BE-MSNet and FBCNet S predictions it
computes the paired difference-of-differences using 10000 resamples and
independent cohort streams from seed 20261023.

For the demographic-only LOSO sensitivity, supply the public dataset
metadata with `participant_id`, `age` and `sex` or `gender`:

```sh
python scripts/demographics.py --manifest data/splits/ds002778/s_outer.csv --metadata data/raw/ds002778/participants.tsv --sex-column gender --output outputs/demographics-778.json
```

## Matched TDCNet comparison

Use saved run directories for the same cohort and protocol:

```sh
python scripts/compare_tdc.py --cohort ds002778 --protocol S --tdc-runs outputs/strict-tdc/ds002778/S/TDCNet --be-msnet-runs outputs/benchmark-s/ds002778/S/BE-MSNet --output outputs/matched-tdc-778-s.json
```

Add `--reference-runs outputs/tdc-reference/ds002778/S/TDCNet-reference` to
include TDCNet†. Each arm must have every test job for seeds 11, 22 and 33.
Seeds 44 and 55 in the benchmark directories are not used. The command
checks that all three seeds use the same held-out windows across models.
It averages probabilities across seeds at each window before averaging
within subjects. It reports strict TDCNet minus each comparator, using
2000 paired, class-stratified subject resamples (seed 20260930) and fixed
threshold 0.5.
W uses the same resampled subjects in all 25 test folds.

The report describes the supplied runs. The ds002778 BE-MSNet comparator in
the manuscript supplement used the initial saved predictions, not the new
main benchmark run; matching seeds alone does not recover those predictions.

The figure command uses the included reference estimates rather than scores
from new runs.
