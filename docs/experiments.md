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
to execute. `--cohort`, `--seed`, `--fold`, `--repeat` and `--variant`
can select subsets. `--smoke` uses a reduced epoch budget and separate
output directory; do not combine these results with the full runs.
Frozen transfer selects on source data only; scoring attaches target
labels after predictions have been saved.

```sh
python scripts/validate_models.py --runs outputs/benchmark-s --device cuda
python scripts/collect_predictions.py --runs outputs/benchmark-s --output outputs/s-predictions.csv
python scripts/analyze.py --predictions outputs/s-predictions.csv --output outputs/s-statistics
```

Use the same fitting device and software versions for checkpoint replay.
The validator checks a maximum absolute error of 1e-7. Add
`--trust-sklearn` only for local model files you created or independently
trust; scikit-learn joblib files are pickle-based. PyTorch state-only
checkpoints are loaded with `weights_only=True`.

For manuscript intervals, use the appropriate bootstrap seed: 20260930 for
primary comparisons and the matched representation control, and
20261018 for harmonized LR and time blocks. Subjects are sampled
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

Training, saved-model inference, and plotting are separate operations.
The figures use reference aggregate estimates; plots of a new run are
not silently substituted for the paper values.
