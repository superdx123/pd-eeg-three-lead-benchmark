# Three-lead Parkinson's disease EEG benchmark

Compare Parkinson's disease classifiers on the same C3, Cz and C4 inputs
under subject-independent evaluation, within-recording window evaluation,
and frozen cross-cohort transfer. The project includes BE-MSNet, the
comparison models, fixed experiment recipes, score aggregation, and
scripts for rebuilding the manuscript figures.

## Install

Use Python 3.10–3.12 in a virtual environment. For a CPU installation:

```sh
python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m unittest discover -s tests -v
```

For the original CUDA setup, use the PyTorch `cu121` wheel index instead.
The original training used PyTorch 2.5.1 with CUDA 12.1. CPU validation is
useful for checking the software; it does not establish numerical identity
with a different device or library build.

## Get the benchmark inputs

The two source datasets are [ds002778](https://doi.org/10.18112/openneuro.ds002778.v1.0.5)
and [ds003490](https://doi.org/10.18112/openneuro.ds003490.v1.1.0).
For exact prepared inputs and split files, use the versioned
`benchmark-inputs-v1.0.0.zip` when available on the
[release page](https://github.com/superdx123/pd-eeg-three-lead-benchmark/releases).
Verify and install a copy from the repository root:

```sh
python scripts/install_inputs.py /path/to/benchmark-inputs-v1.0.0.zip
```

The archive is distributed separately from Git. If it is not available on the
release page, the source checkout alone cannot reproduce the reported values.
[Data preparation](docs/data.md) explains the input format and the pure
eyes-open ds003490 preprocessing. Exact ds002778 reproduction starts from the
retained preprocessed windows because the original raw-to-window execution
cannot be reconstructed.

## Run the benchmark

The following lists a single subject-independent job. Add `--run` to execute
it and save a checkpoint, selection records and predictions.

```sh
python scripts/reproduce.py --experiment benchmark-s --cohort ds002778 --model BE-MSNet --seed 11 --fold 1
```

See [Experiments](docs/experiments.md) for the remaining runs, independent
checkpoint inference and score computation. To render figures directly from
the included reference estimates without data or training:

```sh
python scripts/plot_figures.py --output outputs/figures --formats pdf svg
```

## Reference scores

BE-MSNet subject-level AUC and balanced accuracy:

| Dataset | S AUC | S bAcc | W AUC | W bAcc |
| --- | ---: | ---: | ---: | ---: |
| ds002778 | 0.8458 | 0.7417 | 0.9913 | 0.9378 |
| ds003490 (eyes open) | 0.7248 | 0.7000 | 0.9265 | 0.8488 |

S holds out subjects. W holds out windows within the same recordings and
averages subject-level scores by test fold. These are distinct estimands.
The estimates above are reference results from the study, not from the
synthetic software tests.

BE-MSNet expands three recorded leads to 18 input waveforms before the
multi-scale network. In the original experiment files it was named
`DS28_BR_MS_AUG_LS_WD3`. The four-layer CNN was named `PrimaryCNN`.

## Citation and license

Cite the two OpenNeuro dataset records and the accompanying benchmark
manuscript. Pin the code commit used for your run. Project code is released
under [MIT](LICENSE); upstream software and data retain their own terms.
See [NOTICE](NOTICE.md) for model sources.
