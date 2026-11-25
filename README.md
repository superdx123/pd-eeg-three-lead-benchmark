# Three-lead Parkinson's disease EEG benchmark

Research code for **BE-MSNet (Band-Expanded Multi-Scale Network)** on OpenNeuro **ds002778**, with subject-independent evaluation (**S**) and within-record window evaluation (**W**).

Project-authored code and documentation are licensed under [MIT](LICENSE). The repository is currently private, pending the owner's visibility change. This code release covers the ds002778 BE-MSNet S/W experiments; it is not the complete software release for every analysis in the accompanying manuscript.

## Model names

The manuscript calls the network **BE-MSNet**. Frozen code and configuration files retain the identifier `DS28` and model key `DS28_BR_MS_AUG_LS_WD3`. These names refer to the same configuration, not different models. Keeping the original identifiers preserves compatibility with the recorded configurations, predictions and hashes. Similarly, `PrimaryCNN` is the frozen identifier for the manuscript's **4L-CNN** baseline; this package does not contain that baseline's complete experimental pipeline.

## Included material

| Location | Contents |
| --- | --- |
| `source_snapshot/experiments/S/` | Five-seed nested LOSO configurations and training/validation entry points |
| `source_snapshot/experiments/W/` | Five-seed, five-repeat, five-fold within-record configurations and entry points |
| `source_snapshot/experiments/*/frozen_ds/dsmodel/` | Network definitions and input transforms |
| `source_snapshot/experiments/*/frozen_ds/dstraining/` | Optimization, inner selection and final training |
| `source_snapshot/verify_scores.py` | Metric verification from saved predictions in the separate archive |
| `source_snapshot/predict_saved_weights.py` | Reduced-checkpoint inference adapter for the separate archive |
| `tools/prepare_source_snapshot.py` | Static source and archive integrity checks |
| `tests/` | Standard-library integrity and packaging tests |
| `archive.lock.json` | Filename, size and SHA-256 of the separately stored archive |

The 52 files in `source_snapshot/` are preserved byte-for-byte. Complete ds003490 pipelines, other baseline runs, manuscript figure-generation pipelines, weights, EEG inputs, split tables and participant-level records are outside this release. Dataset and dependency terms are described in [NOTICE.md](NOTICE.md).

## Quick checks

Python 3.10 or later is sufficient for the source checks. They do not require PyTorch, a GPU or EEG data:

```sh
python -B tools/prepare_source_snapshot.py --verify
python -B -m unittest discover -s tests -v
```

These commands validate file hashes, parse Python and JSON, and test packaging safeguards. They do not train models, generate predictions or recalculate study metrics.

## Running the archived experiments

The source snapshot preserves the original layout and some local path defaults. Execution requires the matching inputs and bindings in the separate archive, rather than a standalone installation from this repository. Read [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) and [ENTRYPOINTS_AND_ATTRIBUTION.md](docs/ENTRYPOINTS_AND_ATTRIBUTION.md) before running an experimental entry point. Run without `-O` or `PYTHONOPTIMIZE`, because archived checks use assertions.

The archive is pinned by `archive.lock.json`; no public archive download or DOI has been assigned. To verify an authorized local copy without extracting or executing it:

```sh
python -B tools/prepare_source_snapshot.py --verify --archive "/path/to/ds28_778_LOCAL_REVIEW_CANDIDATE_v3.zip"
```

The reduced checkpoints in that archive passed CPU structure checks. Their real-EEG forward replay remains unverified; validation of the original saved training checkpoints is a separate result. The archive is therefore not included in this code-only publication.

## Reported aggregate results

| Cohort | Protocol | AUC | Balanced accuracy |
| --- | --- | ---: | ---: |
| ds002778 | S: nested subject-level LOSO | 0.8458 | 0.7417 |
| ds002778 | W: mean of 25 within-record test-fold subject metrics | 0.9913 | 0.9378 |

Both results use five-seed probability ensembles. The W summary is not a pooled out-of-fold estimate. These numbers are the recorded aggregate results; preparing this repository does not introduce a new experimental run.

## Integrity and licensing

`repo_files_manifest.json` inventories the released files and their SHA-256 values. `local_ready_receipt.json` records local static checks. Root licensing and scope statements apply to this release; documents mirrored under `source_snapshot/` retain the original archive's preparation status and terminology. Its adapter v2 notes are superseded by the preserved `ADAPTER_V3_CHANGELOG.md`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for maintenance rules and [RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md) for publication steps.
