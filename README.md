# Three-lead Parkinson's disease EEG benchmark

**MIT project code. Private review staging; not publicly released.**

Research code for the completed DS28 experiments on OpenNeuro **ds002778**, covering unseen-subject evaluation (**S**) and within-record window evaluation (**W**). This first source package contains the fixed implementation, configurations, verification tools and accepted aggregate results. The accompanying paper also studies ds003490 and additional baselines; those complete pipelines are outside this package.

The repository will remain private until the owner explicitly requests publication. Project-authored code and documentation use the [MIT License](LICENSE); third-party materials retain their own terms. See [NOTICE.md](NOTICE.md).

## Contents and boundaries

| Location | Purpose |
| --- | --- |
| `source_snapshot/experiments/S/` | Fixed five-seed nested LOSO configuration and training/validation entry points |
| `source_snapshot/experiments/W/` | Five-seed, five-repeat, five-fold within-record configuration and entry points |
| `source_snapshot/experiments/*/frozen_ds/dsmodel/` | Network definitions and deterministic input transforms |
| `source_snapshot/experiments/*/frozen_ds/dstraining/` | Optimization, inner selection and final training |
| `source_snapshot/verify_scores.py` | Recompute aggregate metrics from the archive's saved predictions |
| `source_snapshot/predict_saved_weights.py` | Reduced-checkpoint adapter for the complete archive |
| `tools/prepare_source_snapshot.py` | Static source and archive integrity checks |
| `tests/` | Standard-library integrity and packaging tests |
| `archive.lock.json` | Exact filename, size and SHA-256 of the separate full archive |

`source_snapshot/` preserves 52 selected files byte-for-byte. Its mirrored documents describe the **complete archive** and retain dated preparation notes, including pre-approval license status. The root `LICENSE`, `NOTICE.md` and `PUBLICATION_STATUS.json` describe current licensing and visibility. Historical v2 references are superseded by the preserved `ADAPTER_V3_CHANGELOG.md`; the archive pinned here is v3.

There are no weights, EEG inputs, split tables, per-person predictions, individual CSV files, internal ID mappings, machine-origin manifests or operational logs in this repository. Legacy noncredential path defaults inside unchanged archived source have not been rewritten; the source snapshot is not a turnkey portable installation.

The archived S result is nested subject-level LOSO. W is the mean of 25 within-record test-fold subject metrics, not a pooled out-of-fold estimate or an independent-subject generalization claim. Accepted aggregate AUC / balanced accuracy: S 0.8458 / 0.7417; W 0.9913 / 0.9378. These summaries are copied, not newly recomputed by repository preparation.

## Use the fixed full archive for execution

Python 3.10+ and the standard library are sufficient for source-only checks; no PyTorch, GPU or EEG is needed:

```sh
python -B tools/prepare_source_snapshot.py --verify
python -B -m unittest discover -s tests -v
```

`archive.lock.json` pins `ds28_778_LOCAL_REVIEW_CANDIDATE_v3.zip` by SHA256 and byte size. The archive is a separate local artifact; no download URL or DOI is currently assigned. Do not infer that a public data/weight release has been authorized. The source subset cannot satisfy the full `FROZEN.json` input bindings and must not be advertised as doing so.

Obtain the exact archive through an authorized channel, then verify it without extracting or executing its code:

```sh
python -B tools/prepare_source_snapshot.py --verify --archive "/path/to/ds28_778_LOCAL_REVIEW_CANDIDATE_v3.zip"
```

The command hashes bytes, checks the repository whitelist and manifest, and statically parses Python. It does not load EEG/model objects, train, infer, choose thresholds or recompute metrics. Without `--archive`, `--verify` checks only this source repository.

For training, saved-score analysis or inference, work from the complete verified archive after extraction and preserve its layout and immutable bindings. Follow the [reproduction guide](docs/REPRODUCIBILITY.md) and [entry-point notes](docs/ENTRYPOINTS_AND_ATTRIBUTION.md). Run archived commands without `-O` or `PYTHONOPTIMIZE`, because some checks use assertions. The maintenance helper never launches scientific entry points.

## Validation scope

`repo_files_manifest.json` inventories repository files and their hashes. `local_ready_receipt.json` records local static preparation only; neither file is a scientific replay receipt. The original candidate's reduced checkpoints passed CPU state/architecture structure checks. Real EEG forward replay of the reduced files was **not** tested, and saved-prediction score verification is a different check. Repository preparation adds no scientific experiment or performance claim.

Official ds002778 v1.0.5 metadata declares CC0 for that dataset. The original dataset and study publications remain the appropriate data citations. Data, weights and individual tables are handled separately from the code license; they are not in Git. The stored processed input does not recover incomplete historical raw-preprocessing provenance.

See [CONTRIBUTING.md](CONTRIBUTING.md) for maintenance rules and the [release checklist](docs/RELEASE_CHECKLIST.md) for publication steps. MIT approval does not change repository visibility; no public URL or DOI for the archive is asserted.
