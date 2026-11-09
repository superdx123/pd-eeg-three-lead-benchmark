# PD EEG three-lead benchmark — source snapshot

PUBLICATION_STATUS: **private review staging / license pending / not publicly released**

This repository is a private review staging copy of a small, static source/configuration and aggregate-result subset of the DS28 ds002778 S/W reproducibility candidate. It is not the complete experimental archive, not a clinical tool, and not a claim that every cohort, baseline, paper statistic or figure has been integrated. No project-wide MIT or other open-source license is granted here. Author ownership, licensing, attribution and final publication/privacy review remain pending; see `source_snapshot/LICENSE_STATUS.md`.

## Contents and boundaries

`source_snapshot/` preserves 52 selected files byte-for-byte from the v3 candidate payload: archived Python source, the original fixed protocol/task configurations, five archive-scope documents and `primary_metrics.json`. The mirrored documentation describes the **complete archive**, not the contents of this small repository. Its historical v2 references must be read together with `source_snapshot/ADAPTER_V3_CHANGELOG.md`; v3 is the archive pinned here.

There are no weights, EEG inputs, split tables, per-person predictions, individual CSV files, internal ID mappings, machine-origin manifests or operational logs in this repository. Legacy noncredential path defaults inside unchanged archived source have not been rewritten; the source snapshot is not a turnkey portable installation.

The archived S result is nested subject-level LOSO. W is the mean of 25 within-record test-fold subject metrics, not a pooled out-of-fold estimate or an independent-subject generalization claim. Accepted aggregate AUC / balanced accuracy: S 0.8458 / 0.7417; W 0.9913 / 0.9378. These summaries are copied, not newly recomputed by repository preparation.

## Use the fixed full archive for execution

`archive.lock.json` pins `ds28_778_LOCAL_REVIEW_CANDIDATE_v3.zip` by SHA256 and byte size. The archive is a separate local artifact; no download URL or DOI is currently assigned. Do not infer that a public data/weight release has been authorized. The source subset cannot satisfy the full `FROZEN.json` input bindings and must not be advertised as doing so.

Obtain the exact archive through an authorized channel, then verify it without extracting or executing its code:

```sh
python -B tools/prepare_source_snapshot.py --verify --archive "/path/to/ds28_778_LOCAL_REVIEW_CANDIDATE_v3.zip"
```

The command hashes bytes, checks the repository whitelist and manifest, and statically parses Python. It does not load EEG/model objects, train, infer, choose thresholds or recompute metrics. Without `--archive`, `--verify` checks only this source repository.

For a separately authorized reproduction, work from the complete verified archive after extraction, preserve its original layout and immutable bindings, and follow its `ENVIRONMENT.md` and explicit entry-point documentation. Do not add missing inputs to this Git repository, patch frozen paths, weaken environment/version gates, regenerate frozen data/protocol manifests or substitute the reduced weights for full validator checkpoints. The helper here never launches those entry points.

## Validation scope

`repo_files_manifest.json` inventories repository files and their hashes. `local_ready_receipt.json` records local static preparation only; neither file is a scientific replay receipt. The original candidate's reduced checkpoints passed CPU state/architecture structure checks. Real EEG forward replay of the reduced files was **not** tested, and saved-prediction score verification is a different check. Repository preparation adds no scientific experiment or performance claim.

Official ds002778 v1.0.5 metadata declares CC0 for the dataset, not for project source or weights. Dataset/study citations and code/weight permissions still require final author review. Pseudonymization is not irreversible anonymization. Third-party dependencies retain their own licenses. Public publication remains pending even if the source subset is later privately transferred for review.
