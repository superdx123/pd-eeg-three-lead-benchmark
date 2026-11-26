# Data preparation

The benchmark uses OpenNeuro [ds002778](https://openneuro.org/datasets/ds002778) and [ds003490 version 1.1.0](https://openneuro.org/datasets/ds003490/versions/1.1.0). Download the original files from their repositories and retain their dataset citations. The cited public metadata declare CC0. Use only the two named cohorts with these preparation commands.

## Prepared input format

Each NPZ contains `X` (float32 windows × channels × 512), `y` (int64; HC=0, PD=1), `groups` (subject IDs), `recording_ids`, `channel_names`, `sfreq` (256 Hz), `window_sec` (2 s), and `step_sec` (2 s). Arrays use no Python object serialization. Each subject has one diagnosis label and one selected recording. Classifier channels are ordered C3, Cz, C4.

Use `scripts/install_inputs.py` to install the exact inputs required by
`configs/paper.json`. The commands below write separately to `data/rebuilt/`
for preprocessing checks and do not replace the registered inputs.

The ds002778 study input contains 3022 already standardized windows from 31 subjects (16 HC, 15 PD), stored in the order FC1, FC2, C3, Cz, C4, CP1, CP2. Import selects the three classifier channels without filtering, resampling, QC or another z-score:

```sh
python scripts/prepare_data.py import --cohort ds002778 \
  --source data/input/ds002778_preprocessed_7ch_2s.npz \
  --output data/rebuilt/ds002778.npz
```

The command checks the stored study NPZ SHA-256 `f00315271235e23cff51c28604c20e2ec151ca30508953cef44cc4dc1deaf06b`. The selected three-channel output is a new NPZ with a different file hash. The complete raw-to-window execution history for ds002778 cannot be reconstructed from the retained records; obtaining the raw OpenNeuro dataset alone does not recreate this exact input. This prepared file is therefore required to reproduce the reported ds002778 analyses.

The pure-EO ds003490 study input has 1348 windows from 50 subjects (25 HC, 25 PD-OFF). Its stored input SHA-256 is `87e44b301c4651335d6576561108861c2b11611ef288d0cbd629465c652c1d8a`. An existing prepared copy can be imported with the same `import` command and `--cohort ds003490`.

## Prepare eyes-open ds003490 from BIDS

```sh
python scripts/prepare_data.py eo490 \
  --bids-root data/raw/ds003490 \
  --output data/rebuilt/ds003490.npz
```

`participants.tsv` determines diagnosis and medication state. PD contributes its unique OFF visit; HC contributes its unique available Rest visit. Ambiguous metadata stop processing. A CSV supplied with `--selection-file` must contain `participant_id,session,label` and agree with the BIDS group and medication fields; no automatic choice between ambiguous visits is made.

Preparation requires one continuous sequence of at least 55 `Eyes Open: Every 1000 ms` markers, with adjacent intervals strictly between 0.90 and 1.10 s and no closed-eye or auditory events inside that interval. The first and last EO markers define a single conservative crop; no extra second is appended. Crop limits are rounded inward on the original 500 Hz grid. Only this cropped signal is filtered at 1–40 Hz and resampled to 256 Hz using MNE. Average reference is computed over retained scalp channels before selecting C3, Cz and C4.

Each end loses 2 s after filtering and resampling. Remaining samples form non-overlapping 2 s windows, with incomplete final windows discarded. QC keeps windows with absolute amplitude ≤150 µV and mean channel population SD between 0.5 and 80 µV. Retained channels are standardized within each window using population SD plus 1e-8 µV. The study cohort must retain 26–27 windows per subject. Preprocessing writes relative raw-file SHA-256 entries beside the prepared file. For the original numerical pipeline use MNE 1.12.1 and SciPy 1.15.3.

## Subject and window splits

```sh
python scripts/prepare_data.py splits \
  --cohort ds002778 --dataset data/rebuilt/ds002778.npz --output data/rebuilt/splits/ds002778
python scripts/prepare_data.py splits \
  --cohort ds003490 --dataset data/rebuilt/ds003490.npz --output data/rebuilt/splits/ds003490
```

S uses sorted subject IDs for outer LOSO and stratified five-fold inner validation with seed `730000 + outer_fold` for ds002778 and `830000 + outer_fold` for ds003490. W uses split seeds 11001, 22002, 33003, 44004, 55005. Within each repeat, every subject's window indices are shuffled using `SeedSequence([split_seed, subject_position])` and divided into five balanced test folds. For test fold f, the next cyclic fold is validation and the remaining three are training. Model refitting can then combine the four non-test folds after selection.

The output files are `s_outer.csv`, `s_inner.csv`, and `w_windows.csv`. Split validation checks window coverage, labels and subject membership. S forbids subject overlap between training, validation and testing; W intentionally contains the same subjects across roles while keeping window indices disjoint. Training seeds are separate from the split seeds. If using distributed study manifests, validate and reuse them rather than regenerating them under a different subject order or software version.

Commands write only below the current working directory and refuse to overwrite prepared inputs or manifests. Inputs may be supplied from another location. No command downloads data or starts model training.

## Temporal blocks and representation controls

Temporal evaluation requires actual retained-window coordinates, not row rank. Supply a CSV with `window_index,subject_id,recording_id,start_sec,end_sec`, in dataset row order. Times are seconds within the selected recording; each interval must be 2 s. `build_time_blocks` forms five balanced chronological blocks per subject. `blocked_roles` excludes candidate windows unless the actual interval gap to the relevant test/validation windows is at least 2 s. Final refitting uses the non-test blocks purged against the test block. The ds002778 coordinates come from the original retained-window QC table, whose accepted SHA-256 is `1af1b16e3a0b34860acd8176b82b4dce804f45877c1a4d746425303a40e09b1e`; they cannot be inferred from a list of windows remaining after QC. The EO ds003490 NPZ contains `window_start_seconds`.

The full/R18 control uses five fixed subject folds, separate from nested LOSO. Within each class, original participant strings are ordered by `SHA256("frequency_transport_v1|20261021|label|participant_id")`, then assigned round-robin to folds 0–4. `representation_folds` implements this rule. The study's original ds003490 NPZ stores those same participant strings in `groups`; if a local copy renames groups, provide an explicit one-to-one `participant_ids_map` to recover the original hash inputs.
