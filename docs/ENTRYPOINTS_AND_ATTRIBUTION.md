# Entry points and attribution

## Safe maintenance entry points

`tools/prepare_source_snapshot.py --verify` and `tests/` use file hashing, JSON
validation and static Python parsing. They never import the frozen network/trainer
modules, read clinical records, run a forward pass or start training.

The following archived scripts have different purposes and are **not** generic
read-only quick-start checks:

- `frozen_ds/run/formal.py` has historical default training configurations. Direct
  execution may start S/W experiments; do not use it as an installation check.
- `frozen_ds/run/common.py` retains historical output helpers which can reuse
  directories or replace JSON. Use the explicit new runners for a separate run.
- `scripts/preflight.py` creates audit files and temporary synthetic checkpoints.
  It is an engineering preflight, not a read-only inventory command.
- `verify_scores.py` and `predict_saved_weights.py` retain assertion-based guards.
  Run without `-O`/`PYTHONOPTIMIZE`; preparation does not execute them automatically.

Some old analysis/helper default paths are retained to preserve the accepted source
hashes. Use explicit inputs and the complete archive layout, not those defaults.

## Method references retained in the source

The frozen code includes references to EEGNet (Lawhern et al., 2018), SimCLR-style
and InfoNCE objectives, focal loss (Lin et al., 2017), and Pi-model-style consistency.
These method references remain in the original files. Inclusion of an implementation
or reference does not relicense another project's code or dependencies.

The static license-marker review found no explicit restrictive third-party source
notice in the selected files. This is not proof that every line is original. Any
applicable upstream notice must be preserved; report a concrete provenance or
attribution omission before public release. See the root `NOTICE.md`.
