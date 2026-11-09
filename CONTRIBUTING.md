# Contributing

## Scope

Use issues and pull requests for reproducible software defects, documentation and
new analyses. Do not upload credentials, private clinical fields, EEG or checkpoint
files to issues, pull requests or Git history.

## Preserve scientific evidence

- Do not reformat, deduplicate or edit the accepted `source_snapshot/` files in place.
  A change to an analysis implementation belongs in a separately identified version.
- Preserve the original seeds, splits, threshold, metrics and archive hashes when
  checking an existing result. Corrected results must be distinguished from history.
- Do not overwrite checkpoints, predictions or unsuccessful verification receipts.
- Distinguish static checks, saved-score recomputation and real model prediction
  replay. Passing one is not evidence that the others were performed.

## New maintenance code

- Use Python 3.10+ and standard-library dependencies for integrity tooling.
- Use four spaces, UTF-8, clear function names, explicit CLI arguments and meaningful
  error messages. Keep new lines near 100 characters where practical.
- Resolve paths relative to the tool or an explicit input root, not a developer's
  computer. Validate paths before writing. Do not add network or training side effects.
- Add synthetic tests for both successful verification and rejected inputs.
- Run the two quick-start commands in the root README before submitting changes.

Integrity tests do not install scientific training dependencies or start model
experiments. Do not use `python -O` or `PYTHONOPTIMIZE` with archived entry points;
some of their validation guards use assertions.
