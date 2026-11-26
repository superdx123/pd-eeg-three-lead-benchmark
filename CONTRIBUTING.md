# Contributing

Use a fresh clone and the CPU installation in [README.md](README.md).
Run `python -m unittest discover -s tests -v` before submitting a change.
Tests use synthetic inputs; no study data or GPU is required.

- For bug reports, include the commit, Python version, device, dependency
  versions, command, and a minimal example with synthetic input.
- Keep paths relative to the project root. Do not commit datasets, subject
  records, predictions, checkpoints, credentials, or generated outputs.
- Separate code fixes from changes to scientific recipes. Changes to a
  split, model, threshold, or estimator require a named configuration and
  must not silently overwrite the manuscript configuration or results.
