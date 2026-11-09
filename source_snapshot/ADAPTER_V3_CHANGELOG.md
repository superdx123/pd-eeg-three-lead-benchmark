# Candidate v3: adapter robustness corrections only

The current complete candidate uses `release_manifest_v3.json`; v1 and v2
manifests and their earlier ZIPs are preserved for provenance. No model, training
protocol, labels, probabilities, input data or numerical tolerance was changed.

The inference adapter now hashes its saved reference table against the manifest,
checks the W repeat identity, rejects conflicting CUBLAS configuration, and refuses
an existing output task before CUDA inference. Subject classification mismatches
are counted and saved in a failing receipt rather than asserting before saving
the failed predictions. Probability tolerance remains 1e-7 and threshold 0.5.

Only CPU model-structure/state compatibility was exercised when building this
candidate. Real EEG forward inference of the metadata-reduced files was not run.
Saved-prediction score verification is not equivalent to that inference replay.

`INFERENCE_ADAPTER.md` documents the unchanged interface and comparison scope;
its statement that v2 is current describes the preserved v2 snapshot. This v3
addendum and v3 manifest identify the current local review candidate.
