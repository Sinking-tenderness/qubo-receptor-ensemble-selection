# Uni-Dock Experimental Scripts

This directory contains historical, consumed-train Uni-Dock diagnostics. It
is not part of the current Stage102A canonical pipeline and is not a generic
replacement for CPU Vina.

The archived Uni-Dock experiments failed or remained restricted at their
engine-equivalence and validation gates. Their score matrices must not be
mixed with CPU Vina matrices, and these scripts must not be used to unlock a
validation or test split.

For current development work, start from an admitted score matrix and use:

```text
scripts/run_pipeline.py
configs/pipelines/stage102a_fa10_development_selection.json
configs/pipelines/stage102a_egfr_development_selection.json
```

Any new Uni-Dock experiment requires a new engine-specific protocol, input
hashes, train-only boundary, and independent equivalence evidence.
