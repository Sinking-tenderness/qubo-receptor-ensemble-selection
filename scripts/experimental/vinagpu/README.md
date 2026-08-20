# Vina-GPU Experimental Scripts

This directory contains a historical consumed-train Vina-GPU pilot and its
bounded diagnostics. It is isolated from official CPU Vina evidence and is
not part of the current Stage102A canonical pipeline.

The pilot's deterministic bridge reproduced its recorded reference batch, but
that fact does not establish a production engine gate, a QUBO benefit, or a
quantum advantage. Do not mix GPU and CPU score matrices.

New development work should start from an admitted score matrix and use the
schema 2.0 configurations under `configs/pipelines/`. Any new GPU study needs
its own preregistration, complete equivalence evidence, and train-only data
boundary.
