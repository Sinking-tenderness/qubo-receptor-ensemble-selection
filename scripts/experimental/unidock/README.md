# Uni-Dock Experiment

This directory isolates the Stage 5 Train-160 Uni-Dock diagnostics from the
supported CPU Vina workflow. It contains bundle builders, runners, audits, and
the rigid-macrocycle follow-up.

The enhanced profile was fast but failed one of seven frozen equivalence
checks. It is therefore not an interchangeable implementation of the official
AutoDock Vina 1.2.7 matrices. Any future Uni-Dock study must be treated as a
separate docking engine and must rebuild training evidence before validation.

Historical result bundles created before this directory cleanup retain their
original `scripts/*.py` paths and hashes. Do not rebuild those archives and
expect the historical SHA-256 values to remain unchanged.
