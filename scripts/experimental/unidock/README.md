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

Stage 07 treats Uni-Dock 1.1.3 as a separate engine. The Train-160 search
sensitivity compares the official fast, balance, and detail profiles using
the macrocycle-safe manifest, train-only enrichment, seed stability, and
throughput. It does not repeat the obsolete CPU-score equivalence gate.

Stage 07b follows the failed Stage 07 profile gate with a frozen two-factor
confirmation. It rechecks 512/40 and tests 512/80, 1024/40, and 1024/80 on the
same consumed Train-160 rows and paired seeds. Every output pose is audited for
atom-count and atom-type preservation. Only a candidate that passes the rank,
early-enrichment, seed-stability, engine-warning, and pose-integrity checks can
be frozen for a larger Uni-Dock training matrix.

Stage 07c is the preregistered technical adjudication after enhanced 1024/80
passed every statistical Stage 07b check but emitted one known coordinate
warning event. It adds one new seed and exactly replays the affected batch.
Statistical thresholds are unchanged. The warning is resolved only if every
pose remains valid and all 160 replay scores and pose hashes match exactly.

Stage 08 starts only after Stage 07c freezes enhanced 1024/80. It extends the
label-independent MAPK14 max-min structural pool from 8 to 16 receptors, then
prepares and three-seed cognate-redocks the eight new additions. The large
Train-696 matrix remains blocked until all new receptors and an independent
RMSD audit pass.

Stage 08b is the preregistered technical replacement cycle after two Stage 08
receptors failed the unchanged RMSD gate. It permanently excludes 3ZSH and
2GFS, refills the admitted 14-receptor pool by deterministic max-min selection,
and tests only the replacements 3ITZ and 2BAK. It never relaxes the gate or
reuses docking affinity for structural ranking.

Stage 08c replaces the final failed redocking receptor and freezes the admitted
16-receptor pool. Stage 09 builds the three-seed Train-696 Uni-Dock matrix,
Stage 10 screens the preregistered exact-QUBO and greedy candidates on training
evidence, and Stage 11 evaluates only those frozen candidates on the untouched
1,576-ligand validation panel.

Stage 11 Amendment 01 changes only the finite-score parser guard from 100 to
1000 kcal/mol after Uni-Dock returned a complete, finite high-positive-energy
pose. The final archives are independently adjudicated with:

```text
python scripts/experimental/unidock/adjudicate_stage11_mk14_fresh_validation_amendment01.py \
  --core-root <extracted-core> \
  --diagnostics-root <extracted-diagnostics> \
  --core-archive <core.tar.gz> \
  --diagnostics-archive <diagnostics.tar.gz>
```

The adjudication verifies every pose and batch, independently rebuilds the
score matrices and metrics, and tests the high-positive-score sensitivity. A
positive QUBO point estimate is an application result only; the frozen gate
also requires both paired-bootstrap 95% lower bounds to exceed zero. Even a
gate pass would not by itself establish quantum computational advantage.
