# Stage 51 PPARA Large-Pool Cognate Redocking

Stage51 runs 180 GPU batches: 60 technically prepared PPARA receptors, one
cognate ligand per receptor, and three frozen random seeds. Uni-Dock 1.1.3 uses
the unchanged enhanced profile with exhaustiveness 1024, max step 80, refine
step 5, and one output mode.

The common box was derived from all 60 prepared cognate ligands in the 2P54
frame. Four Stage50 preparation failures remain failures in the frozen
64-receptor denominator and are not replaced.

Each prepared receptor passes when at least two of three top-ranked poses have
symmetry-corrected fixed-frame heavy-atom RMSD at or below 2.0 A and the median
RMSD is at or below 2.0 A. At least 24 of the frozen 64 receptors must pass,
with zero unresolved warning events and zero pose-integrity failures.

Passing authorizes development-panel docking only. It does not establish
enrichment, QUBO superiority, quantum execution, speedup, or quantum advantage.

## Runtime Amendment 01

The first remote invocation stopped before batch 1 because the deterministic
input archive omitted `scripts/prepare_receptor.py`, a transitive import used by
the shared Stage14 runner for SHA-256 verification. No GPU docking result was
created or inspected. Amendment 01 adds the missing source file only; all
receptors, ligands, seeds, docking parameters, gates, and data boundaries remain
unchanged. The amended archive is tested after isolated extraction with an
audit-only invocation before release.

## Observed Result

All 180 planned GPU batches completed with zero unresolved warning events and
zero pose-integrity failures. The frozen gate nevertheless failed: 20 of 64
receptors passed, below the preregistered minimum of 24. Of the 60 technically
prepared receptors, 18 passed in all three seeds, two passed in two seeds,
eight passed in one seed, and 32 passed in no seed.

The result is strongly bimodal. Passing receptors have median top-ranked RMSD
between 0.25 and 1.26 A, while only two failed receptors are near the cutoff
(2.72 and 2.82 A); the next failed median is 5.05 A. The confirmatory PPARA
development-panel docking route is therefore not authorized, and the cutoff
must not be relaxed post hoc.

A label-free post-hoc diagnostic found pronounced cognate-ligand series
effects. For example, all three 8YT structures passed whereas all nine 6LX
structures failed. Passing cognate ligands were generally larger and less
flexible than failing ligands. This suggests that the redocking gate mixes
receptor suitability with chemotype-specific pose-recovery difficulty. These
associations are explanatory only and cannot overturn the frozen gate.
