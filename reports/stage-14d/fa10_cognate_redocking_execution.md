# Stage 14d FA10 Cognate Redocking Execution

## Scope

This bundle runs 16 audited FA10 receptor-cognate-ligand cases with three frozen
Uni-Dock 1.1.3 seeds. It contains no FA10 activity labels, benchmark docking
scores, fresh-validation rows, test rows, or earlier target outcome matrices.

## Frozen Gate

- Engine: Uni-Dock 1.1.3, enhanced profile.
- Search: exhaustiveness 1024, max step 80, refine step 5, one output pose.
- Seeds: 20260801, 20260802, and 20260803.
- Common box: center (1.34, -8.50, -15.76), size (22, 22, 28) A.
- Per-receptor pass: at least two of three top-ranked poses have symmetry-corrected
  heavy-atom RMSD no greater than 2.0 A and median RMSD no greater than 2.0 A.
- Cohort pass: all 16 receptors pass with zero unresolved warning events and zero
  pose-integrity failures.

The gate and search settings are transferred unchanged from the multitarget master
preregistration. The EGFR technical failure was not used to replace a receptor,
relax the RMSD threshold, alter a seed, or tune Uni-Dock for FA10.

## Remote Run

Reuse the existing `qubo-unidock-stage13` environment, then run:

```bash
bash scripts/experimental/unidock/run_stage14d_fa10_cognate_redocking_remote.sh
```

Set `AUTO_POWEROFF=1` when automatic instance shutdown after archive creation is
desired. A scientifically failed RMSD gate is still packaged as a completed result;
runtime or integrity errors create a separate failure archive.

## Interpretation

A pass authorizes preparation and docking of the untouched FA10 Train-696 panel.
A fail is preserved without receptor replacement or protocol retuning, and the
preregistered workflow continues to HIVPR. Neither outcome alone establishes
screening enrichment, QUBO superiority, quantum execution, or quantum advantage.
