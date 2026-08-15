# Stage 13f EGFR Cognate Redocking Execution

## Scope

This bundle runs 16 audited EGFR receptor-cognate-ligand cases with three frozen
Uni-Dock 1.1.3 seeds. It contains no EGFR activity labels, benchmark docking
scores, fresh-validation rows, test rows, or MAPK14 outcome files.

## Frozen Gate

- Engine: Uni-Dock 1.1.3, enhanced profile.
- Search: exhaustiveness 1024, max step 80, refine step 5, one output pose.
- Seeds: 20260801, 20260802, and 20260803.
- Common box: center (18.56, 33.55, 88.93), size (22, 22, 30) A.
- Per-receptor pass: at least two of three top-ranked poses have symmetry-corrected
  heavy-atom RMSD no greater than 2.0 A and median RMSD no greater than 2.0 A.
- Cohort pass: all 16 receptors pass with zero unresolved warning events and zero
  pose-integrity failures.

Three receptors have covalent or irreversible wording in their source citation,
but no explicit protein-ligand covalent connection in the deposited coordinates.
They remain subject to the same noncovalent redocking gate and receive no exception.

## Remote Run

Create or activate the environment in `environment/stage13_unidock_gpu.yml`, then
run:

```bash
bash scripts/experimental/unidock/run_stage13f_egfr_cognate_redocking_remote.sh
```

Set `AUTO_POWEROFF=1` only when automatic instance shutdown after archive creation
is desired. A scientifically failed RMSD gate is still packaged as a completed
result; runtime or integrity errors create a separate failure archive.

## Interpretation

A pass authorizes the untouched EGFR Train-696 production screen. It does not
establish enrichment, QUBO portability, QUBO superiority, statistical significance,
quantum execution, or quantum advantage.
