# Script Guide

The repository preserves experiment-specific scripts because reports pin their
paths and SHA-256 values. They are research records, not 133 alternative ways
to run the current pipeline.

Start with the supported workflow catalog:

```bash
python scripts/workflow.py list
python scripts/workflow.py show dock-vina
python scripts/workflow.py run dock-vina -- --help
```

## Supported Pipeline

| Step | Preferred script |
|---|---|
| Audit SMILES | `check_ligand_smiles.py` |
| Generate ligand 3D SDF | `prepare_ligand_3d_sdf.py` |
| Prepare ligand PDBQT | `batch_prepare_ligand_pdbqt_parallel.py` |
| Align receptor | `align_receptor_structure.py` |
| Prepare receptor PDBQT | `prepare_receptor.py` |
| Validate redocking | `evaluate_redocking_rmsd.py` |
| Run official CPU Vina | `batch_vina_docking_parallel.py` |
| Build one-seed score matrix | `build_score_matrix.py` |
| Aggregate current seed schema | `aggregate_seed_replicates.py` |
| Calculate screening metrics | `evaluate_virtual_screening.py` |
| Fit development-only ensembles | `run_development_scaffold_cv_gate.py` |
| Fit a new EnOpt-style tree baseline | `fit_enopt_xgboost_baseline.py` |

`batch_vina_docking.py` remains in place because the parallel runner imports
its parsing and command helpers. New screening jobs should use the parallel
entry point. `aggregate_vina_seed_replicates.py` is the older CDK2 aggregation
schema; new Stage 5 work uses `aggregate_seed_replicates.py`.

## Current MAPK14 Mainline

The active confirmatory path is the frozen official AutoDock Vina 1.2.7 fresh
validation workflow:

1. `build_stage05_mk14_fresh_validation_remote_bundle.py`
2. `run_stage05_mk14_fresh_validation_remote.sh`
3. admission and hash audit
4. `evaluate_stage05_mk14_fresh_validation.py` exactly once

The evaluation script is intentionally restricted in `workflow.py`. The
locked test is not part of this sequence.

The supplementary EnOpt-style XGBoost models have already been fitted and
frozen on Train-696. Do not rerun or retune them. After the primary validation
result is finalized, `evaluate_enopt_xgboost_fresh_validation.py` may be run
once with its preregistered configuration. It cannot change the primary gate.

## Other Script Classes

- `build_stage*`, `run_stage*`, and `merge_stage*` files are frozen experiment
  orchestration. Reuse their general modules instead of copying a numbered
  experiment wrapper for a new target.
- `audit_*` files independently verify an existing result. They are not data
  preparation or docking entry points.
- `diagnose_*` files investigate a declared failure and must not overwrite the
  source score matrix.
- OpenMM scripts form a separate receptor-generation workflow; they are not
  required to rerun the current MAPK14 fresh validation.
- Unsupported GPU docking experiments live under `experimental/`.

## Adding New Work

Prefer a JSON configuration plus an existing general runner. Add a new Python
script only when the operation itself is new. New supported operations must be
added to `workflow.py`; one-off diagnostics belong under `experimental/` or in
a stage report with an explicit stopping rule.
