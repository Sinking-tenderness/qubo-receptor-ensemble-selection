# Stage 19c PPARG Train-668 Uni-Dock Execution

This bundle contains only the frozen PPARG development-training panel. Fresh
validation and locked test ligands are not included.

## Frozen Workload

- Receptors: 16 exploratory PPARG receptors
- Ligands: 668 (`334 active + 334 decoy`)
- Seeds: 3
- GPU batches: 48
- Receptor-ligand-seed pairs: 32,064
- Uni-Dock: 1.1.3, enhanced profile, exhaustiveness 1024, max step 80

## Run

Reuse the existing environment:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qubo-unidock-stage13
```

The remote wrapper performs an audit-only preflight, resumes valid batches,
independently audits the complete matrix, and writes core and diagnostics
archives:

```bash
nohup env AUTO_POWEROFF=1 \
  bash scripts/experimental/unidock/run_stage19c_pparg_train668_production_remote.sh \
  > stage19c_pparg_train668.log 2>&1 &

echo $! | tee stage19c.pid
tail -f stage19c_pparg_train668.log
```

`AUTO_POWEROFF=1` requests shutdown only after synchronization and archive
creation. Provider-side billing shutdown behavior must still be checked.

## Outputs

```text
/root/autodl-tmp/stage19c_pparg_train668_unidock113_production_core_v1.tar.gz
/root/autodl-tmp/stage19c_pparg_train668_unidock113_production_diagnostics_v1.tar.gz
```

The core archive is sufficient for matrix and train-only method analysis. Keep
the diagnostics archive for pose-level adjudication.

Stage 18e remains a failed confirmatory technical gate. Stage 19c is post-hoc
exploratory training evidence and cannot establish validation performance,
QUBO superiority, or quantum advantage.
