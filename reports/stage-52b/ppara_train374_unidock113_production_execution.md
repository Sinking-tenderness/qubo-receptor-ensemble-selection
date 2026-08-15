# Stage 52b PPARA Train-374 Uni-Dock production

## Scope

Stage 52b is a post-hoc exploratory development experiment. It docks the frozen
Train-374 panel against all 20 Stage51-passing PPARA receptors with three paired
seeds. Stage51 remains a failed confirmation (`20/64`, below the preregistered
minimum of `24/64`).

## Frozen grid

- Receptors: 20, preserving Stage51 gate order.
- Ligands: 374 Train rows, 187 active and 187 decoy.
- Seeds: `20260801`, `20260802`, and `20260803`.
- GPU batches: 60.
- Receptor-ligand-seed score rows: 22,440.
- Uni-Dock: version 1.1.3, enhanced profile, exhaustiveness 1024,
  max step 80, refine step 5, one pose.
- PPARA common box: center `(17.05, 40.18, 28.03)`, size `(28, 30, 24)`.

## Technical gates

The launcher first audits every frozen input and hash. A completed batch is
resumed only when its protocol signature, score file, pose hashes, pose-integrity
audit, and warning adjudication all match. Finalization requires all 60 batches,
zero unresolved warnings, and zero pose-integrity failures. An independent
auditor then rebuilds the full score grid and both seed-aggregation matrices.

## Remote execution

```bash
cd /root/autodl-tmp
mkdir -p stage52b_ppara_train374_unidock113_production_input_v1
tar -xzf stage52b_ppara_train374_unidock113_production_input_v1.tar.gz \
  -C stage52b_ppara_train374_unidock113_production_input_v1
cd stage52b_ppara_train374_unidock113_production_input_v1
sha256sum -c bundle_manifest.sha256

nohup env AUTO_POWEROFF=1 \
  bash scripts/experimental/unidock/run_stage52b_ppara_train374_production_remote.sh \
  > stage52b_ppara_train374_production.log 2>&1 &

echo $! | tee stage52b.pid
tail -f stage52b_ppara_train374_production.log
```

The launcher reuses the first compatible environment among
`qubo-unidock-stage08`, `qubo-unidock-stage13`, and `qubo-unidock-stage07`.
Set `STAGE52_ENV_NAME` only when a specific compatible environment is required.

## Resume and partitioning

Rerun the same command after interruption; `--resume` is enabled internally.
For separate instances, set `SEED_IDS=seed0`, `seed1`, or `seed2`. Merge the
resulting `results/runs/stage52b_ppara_train374_unidock113_production/batches`
directories into one tree, then run with `FINALIZE_ONLY=1`.

## Interpretation boundary

The resulting matrix permits only train-side exploratory QUBO and comparator
analysis. It cannot repair Stage51, and it does not establish independent
validation performance, QUBO superiority, quantum speedup, or quantum advantage.
