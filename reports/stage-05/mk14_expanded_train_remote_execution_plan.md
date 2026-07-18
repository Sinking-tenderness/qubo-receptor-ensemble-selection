# Stage 5 MAPK14 Expanded Train Docking Remote Execution Plan

Date: 2026-07-18

## Why All Eight Columns Are Recomputed

The original four-receptor development matrix used a common box with
`size_z=30`. The final eight-receptor redocking gate required `size_z=32`.
Because box size affects the stochastic Vina search, old z30 columns are not
merged with new z32 columns. All eight receptors are recomputed under one
protocol using only the 160 frozen development-train ligands.

The consumed 40-active/40-decoy validation panel is excluded from the bundle.
No test ligand is present or released.

## Frozen Workload

- receptors: 8
- ligands: 160 (`development_train` only; 80 active and 80 decoy)
- paired seeds: 20260801, 20260802, 20260803
- pairs per seed: 1,280
- total Vina jobs: 3,840
- Vina: 1.2.7, exhaustiveness 16, one output mode, 2 CPU threads/job
- orchestration: 16 workers, maximum 32 CPU threads
- box: center `(-0.49, 3.26, 21.83)` A; size `(22, 24, 32)` A

The previous 32-vCPU run required 4,651 seconds for 960 pairs per seed.
Linear scaling gives about 5.2 hours for this three-seed workload. Use 32 vCPU,
at least 32 GiB RAM, persistent disk, and no GPU.

## Bundle

- file: `stage05_mk14_expanded8_train160_e16_remote_v1.tar.gz`
- local location: workspace parent directory
- size: 2,173,740 bytes
- SHA-256: `B4A103AACDB45DA21AF4E5A928229F3ABA35121748D9E5E70AA917F841695ED0`
- source files: 189; archive entries including manifest: 190
- deterministic replay: identical archive SHA-256

## Remote Commands

Upload the archive to `/root/autodl-tmp/`, then run:

```bash
set -euo pipefail

ARCHIVE=/root/autodl-tmp/stage05_mk14_expanded8_train160_e16_remote_v1.tar.gz
WORK=/root/autodl-tmp/stage05_mk14_expanded8_train160_v1

sha256sum "$ARCHIVE"
mkdir -p "$WORK"
tar -xzf "$ARCHIVE" -C "$WORK"
cd "$WORK"
sha256sum -c bundle_manifest.sha256
python --version

nohup bash scripts/run_stage05_mk14_expanded_train_remote.sh \
  > /root/autodl-tmp/stage05_mk14_expanded8_train160.log 2>&1 &
echo $! | tee /root/autodl-tmp/stage05_mk14_expanded8_train160.pid
```

The first three commands inside the runner are audit-only checks. Each must
report 8 receptors, 160 ligands, 1,280 pairs, zero locked-test rows, zero
outputs, and zero Vina jobs before production docking begins.

## Monitoring and Resume

```bash
tail -f /root/autodl-tmp/stage05_mk14_expanded8_train160.log
```

```bash
ps -p "$(cat /root/autodl-tmp/stage05_mk14_expanded8_train160.pid)" \
  -o pid,etime,%cpu,%mem,cmd
```

If the instance stops, return to the same persistent `WORK` directory and run
the identical `nohup` command. Every seed uses `--resume`; completed receptor
tables are reused and checkpointed ligands continue without overwriting valid
scores.

## Completion Artifacts

After three seeds pass, the runner aggregates the paired scores by median and
retains minimum-score sensitivity values. It then creates:

`stage05_mk14_expanded8_train160_e16_core_results_v1.tar.gz`

The archive contains the three seed summaries and representative-score tables,
the aggregated long table, both 160x8 matrices, and the aggregation summary.
The final line of the log prints its SHA-256 for download verification.

Do not calculate validation or test metrics on the remote instance. After the
core result archive returns locally, the next authorized step is the frozen
train-only matrix admission and QUBO non-degeneracy gate.
