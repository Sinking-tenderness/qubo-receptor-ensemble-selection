# Stage 5 MAPK14 Expanded Train-696 Preparation Record

Date: 2026-07-19

## Decision

The fixed MAPK14 QUBO candidate did not generalize across repeated scaffold
partitions on the 80-active/80-decoy development-train matrix. Further tuning
of that matrix was stopped. The authorized next step is to expand the sealed
development-train panel before any validation or test access.

## Frozen Panel

- target: MAPK14 (`MK14`)
- receptors: 8 previously admitted conformers
- ligands: 696 development-train rows
- labels: 348 active and 348 decoy
- unique split groups: 696
- retained old panel: 160 ligands
- newly selected panel: 536 ligands (268 active and 268 decoy)
- validation rows: 0
- test rows: 0

The 348 active rows are all available frozen-train actives. The old 80 train
decoys are retained, and 268 additional train decoys were selected
deterministically from previously unused split groups.

## Ligand Preparation

The existing 160 prepared PDBQT files were reused. The 536 new ligands were
embedded with RDKit and prepared with Meeko 0.7.1.

- new RDKit 3D structures: 536/536 produced
- new Meeko PDBQT files: 536/536 valid
- complete combined PDBQT manifest: 696/696 `ok`
- SDF preparation status: 666 `ok`, 30 retained warnings
- nonzero formal-charge rows: 126

An SDF optimization warning records non-convergence within the configured
iteration limit; it is not silently converted into a preparation failure.
Every admitted ligand nevertheless has a parseable, hashed PDBQT file.

## Docking Workload

The old 160-ligand e32 results are reused only after exact manifest, receptor,
protocol, seed, and output-hash validation. Only the 536 new ligands are sent
to Vina.

- Vina version: 1.2.7
- exhaustiveness: 32
- output modes: 1
- CPU threads per Vina process: 2
- paired base seeds: 20260801, 20260802, 20260803
- new jobs: 536 ligands x 8 receptors x 3 seeds = 12,864
- reused jobs: 160 ligands x 8 receptors x 3 seeds = 3,840
- complete evidence cells: 16,704
- complete aggregated receptor-ligand pairs: 5,568

The recommended 64-vCPU layout is 32 concurrent workers with two Vina CPU
threads per worker. AutoDock Vina 1.2.7 is CPU-based, so a GPU is not required.

## Admission Boundary

The returned scores must first pass three-seed aggregation and an audited merge
with the frozen 160-ligand e32 evidence. No e64 diagnostic score may replace an
e32 value. The resulting 696x8 matrix is development-train evidence only; it
does not by itself establish enrichment, biological activity, affinity
accuracy, QUBO benefit, or quantum advantage.

## Remote Execution

Recommended instance:

- 64-96 vCPU
- at least 64 GiB RAM
- persistent system or data disk
- GPU not required

The expected wall time is approximately 13-26 hours. The lower estimate
assumes near-linear scaling from the earlier 32-vCPU e32 run; the upper bound
allows for weaker process scaling and ligand-dependent search time.

Upload `stage05_mk14_expanded8_train696_e32_remote_v1.tar.gz` to
`/root/autodl-tmp/`, then run:

```bash
set -euo pipefail

ARCHIVE=/root/autodl-tmp/stage05_mk14_expanded8_train696_e32_remote_v1.tar.gz
WORK=/root/autodl-tmp/stage05_mk14_expanded8_train696_e32_v1
LOG=/root/autodl-tmp/stage05_mk14_expanded8_train696_e32.log
PID=/root/autodl-tmp/stage05_mk14_expanded8_train696_e32.pid

mkdir -p "$WORK"
tar -xzf "$ARCHIVE" -C "$WORK"
cd "$WORK"
sha256sum -c bundle_manifest.sha256

nohup bash scripts/run_stage05_mk14_train696_e32_remote.sh \
  > "$LOG" 2>&1 &
echo $! | tee "$PID"
```

Monitor without interrupting the run:

```bash
tail -f /root/autodl-tmp/stage05_mk14_expanded8_train696_e32.log
```

```bash
ps -p "$(cat /root/autodl-tmp/stage05_mk14_expanded8_train696_e32.pid)" \
  -o pid,etime,%cpu,%mem,cmd
```

The runner uses `--resume`. If the instance stops but the persistent work
directory remains, launch the same `nohup` command from that directory;
completed receptor tables and checkpointed ligand rows are reused.

On success, download:

`stage05_mk14_expanded8_train696_e32_core_results_v1.tar.gz`

Its final SHA-256 is printed at the end of the log. The archive contains all
new three-seed evidence, the 536-ligand aggregate, and the audited 696x8 merge.
