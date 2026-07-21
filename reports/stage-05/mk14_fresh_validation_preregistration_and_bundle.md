# Stage 5 MAPK14 Fresh Validation Preregistration and CPU Bundle

Date: 2026-07-21

## Purpose

The Train-696 marginal pair-synergy QUBO passed its preregistered comparisons
with matched linear and nested greedy. It remains a train-only result. The
previous 40-active/40-decoy validation panel was already consumed and is
prohibited from reuse as fresh evidence.

This gate freezes a new validation panel, score normalization, receptor
subsets, metrics, uncertainty analysis, and thresholds before any new
validation docking score is generated. The locked test partition remains
unreleased and cannot be accessed automatically after this gate.

## Fresh Panel

The source is the validation portion of the original scaffold/source-ID
disjoint MAPK14 split. Every `split_group_id` represented in the consumed
40+40 panel was excluded before selection.

After that exclusion, all remaining active-containing validation groups were
included. Complete decoy-only groups were then ordered by the frozen SHA-256
selection rule and added until at least 20 decoys per active were present.

- Fresh actives: 75
- Fresh decoys: 1501
- Total ligands: 1576
- Complete split groups: 520
- Overlap with 80 consumed split groups: 0
- Locked test rows selected: 0

The single extra decoy is the required group-preserving overshoot above the
1500-decoy target.

## Frozen Methods

Every method uses `min_score` after applying per-receptor Train-696 min-max
bounds separately for the primary, sensitivity, seed0, seed1, and seed2
matrices. Validation values outside a Train-696 range are not clipped.

| Method | Frozen receptor subset |
|---|---|
| Pair-synergy QUBO | 2BAJ + 2QD9 + 3KQ7 |
| Matched linear top-k | 2BAJ + 3KQ7 + 3MPT |
| Nested exhaustive final | 2BAJ + 2QD9 + 3K3J |
| Single best | 3KQ7 |
| Nested greedy final | 2BAJ + 2QD9 + 3KQ7 |

The QUBO and final greedy subsets are identical. Their fixed validation
predictions must therefore be identical and cannot support a QUBO-over-greedy
claim.

## Acceptance

The primary endpoint is BEDROC20. QUBO must be noninferior across primary,
mean-seed, and worst-seed BEDROC to matched linear with a zero margin. The
split-group block-bootstrap 95% lower bound versus matched linear must also be
at least zero.

Against nested exhaustive, the corresponding frozen noninferiority margin is
-0.01 BEDROC. QUBO must also exceed single best on primary BEDROC. All three
seed matrices must be complete with zero failed receptor-ligand pairs.

Bootstrap sampling uses 5000 paired `split_group_id` replicates with seed
20261420. Test remains locked after either outcome.

## Preparation

- RDKit 3D: 1528 `ok`, 48 retained non-convergence warnings, 0 failed
- Meeko PDBQT: 1576 `ok`, 0 failed
- Nonzero formal charge: 630 ligands
- Fixed receptors: 5
- Receptor-ligand pairs per seed: 7880
- Total Vina jobs: 23,640
- Validation metrics calculated so far: no

The score protocol remains Vina 1.2.7, exhaustiveness 32, one pose, two CPU
threads per process, and base seeds 20260801, 20260802, and 20260803.

## Runtime Amendment

Completed Train-696 timing evidence predicted approximately 4-5 days with 16
concurrent two-thread processes. Before any fresh-validation score existed,
execution amendment 01 increased concurrency to 32 processes and 64 total CPU
threads. No scientific input or parameter changed. A 64-vCPU instance with at
least 64 GB RAM is recommended; a GPU is not used. Expected wall time is about
2-3 days, with substantial ligand-dependent variation.

## CPU Bundle

- File: `dist/stage05_mk14_fresh_validation_e32_cpu_v1.tar.gz`
- Size: 3,213,789 bytes
- SHA-256: `3D9EC8C6F6F64FFD6B8E8B092195F0D2DE8431437575F6E1CDC5FF8003259420`
- Source files: 1603
- Archive entries including manifest: 1604
- Deterministic rebuild: identical SHA-256 in two complete builds

After uploading the archive to a CPU instance:

```bash
mkdir -p /root/autodl-tmp/stage05_mk14_fresh_validation_v1
tar -xzf /root/autodl-tmp/stage05_mk14_fresh_validation_e32_cpu_v1.tar.gz \
  -C /root/autodl-tmp/stage05_mk14_fresh_validation_v1
cd /root/autodl-tmp/stage05_mk14_fresh_validation_v1
sha256sum -c bundle_manifest.sha256
chmod +x scripts/run_stage05_mk14_fresh_validation_remote.sh
nohup bash scripts/run_stage05_mk14_fresh_validation_remote.sh \
  > stage05_mk14_fresh_validation.log 2>&1 &
echo $!
```

Monitor without stopping the run:

```bash
tail -n 60 stage05_mk14_fresh_validation.log
```

The same `nohup` command can be rerun after an instance restart. Every seed
uses `--resume`, so completed receptor-ligand jobs are retained.

On success, the working directory contains:

```text
stage05_mk14_fresh_validation_e32_core_results_v1.tar.gz
```

Record its hash before downloading:

```bash
sha256sum stage05_mk14_fresh_validation_e32_core_results_v1.tar.gz
```

The remote script stops after score aggregation. It does not calculate
validation metrics. The returned archive must be admitted locally before the
single preregistered evaluation is executed.
