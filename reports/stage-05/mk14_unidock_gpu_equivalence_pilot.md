# Stage 5 MAPK14 Uni-Dock GPU Equivalence Pilot

Date: 2026-07-22

Source-layout note: the commands recorded below preserve the paths embedded in
the already executed result bundles. The maintained source was subsequently
isolated under `scripts/experimental/unidock/`; historical archives and their
SHA-256 values were not rewritten.

## Purpose

The preregistered fresh MAPK14 validation workload contains 23,640 official
AutoDock Vina 1.2.7 CPU jobs. A 64-vCPU instance is not readily available, so
this pilot tests whether a single RTX 4090 and Uni-Dock can provide a useful
speedup without silently treating a different search engine as identical.

This package does not contain or dock any fresh-validation or test ligand. It
uses the consumed Train-160 panel and existing CPU e32 evidence only.

## Frozen Pilot

- Ligands: 160 development-train rows, 80 active and 80 decoy
- Receptors: 2BAJ, 2QD9, 3K3J, 3KQ7, and 3MPT
- Seeds: 20260801, 20260802, and 20260803
- GPU receptor-ligand-seed pairs: 2,400
- Uni-Dock package: 1.1.3
- Scoring: vina
- Search profile: explicit detail parameters
- Exhaustiveness: 512
- Maximum search steps: 40
- Refinement steps: 5
- Output modes: 1
- Box center: (-0.49, 3.26, 21.83) Angstrom
- Box size: (22, 24, 32) Angstrom

The explicit detail parameters are the high-accuracy Uni-Dock v1 profile. Its
exhaustiveness value is not interpreted as numerically equivalent to official
Vina e32; equivalence is measured from the resulting scores and ranks.

## Gate

The audit requires all 2,400 pairs and checks the following frozen criteria:

- overall median absolute score delta no greater than 0.5 kcal/mol;
- overall 95th-percentile absolute delta no greater than 1.0 kcal/mol;
- minimum Spearman correlation across 15 receptor-seed groups at least 0.95;
- median group Top-5% overlap at least 0.80;
- at least 5-fold throughput over the recorded 32-vCPU Train-160 run;
- active/decoy mean signed-delta gap no greater than 0.5 kcal/mol.

No ROC-AUC, PR-AUC, BEDROC, EF, QUBO objective, or receptor subset is fitted or
evaluated in this pilot.

## Interpretation

A pass admits this exact GPU engine and parameter profile only as a candidate
for a separately preregistered Train-696 recomputation. The existing CPU score
normalization and QUBO cannot be mixed directly with GPU fresh-validation
scores.

A fail means this profile cannot replace official CPU Vina. It may motivate a
new train-only parameter diagnostic, or the project can retain the CPU route.

## Observed Result

The RTX 4090 run completed all 2,400 receptor-ligand-seed pairs in 263.14
seconds. Throughput was 9.12 pairs/second, or 66.20-fold faster than the
recorded 32-vCPU reference. GPU execution therefore succeeded technically.

The frozen equivalence gate failed:

- median absolute score delta: 0.139 kcal/mol, pass;
- 95th-percentile absolute delta: 1.067 kcal/mol, fail;
- minimum receptor-seed Spearman: 0.819, fail;
- median Top-5% overlap: 0.625, fail;
- active/decoy mean-delta gap: 3474.229 kcal/mol, fail;
- throughput speedup: 66.20-fold, pass.

Uni-Dock 1.1.3 with this profile is therefore not admitted as a replacement
for AutoDock Vina 1.2.7 CPU. Train-696 and fresh validation remain closed to
this GPU protocol.

The returned archive has SHA-256
`CC1C10EE167CAFB56E981CCF17846EB1EE4E7BA7C2EA0B4882BD609CDC2EB6F4`.
All 15 batch summaries and all 36 embedded config, score-table, and log hashes
were verified locally.

## Failure Diagnosis

Four Train-160 decoys contain Meeko flexible-macrocycle closure pseudoatom
types `CG0` and `G0`:

- `MK14_decoy_L017314`
- `MK14_decoy_L018641`
- `MK14_decoy_L029808`
- `MK14_decoy_L031427`

These four ligands account for exactly 60 pairs across five receptors and
three seeds. Every one of the 60 pairs differs from CPU Vina by more than 10
kcal/mol; the minimum, median, and maximum absolute deltas are 24.987,
128.534, and 1,861,943.946 kcal/mol. No other ligand exceeds 10 kcal/mol.
The Uni-Dock logs also contain an output-coordinate-size mismatch. These
scores are nonphysical and diagnose an engine/input compatibility failure,
not unusual binding.

Meeko uses numbered `CG*` and `G*` pseudoatoms to represent flexible ring
closure. Reports against GPU Vina-family programs document that these atom
types are not generally accepted or handled equivalently
([Meeko issue 212](https://github.com/forlilab/Meeko/issues/212),
[Meeko issue 158](https://github.com/forlilab/Meeko/issues/158)). Meeko 0.7.1
provides `--rigid_macrocycles` to keep the input macrocycle conformation rigid
and avoid that flexible closure representation.

As a diagnostic only, the four affected ligands were excluded from a second
calculation without changing the formal result. Across the remaining 156
ligands and 2,340 pairs:

- overall Pearson: 0.965;
- overall Spearman: 0.945;
- median absolute delta: 0.132 kcal/mol;
- 95th-percentile absolute delta: 0.824 kcal/mol;
- minimum receptor-seed Spearman: 0.895;
- median Top-5% and Top-10% overlap: 0.875 and 0.875;
- active/decoy mean-delta gap: 0.040 kcal/mol.

This confirms that the catastrophic values come from macrocycle handling, but
the minimum group Spearman still misses the frozen 0.95 threshold. The
diagnostic exclusion cannot replace the formal gate because all four excluded
Train-160 ligands are decoys. The same pseudoatom representation occurs in 15
Train-696 ligands (1 active, 14 decoys) and 54 fresh-validation input ligands
(all decoys); silently dropping them would change class composition and bias
the evaluation.

## Safeguards Added

The GPU runner now:

- detects numbered Meeko `CG*`/`G*` closure pseudoatoms before execution;
- rejects them unless a consumed-train diagnostic policy is explicit;
- fixes `OMP_NUM_THREADS=1` rather than inheriting the invalid value zero;
- rejects a score whose absolute magnitude exceeds 100 kcal/mol.

The original archive and failed gate outputs remain unchanged. The independent
diagnosis is stored in
`data/stage05_mk14_unidock_gpu_equivalence_failure_diagnostic.json`.

## Next Decision

The next GPU work must remain train-only and exploratory:

1. Re-prepare the four affected Train-160 ligands with
   `mk_prepare_ligand.py --rigid_macrocycles` and test their 60 fixed pairs.
2. Run a higher-search diagnostic on the nonmacrocycle receptor-seed groups
   that missed rank equivalence.
3. If both diagnostics are satisfactory, preregister and rerun the complete
   Train-160 equivalence gate with one revised protocol.
4. If that full gate still fails, retain official CPU Vina for Train-696 and
   fresh validation.

No threshold will be loosened retrospectively, and no fresh-validation score
will be calculated during these diagnostics.

## Rigid-Macrocycle Follow-up

The four affected consumed-train ligands were deterministically re-prepared
with Meeko 0.7.1 and `--rigid_macrocycles`. Two independent preparation passes
produced identical PDBQT hashes. The physical ligand atoms, charges, original
Train-160 row order, and seed offsets were preserved; the two `G0` closure
pseudoatoms per ligand were removed and the corresponding `CG0` atoms were
restored to ordinary atom types. No numbered `CG*` or `G*` type remains in the
revised Train-160 manifest.

Official AutoDock Vina 1.2.7 CPU e32 was then rerun for four ligands, five
receptors, and three original seed-offset replicates:

- expected and observed pairs: 60 and 60;
- failed pairs: 0;
- search-quality warnings: 0;
- score range: -11.56 to -7.638 kcal/mol;
- median three-seed score range per receptor-ligand pair: 0.0275 kcal/mol;
- maximum three-seed score range: 0.170 kcal/mol.

Rigid preparation changed the score relative to the previous flexible
macrocycle representation. Across all 60 paired comparisons, the median
rigid-minus-flexible delta was -1.729 kcal/mol, with a range from -2.423 to
+0.178 kcal/mol. The new GPU inputs therefore cannot be compared fairly with
the old flexible-macrocycle CPU scores.

Three revised 800-row CPU references were built for the five GPU receptors.
Each seed table contains 780 unchanged official e32 nonmacrocycle scores and
20 newly measured rigid-macrocycle scores. No score was imputed, deleted, or
copied across seeds.

Two consumed-train GPU profiles are now frozen:

1. `detail_rigid_fix`: the original Uni-Dock detail settings
   (`exhaustiveness=512`, `max_step=40`) isolate the input repair.
2. `enhanced_rigid_search`: doubled search settings
   (`exhaustiveness=1024`, `max_step=80`) test whether additional search can
   rescue the remaining nonmacrocycle rank disagreement.

Both profiles retain the original equivalence thresholds. Running the second
profile is a declared post-failure diagnostic, not retrospective threshold or
model tuning. Validation and test rows remain excluded.

## Follow-up Remote Workflow

Extract the new rigid-macrocycle diagnostic bundle in persistent storage and
verify its manifest before enabling the GPU:

```bash
mkdir -p /root/autodl-tmp/stage05_mk14_unidock_rigid_gpu
tar -xzf /root/autodl-tmp/stage05_mk14_unidock_rigid_gpu_diagnostics_v1.tar.gz \
  -C /root/autodl-tmp/stage05_mk14_unidock_rigid_gpu
cd /root/autodl-tmp/stage05_mk14_unidock_rigid_gpu
sha256sum -c bundle_manifest.sha256
```

Reuse or create the pinned Uni-Dock environment, then run both profiles:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate qubo-unidock
unidock --version
chmod +x scripts/run_unidock_rigid_gpu_diagnostics_remote.sh
nohup bash scripts/run_unidock_rigid_gpu_diagnostics_remote.sh \
  > stage05_mk14_unidock_rigid_gpu_diagnostics.log 2>&1 &
echo $!
```

Monitor with:

```bash
tail -n 80 -f stage05_mk14_unidock_rigid_gpu_diagnostics.log
```

The command is resumable after instance shutdown. On completion, download
`stage05_mk14_unidock_rigid_train160_gpu_diagnostics_core_v1.tar.gz`.

## Follow-up GPU Result

The returned follow-up archive has SHA-256
`25F9B847359C4954ED3C53BE55BDDEF2B97B229E93D04699215E34D697BCCE6B`.
It contains both complete 2,400-pair profiles. Seventy-two embedded config,
score-table, group-table, and batch log hashes were verified, and all 30 batch
summaries matched the top-level batch indexes.

The rigid-macrocycle compatibility repair succeeded. No nonphysical score or
numbered closure pseudoatom remained. Across the 60 affected pairs, the
original-detail profile had a median absolute CPU/GPU delta of 0.066 kcal/mol
and a maximum of 0.930 kcal/mol. The enhanced profile had a median of 0.0335
kcal/mol and a maximum of 1.081 kcal/mol. The earlier million-scale values
were therefore caused by the incompatible flexible-macrocycle representation.

The original-detail profile completed in 251.30 seconds and was 69.32-fold
faster than the recorded 32-vCPU reference. It passed six of seven frozen
checks but failed minimum group Spearman:

- overall Spearman: 0.9422;
- median absolute delta: 0.144 kcal/mol;
- 95th-percentile absolute delta: 0.903 kcal/mol;
- median Top-5% overlap: 0.875;
- best-receptor identity agreement: 0.8042;
- minimum group Spearman: 0.8988, below the frozen 0.95 threshold.

The enhanced-search profile completed in 565.45 seconds and retained a
30.81-fold speedup. It improved every one of the 15 receptor-seed Spearman
values, but still failed the same single gate:

- overall Spearman: 0.9751;
- median absolute delta: 0.054 kcal/mol;
- 95th-percentile absolute delta: 0.575 kcal/mol;
- median Top-5% and Top-10% overlap: 0.875 and 0.9375;
- best-receptor identity agreement: 0.9063;
- minimum group Spearman: 0.9305, below 0.95.

The limiting group was `seed2 / MK14_3MPT_aligned`. The largest rank shifts in
that group were concentrated in several ordinary nonmacrocycle ligands,
including decoys `L027360`, `L003244`, `L032504`, and `L033366`. Removing the
four repaired macrocycles diagnostically still gave a minimum group Spearman
of 0.9290, confirming that macrocycle handling no longer explains the failure.

Uni-Dock emitted two coordinate-size output-container warnings in the detail
profile and one in the enhanced profile. No warning produced a nonphysical
score, and all affected batch score/log hashes were intact, but the core
archive omitted poses and therefore cannot support atom-level output-pose
validation for those warning cases. Future GPU runners must elevate this log
condition to an explicit engine warning and retain targeted pose outputs.

Both frozen profiles formally failed. The enhanced profile is substantially
closer and consistently better, but it is not admitted as a drop-in
replacement for official CPU Vina. Continuing to increase search parameters
on the same consumed Train-160 panel until the threshold passes would be
post-hoc parameter fishing. The formal validation path should retain official
CPU Vina. A future Uni-Dock path must be preregistered as a distinct engine,
recompute its complete training evidence, and be evaluated without mixing CPU
and GPU score matrices.

## Original Remote Workflow

Extract the uploaded archive into persistent storage and verify every source
file before installing or running anything:

```bash
mkdir -p /root/autodl-tmp/stage05_mk14_unidock_gpu_pilot
tar -xzf /root/autodl-tmp/stage05_mk14_unidock_gpu_equivalence_v1.tar.gz \
  -C /root/autodl-tmp/stage05_mk14_unidock_gpu_pilot
cd /root/autodl-tmp/stage05_mk14_unidock_gpu_pilot
sha256sum -c bundle_manifest.sha256
```

Create the pinned environment after GPU mode is restored:

```bash
conda env create -f environment/stage05_unidock_gpu.yml
source /root/miniconda3/etc/profile.d/conda.sh
conda activate qubo-unidock
unidock --version
```

Run in the background:

```bash
chmod +x scripts/run_stage05_mk14_unidock_gpu_equivalence_remote.sh
nohup bash scripts/run_stage05_mk14_unidock_gpu_equivalence_remote.sh \
  > stage05_mk14_unidock_gpu_equivalence.log 2>&1 &
echo $!
```

Monitor without interrupting the run:

```bash
tail -n 80 -f stage05_mk14_unidock_gpu_equivalence.log
```

The same `nohup` command is resumable after an instance restart. Completed
receptor-seed batches are verified by score-table and pose hashes before they
are skipped.

On completion, download:

```text
stage05_mk14_unidock_train160_gpu_equivalence_core_results_v1.tar.gz
```

The core archive excludes pose files but includes all score comparisons,
batch logs, environment evidence, and the final gate decision. Pose files
remain in the persistent remote working directory for targeted follow-up.
