# CDK2 30A/30D Docking Checkpoint

## Pause reason

The computer is being shut down temporarily. All active Vina, Conda, and
Python processes were stopped after checkpoint files were written.

## Benchmark

- Ligands: 30 DUD-E actives + 30 DUD-E decoys
- Preparation: RDKit QC, ETKDGv3 3D embedding, MMFF/UFF optimization, Meeko
  PDBQT preparation
- Ligand preparation status: 60/60 PDBQT files generated successfully
- Docking protocol: AutoDock Vina 1.2.7, shared aligned box, `exhaustiveness=1`,
  `num_modes=1`, `cpu=8`
- Base seed: `20260727`
- Input manifest: `data/processed/dude_cdk2_benchmark_30a_30d_pdbqt_manifest.csv`

## Checkpoint status

| Receptor | Checkpoint ligand count | Successful | Failed | Final score table |
|---|---:|---:|---:|---|
| 1AQ1 | 60 | 60 | 0 | complete |
| 1HCL | 60 | 60 | 0 | complete |
| 1JVP | 60 | 60 | 0 | complete |
| 2C68 | 52 | 52 | 0 | incomplete |
| 3RKB | 0 | 0 | 0 | not started |

The incomplete 2C68 file is:

```text
results/docking/dude_cdk2_benchmark_30a_30d_e1_2C68_scores.checkpoint.csv
```

## Resume commands

Resume 2C68 first:

```powershell
conda run -n qubo-receptor-ensemble python scripts/batch_vina_docking.py `
  --manifest data/processed/dude_cdk2_benchmark_30a_30d_pdbqt_manifest.csv `
  --vina-exe environment/bin/vina_1.2.7_win.exe `
  --receptor receptors/prepared/2C68_A_aligned_receptor.pdbqt `
  --receptor-id CDK2_2C68_aligned_prepared `
  --config configs/cdk2_expanded_pool_exhaustiveness1_vina.txt `
  --output-dir results/docking/dude_cdk2_benchmark_30a_30d_e1_2C68 `
  --log-dir logs/dude_cdk2_benchmark_30a_30d_e1_2C68 `
  --score-table results/docking/dude_cdk2_benchmark_30a_30d_e1_2C68_scores.csv `
  --base-seed 20260727 `
  --resume
```

Then run 3RKB with the same command pattern and replace `2C68` with `3RKB`:

```text
receptor: receptors/prepared/3RKB_A_aligned_receptor.pdbqt
receptor-id: CDK2_3RKB_aligned_prepared
output/log/score stem: dude_cdk2_benchmark_30a_30d_e1_3RKB
```

Do not build the final matrix until 2C68 reaches 60/60 and 3RKB reaches
60/60. Do not treat the checkpoint CSV as a completed score table.

## Repository state at pause

- Last pushed commit: `48085b2 feat: validate expanded receptor pool and local QAOA`
- Current uncommitted code change: `scripts/batch_vina_docking.py` adds
  preflight checks for the Vina executable and receptor PDBQT path.
- Existing unrelated untracked files were left untouched:
  `configs/cdk2_common_box_vina.txt`, `receptors/aligned/`, and
  `reports/stage-02/progress_since_last_meeting_2026-07-10.md`.
