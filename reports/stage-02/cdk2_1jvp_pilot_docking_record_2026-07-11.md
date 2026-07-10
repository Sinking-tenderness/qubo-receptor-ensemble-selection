# CDK2 1JVP Pilot Docking Record

## Protocol

- Receptor: `CDK2_1JVP_P_aligned_prepared`
- Receptor PDBQT: `receptors/prepared/1JVP_P_aligned_receptor.pdbqt`
- Ligand manifest: `data/processed/dude_cdk2_pilot_2a_2d_pdbqt_manifest.csv`
- Vina: AutoDock Vina v1.2.7
- Scoring function: `vina`
- Box center: `(0.52, 27.06, 8.97)` A
- Box size: `(18, 18, 16)` A
- Exhaustiveness: 16
- Number of modes: 10
- Base seed: `20260709`; ligand seeds are base seed plus manifest index.

## Result

All four pilot ligands completed successfully. The batch output contains 40
pose rows and no failed ligands. The checkpoint CSV was written during the
run; the final `--resume` command reused completed pose/log files without
rerunning Vina.

| Ligand | Label | Top-1 score (kcal/mol) | Runtime (s) | Status |
|---|---|---:|---:|---|
| CDK2_A0009 | active | -11.96 | 160.464 | ok |
| CDK2_A0010 | active | -11.86 | 53.677 | ok |
| CDK2_D0037 | decoy | -10.76 | 39.110 | ok |
| CDK2_D0022 | decoy | -10.09 | 160.375 | ok |

The two actives rank above the two decoys in this pilot. This is only a
pipeline check, not a screening-performance claim: four molecules cannot
support stable ROC-AUC, EF, or BEDROC conclusions.

## Reproducibility note

The first outer command reached its wall-clock timeout while a Vina child
process was still running. The checkpoint preserved the completed ligand and
the process later completed the remaining ligands. This validates the need to
use the checkpoint table and `--resume` rather than restarting a long batch.

## Next step

Run the same command with the complete 60-ligand manifest, preserving the
same receptor, box, Vina version, seed rule, and checkpoint path. Do not
interpret the result until all 60 ligand statuses and failures have been
audited.
