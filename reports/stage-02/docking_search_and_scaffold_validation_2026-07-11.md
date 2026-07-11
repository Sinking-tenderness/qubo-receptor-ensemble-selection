# Docking Search and Scaffold Validation Checkpoint

## Search and runtime benchmark

The benchmark used the same CDK2 1AQ1 receptor, four ligands (2 actives and
2 decoys), the same box, and Vina 1.2.7. Each run used `num_modes=1` and
`cpu=1` unless stated otherwise.

| Setting | Total wall time | Notes |
|---|---:|---|
| `exhaustiveness=1`, `cpu=1`, serial | 125.0 s | 4 ligands |
| `exhaustiveness=8`, `cpu=1`, serial | 749.4 s | 4 ligands |
| `exhaustiveness=8`, `cpu=4`, 2 ligands | 43.5 s | same sampled pair |
| `exhaustiveness=16`, `cpu=4`, 2 ligands | completed | 62.3 s and 85.5 s per ligand |
| `exhaustiveness=1`, `cpu=1`, 4 parallel workers | 68.9 s | same 4 ligands |

For A0028, the score was `-10.08` at `e=1`, `-10.09` at `e=8,cpu=4`,
and `-10.03` at `e=16,cpu=4`. For D0011, the score was `-7.545` at both
`e=8,cpu=4` and `e=16,cpu=4`. Higher exhaustiveness therefore increased
runtime but did not guarantee a better score.

The controlled parallel runner is
`scripts/batch_vina_docking_parallel.py`. It records worker count and CPU per
worker, writes a checkpoint after each completed ligand, and supports resume.

## Scaffold-disjoint evaluation

The random split was repeated with Bemis-Murcko scaffold-disjoint assignment.
The resulting split contains 18 active + 18 decoy train, 6 active + 6 decoy
validation, and 6 active + 6 decoy test molecules. There are 59 scaffolds and
no scaffold appears in more than one split.

Using the existing five-receptor `e=1` matrix, the train-selected coverage
QUBO chose `1AQ1 + 3RKB`. On the scaffold-disjoint test set, mean-score
metrics were:

| Method | ROC-AUC | PR-AUC | BEDROC(alpha=20) |
|---|---:|---:|---:|
| best single 3RKB | 0.917 | 0.944 | 1.000 |
| QUBO `1AQ1 + 3RKB` | 0.917 | 0.931 | 0.999 |
| all-receptor mean | 0.917 | 0.931 | 0.999 |

This is not evidence of an ensemble gain. It is a stricter negative result:
the current objective does not improve over the strongest single receptor.

## Decision

The main protocol should not be changed to `exhaustiveness=16` merely because
it is more expensive. The next full benchmark will use an explicitly recorded
CPU allocation and a controlled worker count. The receptor-selection method
still requires a more informative complementarity signal and validation on a
larger, scaffold-aware and eventually AF2/MD-derived conformer pool.
