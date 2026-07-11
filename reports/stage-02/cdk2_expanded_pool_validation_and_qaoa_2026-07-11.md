# CDK2 Expanded Receptor Pool Validation and QAOA Simulation

## Expanded docking matrix

Five receptor conformers were evaluated with the same fast triage protocol:

- 10 active and 10 decoy ligands, fixed by seed `20260723`
- common aligned box: center `(0.52, 27.06, 8.97)`, size `(18, 18, 16)`
- AutoDock Vina 1.2.7, `exhaustiveness=1`, `num_modes=1`, `cpu=8`
- 5 receptors × 20 ligands = 100 ligand-receptor pairs
- successful pairs: 100/100; failed pairs: 0

The matrix and audit summary are local ignored outputs:

- `results/metrics/dude_cdk2_expanded_e1_20_matrix.csv`
- `results/metrics/dude_cdk2_expanded_e1_20_summary.json`

The receptor provenance is recorded in
`data/processed/cdk2_expanded_receptor_pool_manifest.csv`.

## Outer cross-validation

The outer test fold was never used for receptor selection or QUBO weight
selection. Five folds were used, with the next fold serving as validation and
the remaining folds as train.

### Existing 3-receptor, 60-ligand matrix

When QUBO utility and validation selection targeted BEDROC(alpha=20), the
out-of-fold results were:

| Method | ROC-AUC | PR-AUC | BEDROC | Top-10 actives |
|---|---:|---:|---:|---:|
| train-selected single receptor | 0.718 | 0.436 | 0.531 | 3 |
| QUBO sparse subset | 0.714 | 0.459 | 0.590 | 4 |
| all-receptor mean | 0.690 | 0.424 | 0.542 | 3 |

The paired bootstrap mean delta for QUBO minus single was `+0.0666` for
BEDROC, with 95% CI `[-0.0112, 0.2660]`. This is a promising early-recognition
signal, but the interval crosses zero and does not establish a significant
improvement.

### Expanded 5-receptor, 20-ligand matrix

With the same outer-CV protocol and mean-score aggregation:

| Method | ROC-AUC | PR-AUC | BEDROC |
|---|---:|---:|---:|
| train-selected single receptor | 0.830 | 0.851 | 0.968 |
| QUBO sparse subset | 0.730 | 0.773 | 0.964 |
| all-receptor mean | 0.740 | 0.783 | 0.964 |

The expanded result is negative for the current QUBO formulation. It shows
that adding structurally different candidates is not sufficient; the ligand
benchmark and QUBO objective must be improved before claiming innovation.

## Local quantum simulation

The 5-receptor coverage-aware QUBO was simulated with QAOA p=1 on a separate
Qiskit Aer environment. Across seeds `20260724`, `20260725`, and `20260726`,
the exact optimum `1AQ1 + 1JVP` appeared as the best sampled energy. However,
the most frequent bitstring was often `1JVP + 3RKB`, not the exact optimum.

Therefore the evidence supports:

- the QUBO encoding and local quantum simulation pipeline run correctly;
- the simulator can sample the exact optimum for this five-variable problem;
- p=1 QAOA does not yet show stable concentration at the optimum;
- there is no evidence of quantum advantage, and no quantum hardware run has
  been performed.

## Current scientific conclusion

The project has moved beyond a toy three-variable demonstration: it now has a
five-receptor docking matrix, leakage-aware outer validation, paired
uncertainty analysis, and a five-qubit local QAOA simulation. The intended
innovation is not yet validated. The next required experiment is a larger
active/decoy benchmark with more than ten actives, followed by frozen-test
comparison against single-best, random, clustering, greedy, and QUBO subset
selection.
