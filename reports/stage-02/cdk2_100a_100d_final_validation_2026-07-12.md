# CDK2 100A/100D Final Validation

## Benchmark and docking completeness

- Ligands: 100 DUD-E actives and 100 DUD-E decoys.
- Quality control: 200/200 valid RDKit molecules, no duplicate canonical
  SMILES, and no multi-fragment ligands.
- Scaffold split: 196 Bemis-Murcko scaffold groups; 60 active + 60 decoy
  train, 20 active + 20 decoy validation, and 20 active + 20 decoy test.
- Receptors: 1AQ1, 1H00, 1HCL, 1JVP, 1Y8Y, 2C68, 2C69, and 3RKB.
- Docking protocol: Vina 1.2.7, common aligned box, `exhaustiveness=1`,
  `num_modes=1`, eight parallel workers with one CPU per worker.
- Docking completeness: 1600/1600 receptor-ligand pairs succeeded with no
  missing representative scores.

## Raw-score baselines

| Method | ROC-AUC | PR-AUC | BEDROC(alpha=20) | EF5% |
|---|---:|---:|---:|---:|
| best individual receptor, 1AQ1 | 0.732 | 0.787 | 0.963 | 2.0 |
| all-receptor minimum score | 0.713 | 0.752 | 0.931 | 2.0 |
| all-receptor mean score | 0.702 | 0.714 | 0.856 | 2.0 |

Using every receptor is not a strong baseline: it underperformed the best
single receptor on this benchmark.

## Leakage-aware QUBO evaluation

The candidate method used a fixed subset size of two, train-only min-max
normalization for each receptor, QUBO selection on train, hyperparameter choice
on validation, and five-fold scaffold-group outer CV. The experiment was
repeated with scaffold seeds 20260802, 20260803, and 20260804. Scores were
averaged only after each ligand had an out-of-fold prediction in each repeat.

| Method | ROC-AUC | PR-AUC | BEDROC(alpha=20) | EF5% |
|---|---:|---:|---:|---:|
| train-selected single receptor | 0.722 | 0.769 | 0.948 | 2.0 |
| QUBO-selected two-receptor subset | 0.722 | 0.759 | 0.925 | 2.0 |
| all-receptor mean | 0.719 | 0.730 | 0.866 | 1.8 |

Paired bootstrap, QUBO minus single receptor, used 5000 resamples:

- ROC-AUC: `-0.00005`, 95% CI `[-0.0268, +0.0255]`
- PR-AUC: `-0.0101`, 95% CI `[-0.0391, +0.0166]`
- BEDROC: `-0.0227`, 95% CI `[-0.0913, +0.0194]`

## Decision

The 30A/30D candidate improvement did not reproduce on the larger 100A/100D
benchmark. The current docking-score coverage QUBO is therefore **not
validated** as an effective receptor-subset selection method. It is not sent to
local QAOA or quantum hardware as a performance claim.

The pipeline itself is validated: data preparation, receptor preparation,
parallel docking, score-matrix construction, train-only normalization,
scaffold-group CV, and paired bootstrap are reproducible. The scientific next
step is to replace the current score-only complementarity signal with validated
structural/contact features and then test on AF2/MD-derived conformer pools and
additional targets. This negative result should be retained in any report.
