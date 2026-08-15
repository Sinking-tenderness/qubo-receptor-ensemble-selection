# Stage 11 MAPK14 Amendment 01 Independent Adjudication

Status: `stage11_technical_result_accepted_scientific_gate_not_passed`

## Technical result

- All 28368 poses passed independent SHA-256, score, single-model, atom-count, and atom-type checks.
- All 18 batches and 28368 batch score rows were complete.
- Known warnings: 22; unresolved warnings: 0.
- Scores above the original 100 kcal/mol guard: 4.

## Frozen candidate result

| Candidate | Primary BEDROC | Mean-seed BEDROC | Worst-seed BEDROC | PR-AUC | EF1% |
|---|---:|---:|---:|---:|---:|
| exact_pair_synergy | 0.417974 | 0.419337 | 0.418599 | 0.319335 | 17.0733 |
| qubo_forward_greedy | 0.394849 | 0.395408 | 0.391700 | 0.279827 | 14.4467 |
| direct_bedroc_greedy | 0.409294 | 0.409699 | 0.408348 | 0.305069 | 17.0733 |
| full_train_exact_secondary | 0.417371 | 0.418825 | 0.415913 | 0.320766 | 17.0733 |

| Comparison | Primary delta | Bootstrap 95% CI | Positive replicates | Gate |
|---|---:|---:|---:|---|
| exact_pair_synergy vs qubo_forward_greedy | +0.023125 | [-0.026673, +0.077607] | 81.3% | fail |
| exact_pair_synergy vs direct_bedroc_greedy | +0.008680 | [-0.035671, +0.055615] | 64.1% | fail |

## Outlier sensitivity

The four finite positive-score outliers are genuine values in their pose files. None is selected by the three-seed median or minimum aggregation. Clipping them at 100 kcal/mol therefore changes zero primary or sensitivity matrix cells. Treating them as missing or excluding all four affected ligands also leaves the confirmatory gate failed.

## Decision

The technical execution and matrix are accepted. The exact QUBO subset has a positive point estimate against both frozen greedy controls and beats the best single receptor, but both paired-bootstrap 95% intervals cross zero. This supports a receptor-ensemble/QUBO application proof of concept, not a statistically stable QUBO advantage and not a quantum computational advantage.
