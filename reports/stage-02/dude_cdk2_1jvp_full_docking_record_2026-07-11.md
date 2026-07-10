# CDK2 1JVP Full Virtual Screening Record

## Protocol

- Receptor: `CDK2_1JVP_P_aligned_prepared`
- Ligands: 10 actives + 50 DUD-E decoys
- Vina: 1.2.7, `vina` scoring function
- Box center: `(0.52, 27.06, 8.97)` A
- Box size: `(18, 18, 16)` A
- Exhaustiveness: 16
- Modes per ligand: 10
- Seed rule: `20260709 + manifest index`

## Completeness

- Ligands selected: 60
- Successful ligands: 60
- Failed ligands: 0
- Pose rows: 600
- Final score table: `results/metrics/dude_cdk2_1jvp_full_scores.csv`
- Checkpoint table: `results/metrics/dude_cdk2_1jvp_full_scores.checkpoint.csv`

## 1JVP Metrics

| Metric | Value |
|---|---:|
| ROC-AUC | 0.718 |
| PR-AUC / average precision | 0.436 |
| BEDROC, alpha=20 | 0.531 |
| EF1% | 6.0 |
| EF5% | 4.0 |
| EF10% | 2.0 |

Bootstrap 95% intervals from 2000 resamples:

- ROC-AUC: 0.529-0.880
- PR-AUC: 0.158-0.690
- BEDROC: 0.014-0.849
- EF1%: 0-12
- EF5%: 0-8
- EF10%: 0-5

The wide intervals are expected for a 60-ligand benchmark with only 10
actives. The result is useful as a reproducible baseline, not as a precise
estimate of prospective screening performance.

## Comparison With 1AQ1

| Quantity | 1AQ1 | 1JVP |
|---|---:|---:|
| ROC-AUC | 0.644 | 0.718 |
| PR-AUC | 0.459 | 0.436 |
| BEDROC | 0.653 | 0.531 |
| EF1% | 6.0 | 6.0 |
| EF5% | 4.0 | 4.0 |
| EF10% | 3.0 | 2.0 |
| active molecules in Top10 | 4 | 3 |

Score correlation across the 60 shared ligands:

- Spearman: 0.802
- Pearson: 0.838
- Top10 overlap: 7/10
- Mean score delta, 1JVP minus 1AQ1: +0.054 kcal/mol
- Mean active delta: -0.118 kcal/mol
- Mean decoy delta: +0.089 kcal/mol

The two receptors produce related but non-identical rankings. 1JVP improves
pairwise ROC-AUC in this benchmark, while 1AQ1 has better BEDROC, EF10%, and
Top10 active count. Therefore no receptor should be declared universally
best from one metric.

## Simple Ensemble Baselines

Using the two-receptor score matrix:

- Minimum score across receptors: ROC-AUC 0.704, PR-AUC 0.471, BEDROC
  0.623, EF1% 6, EF5% 4, EF10% 3.
- Mean score across receptors: ROC-AUC 0.714, PR-AUC 0.459, BEDROC 0.590,
  EF1% 6, EF5% 4, EF10% 3.

These are descriptive baselines only. They do not establish that two
receptors are complementary enough for a useful ensemble, and they are not
an independent test because the same 60-ligand benchmark is used for
comparison.

## Relevance To QUBO Selection

1AQ1 and 1JVP have correlated scores, but their early rankings differ. A
future receptor-subset objective should therefore include both individual
screening signal and ligand-level complementarity, while controlling the
number of selected receptors. Optimizing only ROC-AUC or EF1% on this small
benchmark would risk selecting a benchmark-specific receptor and would not
demonstrate generalization.
