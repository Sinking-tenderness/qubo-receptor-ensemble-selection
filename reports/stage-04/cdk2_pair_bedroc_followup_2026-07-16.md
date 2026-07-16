# Stage 4 Pair-BEDROC QUBO Follow-up

## Question

The expanded-16 development gate showed higher ROC-AUC and PR-AUC but lower
BEDROC20 than single-best. This follow-up asks two development-only questions:

1. Does the 16-receptor pool contain a small subset with stronger BEDROC20?
2. Can a QUBO pair reward based on direct pairwise ensemble BEDROC select it
   more reliably than the existing coverage QUBO?

The 40-ligand test split remained locked throughout this follow-up.

## Optimistic Development Upper Bound

`scripts/diagnose_development_subset_upper_bound.py` enumerated every receptor
subset of size 1, 2, or 3 after full-development min-max normalization. This is
an optimistic diagnostic, not nested CV and not a test result.

The best primary mean-score subset was:

```text
CDK2_1Y8Y_aligned
CDK2_3RKB_aligned
CDK2_AF2_MD2NS_C06_F077
```

| Matrix | Aggregation | BEDROC20 | PR-AUC | ROC-AUC | EF5% |
|---|---|---:|---:|---:|---:|
| Primary median | mean score | 0.96531 | 0.75819 | 0.70047 | 2.00 |
| Sensitivity minimum | mean score | 0.96497 | 0.75630 | 0.69938 | 2.00 |

The best two-receptor primary mean-score subset was
`CDK2_1AQ1_reference + CDK2_2C68_aligned`, with BEDROC20 `0.96253`.

These optimistic results show that the receptor pool contains combinations
with strong early enrichment. They do not show that such a combination can be
selected without overfitting or generalized across scaffold folds.

Diagnostic output SHA-256:
`AC83BA245E4638ED38BF0938703EEEEB26D5838EB6F92E8054094C11ED587D1E`.

## Pair-BEDROC QUBO

The new `pair_bedroc_qubo` family retains the normalized singleton BEDROC,
active coverage, active overlap, and receptor redundancy terms. It adds a
quadratic reward for each receptor pair's mean-score ensemble BEDROC on the
training data:

```text
- w_pair * pair_BEDROC(i, j) * x_i * x_j
```

The pair utility weights were preregistered as `0.5`, `1.0`, and `2.0`.
Subset sizes, score aggregations, nested scaffold folds, bootstrap settings,
acceptance thresholds, and all existing baselines were unchanged.

## Nested-CV Result

The pair-BEDROC family selected exactly the same subset as coverage-QUBO in
every outer fold:

| Fold | Selected subset |
|---:|---|
| 0 | `2C68 + 3RKB + C06` |
| 1 | `2C68 + C06 + AF2 parent` |
| 2 | `2C68 + C06` |
| 3 | `1AQ1 + 2C68 + C06` |

Consequently, coverage-QUBO and pair-BEDROC QUBO produced identical OOF
metrics.

| Method | Primary ROC-AUC | Primary PR-AUC | Primary BEDROC20 | Sensitivity BEDROC20 |
|---|---:|---:|---:|---:|
| Single-best | 0.66875 | 0.72201 | 0.89949 | 0.89495 |
| Coverage-QUBO | 0.70469 | 0.74238 | 0.89340 | 0.88397 |
| Pair-BEDROC QUBO | 0.70469 | 0.74238 | 0.89340 | 0.88397 |
| Greedy | 0.62797 | 0.68263 | 0.91791 | 0.91468 |
| All 16 receptors | 0.69313 | 0.73257 | 0.90990 | 0.89975 |

The selected QUBO family remained `coverage_qubo`. The acceptance checks were
unchanged:

- Primary BEDROC20 delta: `-0.00609` - fail.
- Primary ROC-AUC delta: `+0.03594` - pass.
- Primary PR-AUC delta: `+0.02037` - pass.
- BEDROC20 bootstrap CI95 lower bound: `-0.14895` - fail.
- Sensitivity BEDROC20 delta: `-0.01098` - fail.

Gate status: `development_gate_rejected_test_locked`.

## Audit

The independent audit reported:

- OOF rows: 2,240 (2 matrices x 7 methods x 160 ligands).
- Locked-test OOF overlap: 0.
- Locked-test matrix overlap: 0.
- Scaffold groups crossing folds: 0.
- Maximum metric reproduction difference: `1.11e-16`.
- Maximum bootstrap reproduction difference: `1.11e-16`.
- All QUBO solves used the exact cardinality lower-bound certificate.

| Output | SHA-256 |
|---|---|
| `summary.json` | `67A453A00ABEC33BD0430A09C2FBFE2CC12309B34DC51074C03D26465A6CD091` |
| `candidate_protocol.json` | `DBB37926B76BCE8D2D6484B8DAE02093D1D273ADEA0D9A658BEB2A8080182AB9` |
| `method_metrics.csv` | `7C4AC22251FDE2BFFA7B44224964E8F84383EB2BEFC55DFCECBE51A3412BB015` |
| `oof_scores.csv` | `7C198D07BCD01ED9D408C635BA0B8E6C47F62B399DCE4F460FA0414F4DA3BABD` |
| `independent_audit.json` | `D0845D8C8F516145C9B28ADF733BB08B661C5680F8F6F2CE210CA5EA436D283A` |

## Interpretation

The optimistic scan confirms that useful small receptor combinations exist,
but the new pairwise reward did not change nested-CV selection. The pair term
is therefore redundant with the tested singleton utility, overlap, and
redundancy terms at the current weight scale. This negative result should be
retained rather than hidden or replaced by the optimistic full-development
subset.

The next development step should focus on selection stability rather than
adding another reward term. A suitable direction is to quantify receptor
selection frequency across outer and inner folds, then test a stability-aware
QUBO or consensus penalty using the same locked-test boundary. The test split
must remain closed.
