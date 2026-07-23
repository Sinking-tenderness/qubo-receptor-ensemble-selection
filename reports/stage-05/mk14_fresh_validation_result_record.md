# MAPK14 Fresh Validation Result Record

## Decision

- Status: `fresh_validation_passed_test_locked`
- Authorization: `stage05-mk14-fresh-validation-after-pair-synergy-gate-20260721-v1`
- Fresh validation panel: 75 actives and 1,501 decoys (1,576 ligands)
- Receptors: 5
- Docking evidence: 3 seeds x 7,880 receptor-ligand pairs = 23,640 successful jobs
- Failed receptor-ligand pairs: 0
- Locked test rows or scores read: 0
- All preregistered acceptance checks: passed

The result supports fresh within-target validation of the frozen three-receptor
pair-synergy subset. The locked test remains unreleased.

## Evidence Admission

| Seed | Base seed | Successful pairs | Search warnings | Archive SHA-256 |
|---|---:|---:|---:|---|
| seed0 | 20260801 | 7,880 / 7,880 | 5 | `E0477F148BC0460A0DD2E84ADE6538392194B835925A46B290E3BE66EE7F216A` |
| seed1 | 20260802 | 7,880 / 7,880 | 2 | `7D0305E073728F3588BC90B29C8AA2CC716800BD8B0ACD82B9F1904F7DC4538F` |
| seed2 | 20260803 | 7,880 / 7,880 | 2 | `C08CF53A8477881D3E9965BF5160807CAF99593CDFC27D308662414C9BB179ED` |

The three ligand manifests, pair-key sets, and label maps were identical. All
hashes recorded inside each seed summary matched the returned files.

The nine warning cells were all decoys and no warning recurred for the same
receptor-ligand pair in another seed. For every warning cell, the other two
seeds produced mutually consistent ordinary scores. Seven warnings involved
`MK14_2BAJ_aligned`, one involved `MK14_3K3J_aligned`, and one involved
`MK14_3KQ7_aligned`. They were retained without score replacement.

Across all 7,880 paired cells, the median three-seed score range was 0.042
kcal/mol. There were 411 ranges above 0.5, 134 above 1.0, 19 above 2.0, and 9
above 5.0 kcal/mol. Pairwise Spearman correlations were 0.979-0.981. After
excluding the nine declared warning cells for diagnosis only, pairwise Pearson
correlations were 0.983-0.984. The preregistered median aggregation therefore
suppressed isolated search failures without deleting evidence.

## Primary Results

All methods below were frozen before fresh-validation scores were evaluated.
Lower normalized docking score ranked earlier; the primary matrix used the
median representative score across three seeds.

| Frozen method | BEDROC (alpha=20) | ROC-AUC | PR-AUC | EF1% | EF5% | EF10% | Actives in top 10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pair-synergy QUBO | 0.5509 | 0.8360 | 0.4265 | 18.39 | 8.78 | 5.32 | 9 |
| Nested greedy | 0.5509 | 0.8360 | 0.4265 | 18.39 | 8.78 | 5.32 | 9 |
| Nested exhaustive | 0.5256 | 0.8323 | 0.3996 | 17.07 | 8.25 | 5.05 | 9 |
| Matched linear top-k | 0.4585 | 0.8167 | 0.2700 | 7.88 | 7.45 | 5.05 | 4 |
| Single best receptor | 0.3760 | 0.7258 | 0.2632 | 13.13 | 5.59 | 3.46 | 8 |

The pair-synergy QUBO subset was:

- `MK14_2BAJ_aligned`
- `MK14_2QD9_reference`
- `MK14_3KQ7_aligned`

Its primary BEDROC deltas were +0.0924 versus matched linear top-k, +0.0252
versus nested exhaustive, and +0.1748 versus the single-receptor baseline.

## Robustness

| Method | seed0 | seed1 | seed2 | Mean seed BEDROC | Worst seed BEDROC | Sensitivity BEDROC |
|---|---:|---:|---:|---:|---:|---:|
| Pair-synergy QUBO | 0.4872 | 0.5497 | 0.5409 | 0.5260 | 0.4872 | 0.5580 |
| Nested exhaustive | 0.4773 | 0.5286 | 0.5137 | 0.5065 | 0.4773 | 0.5313 |
| Matched linear top-k | 0.4481 | 0.4548 | 0.4544 | 0.4524 | 0.4481 | 0.4680 |
| Single best receptor | 0.3609 | 0.3772 | 0.3766 | 0.3716 | 0.3609 | 0.3758 |

The QUBO-minus-linear BEDROC delta remained positive for the mean seed
(+0.0735) and worst seed (+0.0391). The QUBO-minus-exhaustive delta also
remained positive for the mean seed (+0.0194) and worst seed (+0.0099).

The preregistered 5,000-replicate split-group block bootstrap gave:

| Comparison | Mean BEDROC delta | 95% interval |
|---|---:|---:|
| QUBO - matched linear top-k | +0.0924 | [0.0010, 0.1858] |
| QUBO - nested exhaustive | +0.0248 | [0.0043, 0.0510] |

Both lower bounds were positive and passed their frozen thresholds.

## Interpretation Boundary

This is evidence that the frozen pair-synergy receptor subset generalizes to a
new MAPK14 validation panel and improves early enrichment over the specified
matched linear, exhaustive, and single-receptor comparators.

It is not evidence that QUBO outperforms greedy selection: the final QUBO and
greedy subsets are identical, so their predictions and metrics are identical.
It also does not establish quantum computational advantage, test performance,
cross-target generalization, binding affinity, or biological activity.

## Recorded Outputs

- Validation result: `data/stage05_mk14_fresh_validation_result.json`
- Validation result SHA-256: `8ACD7DCB63EFDA097DE22FE281E24F0BF156D0176FFABAD1D9E4FD2844C59BA0`
- Aggregation summary: `results/runs/stage05_mk14_fresh_validation_e32_aggregated/summary.json`
- Aggregation summary SHA-256: `143F798F95181C24D3A38221DD1897784F1A816103E75C00F0290C9019AFC3A2`
- Primary matrix SHA-256: `059304514B531DA92E20D376733146DC357CDC791227819BD04BC7CAE7E61179`
- Sensitivity matrix SHA-256: `15D493237F0F8799FE54F0B6447A70E1639EB881CB73320628BCB8DEC34E8D0E`

## Next Gate

The primary result must remain frozen and the test split must remain locked.
The preregistered EnOpt-style XGBoost models may now be evaluated once as a
supplementary baseline; they cannot change this primary decision. A separate
authorization is required before any test release.

To support a QUBO-over-greedy claim, a later experiment must prospectively
freeze a larger or more constrained receptor search in which QUBO and greedy
select different subsets, then compare them on untouched data. Solver runtime
or quantum advantage requires a separate scaling study and is not implied by
the present screening result.
