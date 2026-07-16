# Stage 4 Expanded-16 Development Scaffold-CV Gate Record

## Status

- Gate status: `development_gate_rejected_test_locked`
- Run date: 2026-07-16
- Target: CDK2
- Receptor pool: 16 receptor conformers
- Development ligands: 160 (80 active, 80 decoy)
- Locked test ligands: 40 (20 active, 20 decoy)
- Test scores evaluated: no

The gate was evaluated only on the 160 development ligands. The 40-ligand
test split remains unavailable to fitting, tuning, ranking, metric calculation,
and bootstrap operations.

## Inputs

The merged matrices contain 16 x 160 = 2,560 development score cells. The
primary matrix uses the across-seed median score. The sensitivity matrix uses
the across-seed minimum score. No matrix cell was selectively replaced.

| Input | SHA-256 |
|---|---|
| `stage04_cdk2_expanded16_development_median_score_matrix.csv` | `CAB398C59AC49088ABE6065A58E216D46DC205BE862F0D0A8E57050CC45D96E4` |
| `stage04_cdk2_expanded16_development_minimum_score_matrix.csv` | `EF4B68E764ADFA976AA365D04E8D97715D30F8208135C33E43D3AAB15B6EDED5` |
| `stage04_cdk2_expanded16_development_seed_stability_warnings.csv` | `33D6BD1B6CCFD083AF0BF2DED3212138BB9BDCECF6C0D20D5E8E3324A7AF366D` |
| `stage04_cdk2_expanded16_development_matrix_summary.json` | `83FC0634CBB53C158ED93D116FCD79225DB2F061B2E41C54DC6AF9FF3287AF2F` |
| `dude_cdk2_benchmark_100a_100d_scaffold_split_manifest.csv` | `2F3D68B16856EB8C9310E0A2D15283B95CBBA4D6E3A83E548385371C6440C167` |
| `dude_cdk2_benchmark_100a_100d_scaffold_split_summary.json` | `4DFA7626F47D2B6450E670C2903A2A69ECAC3B894DBD48D3B3AA7FBE97659DAE` |

The merged input contains 72 development warning pairs: 55 from the original
train split and 17 from the original validation split. The 12 warning rows
belonging to the locked test split were excluded because the development
matrices do not contain those scores.

## Protocol

- Four scaffold-disjoint outer folds, with 20 active and 20 decoy ligands per fold.
- Inner selection uses only the outer training portion and the remaining development folds.
- Selection metric: `bedroc_alpha_20`, with PR-AUC and ROC-AUC tie-breakers.
- Score normalization: train-only min-max normalization.
- Candidate subset sizes: 1, 2, and 3 receptors.
- Aggregations: `min_score` and `mean_score` when the subset has at least two receptors.
- Compared families: single-best, exhaustive, greedy, all-receptors, coverage-QUBO, and discriminative-QUBO.
- QUBO weights were selected from the preregistered grids; size penalty was 20.0.
- Bootstrap: 5,000 paired resamples, seed `20261003`.

Exact small-pool QUBO solving remains exact. For the 16-receptor pool, the
solver enumerates the requested cardinality slice and uses a conservative
global lower-bound certificate; it falls back to all 65,536 states when the
certificate is inconclusive. The audit found only certificate-backed solves,
with 120 or 560 requested-cardinality states evaluated per solve.

Ranking ties are resolved deterministically by `(score, ligand_id)`. This is
important because sensitivity aggregation produced tied scores; without an
explicit tie rule, a CSV round trip could change average precision.

## Development OOF Results

Metrics below are computed from 160 out-of-fold predictions per method. The
primary matrix is shown first; the minimum-score sensitivity matrix follows.

| Method | Primary ROC-AUC | Primary PR-AUC | Primary BEDROC20 | Sensitivity ROC-AUC | Sensitivity PR-AUC | Sensitivity BEDROC20 |
|---|---:|---:|---:|---:|---:|---:|
| Single-best | 0.66875 | 0.72201 | 0.89949 | 0.67063 | 0.72147 | 0.89495 |
| Exhaustive | 0.64805 | 0.69192 | 0.87236 | 0.65086 | 0.69374 | 0.87034 |
| Greedy | 0.62797 | 0.68263 | 0.91791 | 0.63195 | 0.68431 | 0.91468 |
| All 16 receptors | 0.69313 | 0.73257 | 0.90990 | 0.69391 | 0.72990 | 0.89975 |
| Coverage-QUBO | 0.70469 | 0.74238 | 0.89340 | 0.70344 | 0.73562 | 0.88397 |
| Discriminative-QUBO | 0.70563 | 0.72143 | 0.82607 | 0.70359 | 0.71605 | 0.82161 |

The selected development-only candidate is:

```text
CDK2_2C68_aligned
CDK2_AF2_MD2NS_C05_F074
CDK2_AF_P24941_F1_v6
```

This is a candidate nomination for later review, not a released final subset.

## Acceptance Checks

The preregistered comparison is selected QUBO family versus single-best.

| Check | Observed | Threshold | Result |
|---|---:|---:|---|
| Primary BEDROC20 delta | -0.00609 | >= 0.02000 | FAIL |
| Primary ROC-AUC delta | +0.03594 | >= 0.00000 | PASS |
| Primary PR-AUC delta | +0.02037 | >= 0.00000 | PASS |
| Primary BEDROC20 bootstrap CI95 lower bound | -0.14895 | >= 0.00000 | FAIL |
| Sensitivity BEDROC20 delta | -0.01098 | >= 0.00000 | FAIL |

The paired bootstrap for coverage-QUBO minus single-best produced:

| Metric | Mean delta | 95% interval |
|---|---:|---:|
| ROC-AUC | +0.03578 | [-0.00674, +0.07911] |
| PR-AUC | +0.02106 | [-0.04225, +0.08393] |
| BEDROC20 | -0.00163 | [-0.14895, +0.13751] |

Therefore, this run does not support the claim that the current QUBO
objective improves early enrichment. It does show a possible improvement in
global ranking metrics, but that signal is not aligned with the preregistered
early-enrichment gate and is not yet stable under resampling.

## Independent Audit

`scripts/audit_development_scaffold_cv_gate.py` independently rebuilt the
metrics from `oof_scores.csv` and reproduced the paired bootstrap without
importing the gate implementation.

- OOF rows: 1,920 (2 matrices x 6 methods x 160 ligands).
- Every OOF group contains exactly the 160 development ligands.
- OOF overlap with locked test: 0.
- Matrix overlap with locked test: 0 for both matrices.
- Scaffold groups crossing folds: 0.
- Maximum metric reproduction difference: `1.11e-16`.
- Maximum bootstrap reproduction difference: `1.11e-16`.
- All-receptor metadata records target size 16.

Audit output:
`results/runs/stage04_cdk2_expanded16_development_scaffold_cv_gate/independent_audit.json`

## Interpretation and Next Step

This is a valid negative development result, not evidence that QUBO is
useless. The current data show that selecting a small receptor combination can
improve ROC-AUC and PR-AUC while reducing the early enrichment signal that the
project prioritizes. The discriminative-QUBO family is especially unfavorable
for BEDROC20 in this run, and greedy/all-receptor baselines also demonstrate
that a higher-complexity subset is not automatically better.

The locked test must remain closed. The next development experiment should
diagnose objective alignment and subset redundancy using only the 160
development ligands: compare a BEDROC-oriented QUBO utility against the
current coverage utility, inspect the selected receptors' active-top-set
overlap and warning strata, and preregister any revised weights or acceptance
rule before another gate run. Only a passing, manually reviewed development
protocol may proceed to the 40-ligand test evaluation.

## Output Hashes

| Output | SHA-256 |
|---|---|
| `summary.json` | `E50768C1514943AF974C2CDC2079B87C4C17B609306B63C75FD1DD9C3EB50678` |
| `candidate_protocol.json` | `1BFAEF344274386296B37B953B40E26A9E379AE0BDBEBA3CE9197B055F2BDB85` |
| `method_metrics.csv` | `CCB426EED896D8E4F578C081DCD9C4C7E096948BF5015E4337A2C146057F58FD` |
| `oof_scores.csv` | `86B3AC90FC1009525D308A0DE7FDD9DBB09E34D429D6CF262BC4D20AE4DA03ED` |
| `independent_audit.json` | `8F4163BE7F0156569D656A5A944CD7222C8CD41725BB72C4DA199763DFA641FE` |
