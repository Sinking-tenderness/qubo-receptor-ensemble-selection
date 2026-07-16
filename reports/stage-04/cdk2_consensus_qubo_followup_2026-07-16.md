# Stage 4 Consensus-Constrained QUBO Follow-up

## Protocol

The previous stability-aware linear reward did not improve either receptor
selection stability or early enrichment. This follow-up therefore tested
stability as a hard constraint rather than another additive QUBO term.

Within each outer scaffold-CV fold:

1. Coverage-QUBO was tuned using only the three inner train/validation
   contexts.
2. Receptors selected in at least two of the three inner fitted subsets were
   designated as consensus receptors.
3. Consensus receptors were required during the outer refit.
4. Only receptor budgets of two or three were allowed. Candidate budgets
   smaller than the required consensus set were excluded as infeasible.
5. The remaining budget, when available, was optimized with the original
   coverage-QUBO objective.

No outer-validation score was used to define the consensus set. The 40 locked
test ligands remained absent from every matrix, fit, tuning operation, metric,
and bootstrap calculation.

## Result

The consensus-constrained family did not improve development OOF performance:

| Method | Primary ROC-AUC | Primary PR-AUC | Primary BEDROC20 | Sensitivity BEDROC20 |
|---|---:|---:|---:|---:|
| Single-best | 0.66875 | 0.72201 | 0.89949 | 0.89495 |
| Coverage-QUBO | 0.70469 | 0.74238 | 0.89340 | 0.88397 |
| Consensus-QUBO | 0.67656 | 0.71094 | 0.88260 | 0.86809 |
| Greedy | 0.62797 | 0.68263 | 0.91791 | 0.91468 |
| All 16 receptors | 0.69313 | 0.73257 | 0.90990 | 0.89975 |

Relative to single-best, consensus-QUBO changed primary BEDROC20 by
`-0.01689`, primary ROC-AUC by `+0.00781`, primary PR-AUC by `-0.01107`, and
sensitivity BEDROC20 by `-0.02686`.

Outer consensus subsets and their independently derived constraints were:

| Fold | Required consensus receptors | Outer subset |
|---:|---|---|
| 0 | `3RKB + C06` | `3RKB + C06` |
| 1 | `2C68 + C06 + AF2 parent` | `2C68 + C06 + AF2 parent` |
| 2 | `1AQ1 + 2C68` | `1AQ1 + 2C68` |
| 3 | `1AQ1 + 2C68 + AF2 parent` | `1AQ1 + 2C68 + AF2 parent` |

In all four outer folds, the required consensus set consumed the entire
selected receptor budget. The QUBO therefore had no residual slot in which to
choose an additional complementary receptor. This protocol behaved as a hard
consensus selector rather than a consensus-core-plus-QUBO selector.

Consensus outer pairwise Jaccard was `0.27778`, below the coverage-QUBO value
of `0.58333`. The consensus stability audit retained only `2C68` under the
predefined outer-frequency and inner-frequency rule. Hard inner consensus did
not produce a globally stable outer subset.

Gate status remains:

```text
development_gate_rejected_test_locked
```

Coverage-QUBO remained the highest-BEDROC configured QUBO family and was
therefore retained as the selected development family. Its full-development
refit subset remained `2C68 + C05 + AF2 parent`. The locked test split was not
evaluated.

## Audit

- Development-only OOF and matrix locked-test overlap: `0`.
- Scaffold groups crossing folds: `0`.
- Independent metric reproduction difference: `1.11e-16`.
- Independent bootstrap reproduction difference: `1.11e-16`.
- Consensus frequency, config, budget, and subset constraints: independently
  reproduced and passed.
- Full regression tests: `142 passed, 1 skipped`.

Results:
`results/runs/stage04_cdk2_expanded16_consensus_development_scaffold_cv_gate/`

| Output | SHA-256 |
|---|---|
| `summary.json` | `1B5F9C9A5351BA2EA1BE44B0D28B36BA514E5BCC52C037527E11A6BF53999921` |
| `candidate_protocol.json` | `79B41502828E10C4AC68A420FC0C2295618241D9B674FD78672D79D37D346364` |
| `method_metrics.csv` | `441A63318607F74D9ACFC05EC0DADC835C851E08D92BA7775232ED2F3F12CE73` |
| `oof_scores.csv` | `89F00179650E020731B274E35818AF9B560DAA93D77A6072596659AC644C03E6` |
| `outer_fold_results.csv` | `2793358B316CA5A41A241EC2ABB06F10775A172AB9D2F24BE98A9F29514619ED` |
| `independent_audit.json` | `E19EEFACDAAF46D49DB87BAF2407BE663650C6A9ECF9DF978F40843FAC3FDA13` |
| `selection_stability.json` | `03B0411399593E8610AEC39ED1EE4269F03DC125E220407AEC81EA87996100B3` |

## Interpretation

The negative result does not show that stable receptors are unimportant. It
shows that a two-of-three hard threshold can overconstrain a receptor budget
of only two or three, while the identity of the inner consensus set can still
change across outer folds.

The next development experiment should preserve the stable core but reserve a
preregistered residual slot for complementarity. One defensible design is a
fixed core budget plus one QUBO-selected receptor, with feasibility and
release rules defined before rerunning nested scaffold CV. This remains a
development-only hypothesis and does not justify opening the locked test set.
