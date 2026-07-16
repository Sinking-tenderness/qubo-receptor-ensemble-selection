# Stage 4 Stability-Aware QUBO Follow-up

## Protocol

The previous stability audit found two receptors selected consistently across
outer folds, while the third receptor changed between folds. This follow-up
tested a `stability_qubo` family with a fixed stability weight of `1.0`.

For each outer fold, the stability term was computed only from the three inner
training contexts:

```text
stability_i = mean(inner_train_BEDROC_i)
               - population_std(inner_train_BEDROC_i)
```

The resulting receptor-level term was min-max normalized and added as a linear
QUBO reward during the outer refit. Outer validation and locked-test scores
were not used to construct the term. The pair-BEDROC, coverage, discriminative,
classical, and single-best baselines were retained unchanged.

## Result

The stability-aware family changed some outer subsets but did not improve
early enrichment:

| Method | Primary ROC-AUC | Primary PR-AUC | Primary BEDROC20 | Sensitivity BEDROC20 |
|---|---:|---:|---:|---:|
| Single-best | 0.66875 | 0.72201 | 0.89949 | 0.89495 |
| Coverage-QUBO | 0.70469 | 0.74238 | 0.89340 | 0.88397 |
| Pair-BEDROC QUBO | 0.70469 | 0.74238 | 0.89340 | 0.88397 |
| Stability-QUBO | 0.68125 | 0.72304 | 0.89204 | 0.88069 |
| Greedy | 0.62797 | 0.68263 | 0.91791 | 0.91468 |
| All 16 receptors | 0.69313 | 0.73257 | 0.90990 | 0.89975 |

Outer stability-QUBO subsets:

| Fold | Subset |
|---:|---|
| 0 | `1AQ1 + 3RKB + C06` |
| 1 | `2C68 + C06 + AF2 parent` |
| 2 | `2C68 + C06` |
| 3 | `1AQ1 + 2C68 + C06` |

The stability-aware outer pairwise Jaccard was `0.46389`, lower than the
coverage/pair-BEDROC value of `0.58333`. Thus the linear stability reward did
not stabilize the third receptor and reduced screening performance slightly.

Gate status remains:

```text
development_gate_rejected_test_locked
```

The selected family for the preregistered gate remained coverage-QUBO because
it had the highest development OOF BEDROC among the configured QUBO families.
The 40-ligand test split was not evaluated.

## Audit

- Development-only OOF and matrix test overlap: `0`.
- Independent metric reproduction difference: `1.11e-16`.
- Independent bootstrap reproduction difference: `1.11e-16`.
- Scaffold groups crossing folds: `0`.
- Full regression tests: `134 passed, 1 skipped`.

Results:
`results/runs/stage04_cdk2_expanded16_stability_development_scaffold_cv_gate/`

Stability output:
`results/runs/stage04_cdk2_expanded16_stability_development_scaffold_cv_gate/selection_stability.json`

| Output | SHA-256 |
|---|---|
| `summary.json` | `2F7A2679FFDAB630EA9A2959A80822696A0E7FA2E2FBB88D9A3B393DB8D8E625` |
| `candidate_protocol.json` | `A4F0C3DF5E264B0031558105F48DB1492A637921892BA66F46C516BABE054A5B` |
| `method_metrics.csv` | `CD6E3F1E12E2D97D2BD50BCCDB344A6FE321B3414906E1B16414EE7DF4F97790` |
| `oof_scores.csv` | `52C01C938FBB792D5449946ECDA3165C7022FA1FD40B15B05CFE7B96B3869DF7` |
| `outer_fold_results.csv` | `72503D55CBB36695A42BAC9DDA4F3DCD4AED101B7D42C1F11172E56737FD6BD6` |
| `independent_audit.json` | `CD6AFDAF4C4524F3C0207CBEF258EDD2E456ABC9FDCDE78E7E31744CF68EB04C` |
| `selection_stability.json` | `D78BDE691FDCD5F369857C27942FB96F73D5D29736BFC1D2263E82D8B8A5C9FD` |

## Interpretation

The current evidence argues against treating stability as another additive
QUBO reward. The stable core is useful as a diagnostic descriptor, but a
linear reward based on inner BEDROC consistency can conflict with the actual
early-enrichment objective and can even lower subset stability.

The next protocol should treat stability as a constraint or release rule: for
example, require a receptor to exceed a preregistered inner/outer frequency
threshold, then optimize the remaining budget using the original objective.
That approach should be tested as a separate consensus-selection method on the
development set, with no locked-test access.
