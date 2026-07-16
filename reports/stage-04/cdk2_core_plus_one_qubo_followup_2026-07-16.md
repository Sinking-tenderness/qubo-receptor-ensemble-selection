# Stage 4 Core-Plus-One QUBO Follow-up

## Protocol

The previous consensus-constrained QUBO could consume the entire receptor
budget with hard consensus receptors. This follow-up fixed the budget at three
receptors and reserved one residual slot for the original coverage-QUBO
objective.

Within each outer scaffold-CV fold:

1. Coverage-QUBO was tuned using only the three inner contexts.
2. Receptors meeting the preregistered inner frequency threshold of two out of
   three selections were identified.
3. The two highest-frequency qualified receptors formed a hard consensus core.
   Frequency ties were resolved by receptor ID for deterministic behavior.
4. The core occupied two positions. Exactly one additional receptor was then
   selected by the original coverage-QUBO objective.
5. The outer validation fold was used only after the core and residual receptor
   had been fitted.

The 40 locked test ligands remained absent from every fit, tuning operation,
ranking, metric, and bootstrap calculation.

## Result

Core-plus-one improved the development OOF BEDROC20 over both the single-best
and unconstrained coverage-QUBO baselines, but did not pass the preregistered
release gate:

| Method | Primary ROC-AUC | Primary PR-AUC | Primary BEDROC20 | Sensitivity BEDROC20 |
|---|---:|---:|---:|---:|
| Single-best | 0.66875 | 0.72201 | 0.89949 | 0.89495 |
| Coverage-QUBO | 0.70469 | 0.74238 | 0.89340 | 0.88397 |
| Core-plus-one QUBO | 0.68984 | 0.73350 | 0.91378 | 0.90437 |
| Greedy | 0.62797 | 0.68263 | 0.91791 | 0.91468 |
| All 16 receptors | 0.69313 | 0.73257 | 0.90990 | 0.89975 |

Relative to single-best, core-plus-one changed primary BEDROC20 by `+0.01429`,
primary ROC-AUC by `+0.02109`, primary PR-AUC by `+0.01149`, and sensitivity
BEDROC20 by `+0.00942`. The preregistered primary BEDROC20 improvement
threshold was `+0.02`, so this result remains below the acceptance criterion.

The paired bootstrap BEDROC20 delta had mean `+0.01794` with a 95% interval
of `[-0.11798, 0.14932]`. The lower bound is below zero and therefore does not
support a reliable improvement claim.

## Outer-Fold Selections

| Fold | Two-receptor core | Residual receptor | Outer subset |
|---:|---|---|---|
| 0 | `3RKB + C06` | `1AQ1` | `1AQ1 + 3RKB + C06` |
| 1 | `2C68 + C06` | `AF2 parent` | `2C68 + C06 + AF2 parent` |
| 2 | `1AQ1 + 2C68` | `C06` | `1AQ1 + 2C68 + C06` |
| 3 | `1AQ1 + 2C68` | `C06` | `1AQ1 + 2C68 + C06` |

The outer subset Jaccard was `0.53333`, lower than the unconstrained
coverage-QUBO value of `0.58333`, but substantially higher than the previous
full consensus-constrained value of `0.27778`. The stability audit identified
`1AQ1 + 2C68 + C06` as the stable outer subset under the existing frequency
definition, although only four outer folds are available.

The full-development refit selected:

```text
core: 2C68 + C06
residual: C05
final subset: 2C68 + C05 + C06
```

This is a development candidate only. It has not been evaluated on the locked
test split.

## Gate and Audit

Gate status:

```text
development_gate_rejected_test_locked
```

- Development-only OOF and matrix locked-test overlap: `0`.
- Scaffold groups crossing folds: `0`.
- Independent metric reproduction difference: `1.11e-16`.
- Independent bootstrap reproduction difference: `1.08e-16`.
- Core frequency, reference subset, budget, and selected-subset constraints:
  independently reproduced and passed.
- Full regression tests: `144 passed, 1 skipped`.

Results:
`results/runs/stage04_cdk2_expanded16_core_plus_one_development_scaffold_cv_gate/`

| Output | SHA-256 |
|---|---|
| `summary.json` | `3AC82B777D0D7699CB0A5DAC91BEF50C73D8B249FA051C1752374C469CB2F6F6` |
| `candidate_protocol.json` | `59E2295A6950ADD6D9DE4C38C30E5FBD3817F9BE0915EDEAF465F071EB0C17BC` |
| `method_metrics.csv` | `FF0C0FD14E08C1D21AF76A9138CA1C47C53C24F4F7B6CC3C775EDABEBD7D10DD` |
| `oof_scores.csv` | `9F244C893CF901D1D05447B7DA7255778B98A00DF904CAD107F5EE58FA5C61C4` |
| `outer_fold_results.csv` | `F53448DD5C4B186385893135A33344AD53442D5CC326D4C8506387C9897CEE79` |
| `independent_audit.json` | `8EF8B9617239C406F382C10EA225AFE7B9CBC053BF87C8513BD3F0F34947C0FD` |
| `selection_stability.json` | `034790F7292FBAF13EFE9A70D2B519260F8818A5AAE750AA0D2FACA537279D96` |

## Interpretation

The residual-slot design is the strongest development result among the tested
QUBO variants so far, but it is not yet a validated improvement. The positive
BEDROC delta is smaller than the preregistered threshold and its bootstrap
interval is wide because the scaffold-CV development set still provides only
four outer folds.

The result supports continuing with this formulation as the current MVP
candidate, while keeping the locked test split closed. The next scientific
step is not to tune the acceptance threshold after seeing this result; it is
to document the core-plus-one protocol as the candidate method and decide,
before any test release, whether an independent development replicate or a
larger scaffold-disjoint benchmark is required.
