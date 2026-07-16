# Stage 4 Core-Plus-One Repeated Scaffold-CV Robustness

## Purpose

The single four-fold scaffold-CV run found a positive but sub-threshold
core-plus-one BEDROC20 delta of `+0.01429`. This follow-up tested whether that
result depended on one particular scaffold assignment.

Five fold seeds were preregistered:

```text
20261002, 20261003, 20261004, 20261005, 20261006
```

Each repeat used the same 160 development ligands, the same 16-receptor score
matrices, the same core-plus-one protocol, and a different deterministic
scaffold fold assignment. The 40 locked test ligands remained unavailable.

## Feasibility Result

Only the original seed `20261002` completed successfully. The other four
seeds failed at the preregistered core construction step because at least one
outer fold had fewer than two receptors meeting the two-of-three inner
frequency threshold.

| Fold seed | Result |
|---:|---|
| `20261002` | feasible; gate completed and audit passed |
| `20261003` | infeasible: fewer than two qualified core receptors |
| `20261004` | infeasible: fewer than two qualified core receptors |
| `20261005` | infeasible: fewer than two qualified core receptors |
| `20261006` | infeasible: fewer than two qualified core receptors |

Feasibility was therefore `1/5 = 20%`. The runner retained each failed
derived config, status file, stdout log, stderr log, and error tail. No fallback
threshold or post-hoc receptor substitution was applied.

## Successful Repeat

The only feasible repeat reproduced the original result:

| Metric | Core-plus-one minus single-best |
|---|---:|
| Primary BEDROC20 | `+0.01429` |
| Primary ROC-AUC | `+0.02109` |
| Primary PR-AUC | `+0.01149` |
| Sensitivity BEDROC20 | `+0.00942` |

Because four of five repeats were infeasible, the averaged OOF metrics are
not a five-repeat validation estimate. The aggregate primary BEDROC20
bootstrap comparison had a 95% lower bound of `-0.11874`, and cannot support a
reliable improvement claim.

The repeated protocol status is:

```text
ok_with_infeasible_repeats
```

The locked test split was not evaluated.

## Audit

- Successful repeat independent audits: all checks passed.
- Successful repeat locked-test OOF overlap: `0`.
- Successful repeat scaffold-crossing count: `0`.
- Successful repeat metric reproduction difference: `1.11e-16`.
- Successful repeat bootstrap reproduction difference: `1.08e-16`.
- Full regression tests after the runner changes: `147 passed, 1 skipped`.

Results:
`results/runs/stage04_cdk2_core_plus_one_repeated_scaffold_cv/`

| Output | SHA-256 |
|---|---|
| `summary.json` | `9F32D4A3008384495C6A56F4479334F085EACFFC9C98001E67DCA6EC271040EA` |
| `repeat_metrics.csv` | `5334893F19CB8A7454337BC15FE31D3B678B7D529A536B5E623D0046C1AD00A9` |
| `outer_selections.csv` | `B9CC4F8EEA697371C20A23E4CC6AD9E8533336F8BB80313FDC92A9D7FAF8CE22` |
| `aggregate_oof_scores.csv` | `FDE1280FC38B905179D0624867F77CD46A8A1FFFB9EE83B504B35BFB80027755` |

## Interpretation

The strict two-core plus one-residual protocol is not a robust MVP: its
feasibility is only 20% across the preregistered scaffold partitions. The
positive single-seed result must therefore be treated as a conditional result,
not a general improvement.

The next candidate should change the protocol definition before another run,
for example one frequency-qualified hard core receptor plus two QUBO-selected
residual slots, or a top-two core with an explicit confidence flag rather than
a hard two-of-three requirement. That alternative must be separately
preregistered and evaluated only on development data. The locked test set
should remain closed until a protocol is both feasible across repeats and
meets the acceptance criteria.
