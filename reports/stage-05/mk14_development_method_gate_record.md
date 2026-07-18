# Stage 5 MAPK14 Development Method Gate

## Decision

- Status: `development_gate_failed_test_locked`
- Train rows: 80 active + 80 decoy
- Validation rows: 40 active + 40 decoy
- Test rows read: 0
- Selected QUBO family: `coverage_qubo`
- QUBO promotion: rejected
- Test release: prohibited

The project-environment execution selected exactly the same single receptor for
single-best, exhaustive, greedy, coverage-QUBO, and discriminative-QUBO. The
QUBO candidate therefore had zero validation improvement over single-best and
failed the preregistered `+0.02` BEDROC20 requirement.

## Runtime Lock

The formal execution environment was locked before the canonical result was
regenerated:

| Component | Version |
|---|---|
| Conda environment | `qubo-receptor-ensemble` |
| Python | `3.11.15` |
| NumPy | `1.26.4` |
| SciPy | `1.17.1` |

An initial preflight was accidentally run with base Python 3.12.4. Python's
seeded shuffle produced a different fold assignment across interpreter
versions, so that preflight was invalidated rather than chosen or averaged. It
did not alter the preregistered candidates, thresholds, tie breakers, or locked
test boundary. The runner now rejects a mismatched runtime before reading any
score metric. All results below are from the locked project environment.

## Admitted Inputs

The unchanged e16 median and minimum matrices were admitted through the
predefined e32 rescue path. The two raw-range failures converged at e32, no
matrix cell was replaced, and the rescue audit authorized both original e16
matrices.

| Input | SHA-256 |
|---|---|
| Preregistration | `1F2D81BE000A13F4B1E0EA867299073663C784B3BE1E41BBA52AAABCCB1CCD23` |
| Primary median matrix | `921348DF346C0104438C245CD116A51222850E7E801E8E27884B7968E9786612` |
| Sensitivity minimum matrix | `BEC4A92BCF439BD3558B7EC045AA84E72275B4765D2AED469F93E0750B20DD7E` |
| Matrix-rescue audit | `51283A625594618078E81AEC9E239D64DF684FD05F1BE24BF14FF673FB492731` |

Input auditing confirmed 240 development ligands, four receptors, no
nonnegative docking scores, no train-validation ligand overlap, no split-group
or scaffold overlap, and no test row.

## Train-Only Selection

Four balanced scaffold/group folds were constructed from the 160 train rows.
Every fold contained 20 actives and 20 decoys. Per-receptor min-max bounds were
fit only on each inner-training partition.

The frozen search evaluated 457 configurations:

| Method | Candidate count | Final subset | Aggregation | Mean inner BEDROC20 |
|---|---:|---|---|---:|
| Single-best | 1 | 3KQ7 | minimum | 0.87664 |
| Exhaustive | 5 | 3KQ7 | minimum | 0.87664 |
| Greedy | 5 | 3KQ7 | minimum | 0.87664 |
| All receptors | 2 | 2QD9 + 1A9U + 3K3J + 3KQ7 | mean | 0.85262 |
| Coverage QUBO | 111 | 3KQ7 | minimum | 0.87664 |
| Discriminative QUBO | 333 | 3KQ7 | minimum | 0.87664 |

Coverage QUBO won the train-only family tie according to the frozen ordering.
No validation metric was used to select its family, weights, subset size,
aggregation, or receptor subset.

## Validation Results

### Primary Median Matrix

| Method | ROC-AUC | PR-AUC | BEDROC20 | EF5% | Top-10 active |
|---|---:|---:|---:|---:|---:|
| Single-best / exhaustive / greedy | 0.68781 | 0.74609 | 0.92418 | 2.0 | 9 |
| Coverage / discriminative QUBO | 0.68781 | 0.74609 | 0.92418 | 2.0 | 9 |
| All receptors, mean | 0.77000 | 0.81348 | 0.98270 | 2.0 | 10 |

### Sensitivity Minimum Matrix

| Method | ROC-AUC | PR-AUC | BEDROC20 | EF5% | Top-10 active |
|---|---:|---:|---:|---:|---:|
| Single-best / exhaustive / greedy | 0.68938 | 0.74636 | 0.92388 | 2.0 | 9 |
| Coverage / discriminative QUBO | 0.68938 | 0.74636 | 0.92388 | 2.0 | 9 |
| All receptors, mean | 0.77750 | 0.82198 | 0.98507 | 2.0 | 10 |

EF1% and EF5% are saturated at their maximum value of 2.0 for several methods
because the validation set is balanced. They do not distinguish these methods.

## Promotion Checks

| Required check | Observed | Threshold | Result |
|---|---:|---:|---|
| Primary BEDROC20 delta | 0.00000 | at least +0.02000 | **FAIL** |
| Primary ROC-AUC delta | 0.00000 | at least 0 | PASS |
| Primary PR-AUC delta | 0.00000 | at least 0 | PASS |
| Sensitivity BEDROC20 delta | 0.00000 | at least 0 | PASS |
| BEDROC20 bootstrap CI95 low | 0.00000 | at least 0 | PASS |

The paired-bootstrap deltas and intervals were exactly zero because the selected
QUBO and single-best produced identical validation score vectors. Since all
checks were required, the failed primary-improvement check rejects promotion.

## QUBO-Specific Finding

The selected coverage configuration was:

```text
target_size = 1
aggregation = min_score
active_coverage = 0
active_overlap = 0
redundancy = 0
decoy_exposure = 0
```

For target size one, there is no receptor-pair decision. The model reduces to
the normalized singleton BEDROC utility plus a one-receptor cardinality
constraint and selects 3KQ7, exactly as single-best does. The selected
discriminative configuration used decoy exposure 0.5 but also selected the same
single receptor.

This is a scientifically useful negative result: the current four-receptor
pool and frozen objective do not produce a complementary multi-receptor QUBO
solution. The stronger all-receptor mean result is descriptive evidence that
ensembling may help, but validation cannot be used post hoc to replace the
train-selected protocol.

The exact random-subset tables contain all 48 matrix/subset/aggregation rows and
10 distribution summaries. They remain descriptive baselines; none was used to
reselect a receptor subset after validation was observed.

## Independent Audit

The separate project-environment audit reproduced:

- all 12 matrix-method validation metric groups;
- 80 validation rows with 40 actives and 40 decoys per group;
- all 5,000 paired-bootstrap iterations;
- all 48 exact fixed-subset metric rows;
- all 10 exact random-subset distribution summaries;
- the failed gate decision and zero observed test rows.

Independent audit status: `independent_audit_ok`.
Audit SHA-256: `5D0D26A174EDF750D3F6A0C8860FFDC183BBA6032EAE80857F31DFE418FC14DB`.

## Next Decision

The current protocol is frozen as a failed development gate. The test split
must not be opened and the validation-best ensemble must not replace the
train-selected single receptor.

The next preregistered development cycle should address QUBO specificity before
spending compute on a larger validation set:

1. Add an explicit linear top-k singleton-utility baseline matched to each QUBO
   subset size.
2. Expand the structurally diverse receptor pool so pairwise complementarity is
   a meaningful problem rather than four choose one to three.
3. Require a QUBO candidate to beat single-best, all-receptor, random-subset,
   and matched linear top-k baselines before making a QUBO-specific claim.
4. Record whether pairwise QUBO terms are nonzero and whether they change the
   selected subset; do not force nonzero weights merely to obtain a QUBO label.
5. Only after a non-degenerate train-only candidate exists, preregister a larger
   untouched validation extension to reduce metric uncertainty.

No current result establishes biological activity, accurate binding free
energy, general cross-target superiority, production screening value, quantum
speedup, or quantum advantage.
