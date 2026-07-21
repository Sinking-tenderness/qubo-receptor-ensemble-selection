# Stage 5 MAPK14 Train-696 Marginal Pair-Synergy QUBO Gate

Date: 2026-07-21

## Motivation

The inherited active-overlap QUBO and the aggregation-aligned raw pair-utility
QUBO both failed on Train-696. Raw pair BEDROC substantially improved the
objective, but it still repeated singleton information: the final QUBO and
matched linear subsets were identical.

This final train-only method hypothesis removed that double counting. For each
receptor pair and aggregation, the quadratic evidence was frozen as:

`S_ij = BEDROC(i,j) - max(BEDROC(i), BEDROC(j))`.

The terms were fitted only inside the current nested training fold. Signed
maximum-absolute scaling preserved positive synergy as a reward, negative
synergy as a penalty, and zero as no interaction. It did not min-max shift
negative evidence into a positive reward.

## Frozen Protocol

The 24 candidates combined two subset sizes, two aggregation methods, three
synergy weights, and two stability weights. Singleton BEDROC was the only
performance linear term. Marginal pair synergy was the only performance
quadratic term. Active coverage, decoy exposure, active overlap, redundancy,
and raw pair utility were fixed to zero.

The same four outer grouped-scaffold folds, three inner folds, three Vina seed
matrices, and deterministic tie breakers were retained. Acceptance required:

- nonconstant non-cardinality quadratic coefficients;
- a final subset different from matched linear top-k;
- noninferior primary, mean-seed, and worst-seed BEDROC versus matched linear;
- the same three checks versus nested greedy; and
- outer and final seed-subset mean Jaccard at least 0.5.

Validation and test rows remained unread. The preregistration was committed
before implementation under commit `652565e`.

## Selected Candidate

The nested procedure selected:

- family: `pair_synergy_qubo`;
- target size: 3;
- aggregation: `min_score`;
- synergy weight: 2.0;
- stability weight: 0.5; and
- subset: `2BAJ + 2QD9 + 3KQ7`.

Matched linear selected `2BAJ + 3KQ7 + 3MPT`. The 28 non-cardinality
quadratic coefficients had range 2.3354, so the QUBO was structurally
non-degenerate. Final seed-specific subset mean Jaccard was 0.667 and the mean
across outer fits was 0.833.

## OOF Results

| Method | ROC-AUC | PR-AUC | BEDROC20 | EF5% |
|---|---:|---:|---:|---:|
| Pair-synergy QUBO | 0.7517 | 0.7658 | 0.9225 | 1.8857 |
| Matched linear top-k | 0.7492 | 0.7579 | 0.8958 | 1.8286 |
| Nested greedy | 0.7417 | 0.7548 | 0.9151 | 1.9429 |
| Nested exhaustive | 0.7740 | 0.7865 | 0.9339 | 1.9429 |

Pair-synergy QUBO BEDROC deltas were:

- primary versus matched linear: +0.02665;
- mean seed versus matched linear: +0.02347;
- worst seed versus matched linear: +0.01629;
- primary versus nested greedy: +0.00741;
- mean seed versus nested greedy: +0.00530; and
- worst seed versus nested greedy: +0.00006.

All preregistered checks passed. The candidate ranked at the 94.6th percentile
among all 56 fixed triples on primary BEDROC, the 89.3rd percentile on
mean-seed BEDROC, and the 83.9th percentile on worst-seed BEDROC.

## Important Boundary

The positive result is narrow. Nested exhaustive remained better than the
QUBO by 0.01137 primary BEDROC, and this comparison was not a preregistered
acceptance condition. The final QUBO subset also equaled the final greedy
subset, although their nested OOF selection paths differed in two of four
outer folds. Therefore this result supports a useful quadratic interaction
relative to its matched linear objective and the frozen nested-greedy gate. It
does not show superiority to exhaustive classical search in an eight-receptor
pool or quantum advantage.

Two complete executions produced identical SHA-256 values for all ten output
files. The full summary SHA-256 is
`A074EE8B106DAA0E5BB163D8504AC9328BA38E3E3E56083F7475EF7CB96E325E`.
The repository test suite completed with 259 passed and 1 skipped in the
project Conda environment.

Status: `train_pair_synergy_qubo_gate_passed_validation_unavailable`.

## Decision

Train-696 objective and weight tuning stops here. The candidate is frozen, but
it remains a development-train result. The previously evaluated 40-active and
40-decoy validation panel is consumed and cannot be reused as fresh evidence.

The next gate must be separately preregistered and use only previously unused
rows from the already locked, scaffold-disjoint MAPK14 validation partition.
It must freeze the validation sample, the five-receptor comparator union, the
three-seed e32 docking protocol, metrics, and pass/fail criteria before scores
are generated. The locked test partition remains unreleased.
