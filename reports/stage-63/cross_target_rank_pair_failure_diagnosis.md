# Stage63 cross-target rank-pair QUBO failure diagnosis

## Scope

This is a post-hoc, development-only mechanism diagnosis. It reads existing BACE1,
PPARG, PPARA, and PPARD fold results, performs no docking, accesses no fresh
validation or locked test rows, and does not tune a replacement objective.

## Main finding

The dominant failure is objective transfer, not solver search. The normalized
rank-pair QUBO training objective selected `k=2` as the best cardinality in
16/16
outer folds. Yet `k=2` beat `k=1` on only
2/16 holdouts.
The mean training objective change was
+0.033495, while mean holdout BEDROC
changed by -0.144086.

Training objective and holdout BEDROC were negatively rank-correlated in
15/16
folds (mean Spearman -0.579). This is the
signature of an over-optimistic pair complementarity term.

## Target-level fixed-k behavior

| Target | Best mean k | Mean k=1 BEDROC | Best mean BEDROC | Gain over k=1 |
|---|---:|---:|---:|---:|
| BACE1 | 6 | 0.878885 | 0.916917 | +0.038032 |
| PPARG | 6 | 0.869726 | 0.886901 | +0.017175 |
| PPARA | 1 | 0.874582 | 0.874582 | +0.000000 |
| PPARD | 1 | 0.809190 | 0.809190 | +0.000000 |

BACE1 and PPARG benefit on average from larger ensembles, while PPARA and PPARD
prefer one receptor. Therefore, ensemble benefit is target dependent; one universal
unshrunk pair reward is not supported.

## Solver diagnosis

Across 171 exact-certified cells, exact
optimization and the beam-plus-swap strong classical search had
0 positive objective gaps and
0 subset differences. A weaker
direct greedy search missed the stronger solution in
48/
162 comparable cells. The landscape can
trap weak greedy search, but the current instances do not separate exact QUBO from
a strong classical solver.

## PPARD nested-k diagnosis

| Fold | Inner-selected k | Outer oracle k | Curve Spearman | Selection regret |
|---:|---:|---:|---:|---:|
| 0 | 1 | 6 | 0.657 | 0.004281 |
| 1 | 1 | 2 | -0.200 | 0.042302 |
| 2 | 5 | 1 | -0.143 | 0.126958 |
| 3 | 1 | 1 | -0.257 | 0.000000 |

The mean inner-versus-outer k-curve Spearman was
0.014; mean selected-k regret
was 0.043385. The large fold-2 miss is
consistent with finite-sample cardinality instability.

## Next objective requirements

1. Estimate singleton and pair terms across resamples.
2. Reward a pair only when a preregistered lower confidence bound remains positive.
3. Select k using nested holdout lower-confidence bounds with k=1 as the fallback.
4. Keep best-single, linear Top-k, direct BEDROC greedy, strong same-QUBO search,
   and exact optimization where tractable.
5. Freeze all constants across historical targets, then test once on a genuinely
   new target. Do not retune and retest on PPARD.

## Decision

The current rank-pair QUBO remains a useful baseline but is not supported for a new
application claim. Stage63 authorizes cross-target development of an
uncertainty-shrunk objective, not fresh validation or quantum hardware.
