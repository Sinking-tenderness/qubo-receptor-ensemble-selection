# QUBO failure diagnosis and objective repair

## What failed

The historical experiments do not show a single software bug. They show five separable model-design failures.

1. **Cardinality pressure replaced ranking quality.** Stage42e showed that the old coverage objective increased monotonically from `k=1` to `k=6`, while held-out BEDROC peaked at `k=2` and then declined. The objective rewarded adding receptors even when early recognition worsened.
2. **A pairwise proxy did not equal biological complementarity.** The old rank-pair QUBO rewarded the best rank from either receptor. This let one extreme receptor score dominate a ligand and increased false-positive exposure as the ensemble grew. Stage98 also showed that low receptor-score correlation was not a reliable label-free proxy: only one of five targets gained at `k=3`, with a mean gain of `-0.046388`.
3. **The objective generalized poorly.** The old QUBO appeared positive on full PPARG data (`+0.026958` over a single receptor) but lost `-0.127250` in outer folds. On PPARA it lost `-0.269477` versus a single receptor. This is outcome overfitting, not solver failure.
4. **One fixed receptor count was biologically inappropriate.** BACE1 already had a near-ceiling single receptor, while MK14 and PPARD could benefit from an ensemble. Stage20 and Stage99 both show that the preferred `k` changes by target and fold.
5. **Optimization and hardware were not the scientific bottleneck.** Strong classical search matched exact references throughout the certified Stage74/75 frontier, Stage80 found zero multi-move traps, and Stage86 returned zero feasible physical samples in 25 global-penalty runs. The first two results mean the QUBO landscapes were mostly easy; the last result is a separate penalty/encoding failure.

## Frozen repair

Stage99 replaces the old pair reward with a conservative two-support objective. Within each outer training fold, receptor scores are transformed to empirical percentile ranks $r_{li} \in [0,1]$, where lower is better. The singleton utility is

$u_i = \mathbb{E}_{a \in A}[e^{-20r_{ai}}] - 1.5\,\mathbb{E}_{d \in D}[e^{-20r_{di}}].$

For a receptor pair, one extreme score is no longer sufficient. Pair support uses the mean of the two ranks:

$u_{ij} = \mathbb{E}_{a \in A}\left[e^{-20(r_{ai}+r_{aj})/2}\right] - 1.5\,\mathbb{E}_{d \in D}\left[e^{-20(r_{di}+r_{dj})/2}\right].$

The quadratic coefficient is

$\delta_{ij}=u_{ij}-(u_i+u_j)/2.$

For a fixed cardinality $k$, the repaired QUBO maximizes

$Q_k(x)=\frac{1}{k}\sum_i u_i x_i+\frac{1}{\binom{k}{2}}\sum_{i<j}\delta_{ij}x_ix_j,$

subject to $\sum_i x_i=k$. The exact screen enforces this Hamming weight directly, so its result is not confounded by a large cardinality penalty.

## Stage99 result

Under five-fold group holdout, the repaired fixed-`k=3` QUBO improved on the previous pair QUBO on PPARG, BACE1, PPARA, and PPARD. Relative to the best train-selected single receptor, its target gains were:

| Target | Single BEDROC | Repaired k=3 BEDROC | Gain |
|---|---:|---:|---:|
| MK14 | 0.370233 | 0.423855 | +0.053622 |
| PPARG | 0.887457 | 0.926855 | +0.039397 |
| BACE1 | 0.983155 | 0.969630 | -0.013525 |
| PPARA | 0.832194 | 0.730674 | -0.101520 |
| PPARD | 0.697470 | 0.755831 | +0.058361 |

The mean gain was only `+0.007267`, and the worst-target loss was `-0.101520`, so the frozen gate failed. Nested adaptive selection of `k in {1,2,3}` also failed: mean gain `-0.008496`, worst-target gain `-0.070444`.

Exact QUBO enumeration differed from greedy plus one-swap in `5/75` fixed-`k` cells. On those five outer holdouts, exact selection improved BEDROC three times and reduced it twice. Therefore better QUBO energy is not yet a dependable surrogate for better virtual screening.

## What we learned

- The repaired objective is directionally better than the old pair QUBO, so failure analysis produced a real improvement rather than a cosmetic rewrite.
- PPARA is the decisive counterexample. Retuning the decoy penalty on the same matrices would convert diagnosis into target-specific overfitting and is not authorized.
- The remaining scientific gap is not finding a stronger generic optimizer. It is learning a transferable utility model that predicts which receptor combinations will improve unseen ligand ranking.
- Quantum hardware remains blocked. A physical solver can optimize the repaired coefficients, but it cannot repair their residual cross-target mismatch.

## Next defensible experiment

Do not tune another scalar weight on these five matrices. Freeze Stage99 as a post-hoc diagnostic and design one prospective target experiment with:

1. receptor selection trained on separate historical targets or a separate ligand panel;
2. a target-specific stopping rule for $k$ frozen before the protected panel is opened;
3. single receptor, linear top-`k`, nested greedy, exact QUBO, and one-swap controls;
4. a solver-value gate requiring exact QUBO to beat one-swap in objective and protected BEDROC;
5. no hardware submission unless both application and solver-value gates pass.

This preserves a viable research question: not whether any QUBO can be made to fit these matrices, but whether a conservative pairwise ensemble utility transfers to a genuinely unseen screening problem and contains a nontrivial optimization gap.
