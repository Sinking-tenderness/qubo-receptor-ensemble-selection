# Next optimization specification after Stage101

## What is retained

Stage99's robust pair QUBO remains the receptor-subset selector at a fixed cardinality. Stage101 shows that the unresolved component is not exact subset search, but predicting whether the next receptor will improve held-out screening.

## Why threshold tuning stops

For the `k=1` to `k=2` transition, the inner-fold marginal BEDROC and held-out marginal BEDROC have Spearman correlation `-0.4723` (`p=0.0171`) and only `36%` sign agreement. A stricter threshold reduces receptor additions but does not produce a positive mean target gain. Another threshold fitted to the same matrices would therefore be post-hoc overfitting.

At the same time, the held-out oracle over `k in {1,2,3}` has a mean target gain of `+0.041197`. This is a ceiling, not an attainable result, but it establishes that variable cardinality has potential value if a transferable marginal signal can be learned.

## Prospective marginal-value model

For every proposed transition from `k-1` to `k`, compute features using only the calibration panel:

1. Mean, standard error, worst value, and positive fraction of inner-fold marginal BEDROC.
2. Bootstrap probability that the marginal gain is positive, grouped by ligand scaffold.
3. Receptor score-rank disagreement and active-ligand coverage overlap.
4. Structural pocket distance and pocket-state cluster coverage.
5. Docking-seed agreement and score variance.
6. QUBO optimum gap and stability of the selected subset under ligand bootstrap.
7. Incremental compute cost of adding the receptor.

The target is the marginal BEDROC on a disjoint evaluation panel. Targets, ligand scaffolds, and evaluation labels must remain held out from predictor fitting.

## Adaptive QUBO after the signal gate

Let `x_i` select receptor `i`, and let one-hot variable `y_k` select the ensemble size. Once a marginal-value prior has passed prospective validation, use

$$
\max_{x,y}\left[
\sum_i u_i x_i
+ \eta\sum_{i<j}\delta_{ij}x_ix_j
+ \sum_{k=1}^{K}\gamma_k y_k
- A\left(\sum_i x_i-\sum_{k=1}^{K}k y_k\right)^2
- B\left(\sum_{k=1}^{K}y_k-1\right)^2
\right].
$$

Here, `gamma_k` is a frozen, out-of-target estimate of cumulative ensemble value minus compute cost. The model chooses both the subset and its size; `k=3` is no longer imposed.

An equivalent unconstrained form is

$$
\max_x\left[
\sum_i (u_i-\lambda_i)x_i
+ \eta\sum_{i<j}\delta_{ij}x_ix_j
\right],
$$

but a single global receptor cost is not adequate until the marginal-value model is validated.

## Prospective experiment

1. Freeze the five current targets as consumed diagnostic data.
2. Add independent proteins from at least three non-homologous target families.
3. For each protein, split ligands by scaffold into calibration and untouched evaluation panels.
4. Fit the marginal model using other proteins only; predict the held-out protein.
5. Compare adaptive QUBO with single receptor, fixed `k`, greedy stopping, mean Top-k, and an exact oracle reported only as a ceiling.
6. Do not use quantum hardware until the biological objective gate passes.

## Go/No-Go gate

The adaptive route advances only if all conditions hold prospectively:

- Mean target gain over the train-selected single receptor is at least `+0.02` BEDROC.
- Worst-target gain is at least `-0.02`.
- At least three targets gain at least `+0.02`.
- At least eight outer folds select more than one receptor.
- Selection remains stable across docking seeds and scaffold bootstrap.
- The exact QUBO optimum differs from the strongest classical local baseline on at least one validated hard instance.

Failure means the paper should present receptor-ensemble selection and negative quantum-readiness evidence without claiming a quantum optimization advantage.
