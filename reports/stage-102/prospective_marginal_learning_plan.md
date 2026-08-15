# Stage102 prospective marginal learning plan

## Decision

Do not tune another QUBO coefficient on MK14, PPARG, BACE1, PPARA, or PPARD. Do not use quantum hardware. First collect independent target-level evidence about whether adding a receptor improves held-out early recognition.

## Phase A: two new development targets

Use the receptors that already passed the frozen three-seed cognate-redocking gate:

- EGFR: 12 receptors.
- FA10: 13 receptors.

For each target, deterministically allocate 120 actives and 480 decoys by joined source-ID, canonical-SMILES, and Bemis-Murcko-scaffold groups. Run Uni-Dock 1.1.3 with three seeds and retain each seed-level score.

The expected work is:

| Target | Receptors | Ligands | Seeds | Receptor-ligand-seed pairs |
|---|---:|---:|---:|---:|
| EGFR | 12 | 600 | 3 | 21,600 |
| FA10 | 13 | 600 | 3 | 23,400 |
| Total | 25 | 1,200 | 3 | 45,000 |

These two targets are development expansion, not confirmation. Their purpose is to test whether scaffold-bootstrap uncertainty, active-versus-decoy rescue, receptor disagreement, structural distance, seed agreement, and QUBO solution stability can predict marginal held-out BEDROC20.

Exactly two model candidates are allowed: a mechanistic bootstrap lower-bound rule and target-held-out L2 ridge with alpha 1.0. No third model or same-target threshold tuning may be introduced after results are seen.

## Phase A gate

Advance only if all frozen criteria pass:

- Mean target gain over the train-selected single receptor is at least +0.02 BEDROC20.
- Worst-target gain is at least -0.02.
- At least four targets gain at least +0.02 after combining old diagnostic targets with the two new targets.
- At least 12 outer folds select more than one receptor.
- At least one of EGFR or FA10 has positive gain.
- Marginal-sign accuracy on the two new targets is at least 60%.

Failure stops adaptive-cardinality development and all hardware work.

## Phase B: untouched PARP1

PARP1 remains unopened until Phase A passes and exactly one model is frozen. Build a PARP1 structural pool, require at least 12 receptors to pass cognate redocking, then run the frozen model once on 75 fresh-validation actives and 1,501 decoys. No PARP1-specific objective, threshold, feature, or receptor-count tuning is permitted.

## Adaptive QUBO

Only after the biological gate passes, encode the frozen cardinality prior in

$$
\max_{x,y}\left[
\sum_i u_i x_i+
\eta\sum_{i<j}\delta_{ij}x_ix_j+
\sum_k\gamma_k y_k-
A\left(\sum_i x_i-\sum_k k y_k\right)^2-
B\left(\sum_k y_k-1\right)^2
\right].
$$

The one-hot variable $y_k$ chooses $k\in\{1,2,3\}$. The value $\gamma_k$ comes from the single frozen cross-target marginal model, not from PARP1 outcomes.

## Hardware boundary

Quantum hardware remains blocked until PARP1 fresh validation passes and the exact adaptive-QUBO optimum both differs from and outperforms direct greedy or one-swap local search on a validated hard instance.
