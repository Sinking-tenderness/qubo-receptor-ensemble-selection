# Stage91 BACE1 group-robust rescue preregistration

Status: `stage91_bace1_group_robust_rescue_preregistered`.

The rescue now tests whether a six-receptor subset can protect the worst-served medicinal-chemistry series rather than merely maximize average ligand ranking.

## Frozen data roles

| Role | Molecules | High | Low | Gray | Core series | Score status |
|---|---:|---:|---:|---:|---:|---|
| development | 365 | 248 | 52 | 65 | 6 | development preparation only |
| confirmation_a | 221 | 168 | 26 | 27 | 6 | locked |
| confirmation_b | 185 | 30 | 56 | 99 | 7 | locked |
| locked_test | 115 | 59 | 20 | 36 | 4 | locked |

## Frozen objective

`maximize 0.40*t + 0.30*mean_group_coverage - 0.20*global_low_potency_exposure - 0.10*mean_pair_receptor_overlap`

Primary k is 6 over 34 receptors (1,344,904 subsets). Coefficients cannot be tuned after docking. k=4 and k=8 are sensitivity analyses only.

## Release gate

Development must show a strict certified improvement over greedy plus all one-swaps and a reproducible multi-move trap. Only then may confirmation A be prepared and docked. Confirmation B, locked test, and quantum execution remain sequentially blocked.
