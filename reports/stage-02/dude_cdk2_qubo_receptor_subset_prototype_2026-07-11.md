# QUBO Receptor Subset Prototype

## Objective

The first transparent QUBO prototype uses one binary variable per receptor:

\[
Q(x)=-\sum_i u_i x_i+\lambda_r\sum_{i<j}r_{ij}x_ix_j
 +\lambda_c\sum_i x_i+\lambda_k(\sum_i x_i-K)^2.
\]

- `u_i`: train ROC-AUC of receptor i.
- `r_ij`: clipped non-negative train Spearman correlation between receptor
  score columns, used as a redundancy proxy.
- `K`: target subset size.
- `lambda_r`: redundancy penalty.
- `lambda_c`: per-receptor cost.
- `lambda_k`: soft target-size penalty.

This is a classical exhaustive QUBO prototype. No quantum hardware or
quantum advantage is claimed.

## Parameters

- Selection split: train only.
- Target size: `K=2`.
- Redundancy weight: `0.25`.
- Count weight: `0.10`.
- Size penalty: `1.0`.
- Receptors: 1AQ1, 1HCL, 1JVP.

## Train Inputs

| Receptor | Train ROC-AUC |
|---|---:|
| 1AQ1 | 0.694 |
| 1HCL | 0.611 |
| 1JVP | 0.700 |

Train score-column redundancy values:

- 1AQ1-1HCL: `0.859`
- 1AQ1-1JVP: `0.809`
- 1HCL-1JVP: `0.801`

## Exhaustive Solution

The minimum-QUBO subset is:

```text
1AQ1 + 1JVP
```

The output is generated at
`results/metrics/dude_cdk2_qubo_train_prototype.json`.

This agrees with the classical train-selected two-receptor baseline. The
agreement is a consistency check, not evidence of a quantum benefit.

## Limitations

1. ROC-AUC is only a provisional utility; early-enrichment utility should be
   tested separately.
2. Spearman correlation is only a score-level redundancy proxy and does not
   establish biological pocket complementarity.
3. The train split contains only six actives.
4. The weights were chosen for a transparent teaching example, not optimized
   or validated as final research parameters.
5. With only three receptors, exhaustive enumeration is sufficient. A larger
   AF2/MD conformer pool is needed before QUBO scalability becomes meaningful.
