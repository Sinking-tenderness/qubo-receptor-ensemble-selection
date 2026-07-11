# Eight-Receptor and Train-Only Normalization Checkpoint

## Eight-receptor matrix

The original 30A/30D benchmark was docked against three additional prepared
CDK2 structures: 1H00, 1Y8Y, and 2C69. The final matrix contains 60 ligands,
8 receptors, and 480 successful receptor-ligand pairs with no missing scores.

Using raw Vina scores, the strongest single receptor was 1H00 (ROC-AUC
0.836, PR-AUC 0.853). The all-receptor mean ensemble was lower (ROC-AUC
0.794, PR-AUC 0.814). Low score correlation for 2C68 and 2C69 confirmed that
the pool contains ranking diversity, but diversity alone did not produce an
ensemble gain.

## Why normalize scores

Raw Vina scores from different receptor conformers may have different offsets
and scales. Directly averaging them can reward a receptor because of its score
calibration rather than because it ranks active ligands well.

The new `train_minmax` mode fits each receptor's minimum and maximum on the
training ligands only, then transforms validation and test scores with those
fixed training bounds. No held-out score distribution is used for selection.

## Scaffold-group repeated CV

Five-fold outer CV was repeated with three scaffold assignments using seeds
20260731, 20260732, and 20260733. Each fold kept Bemis-Murcko scaffolds in one
fold only. The QUBO target size was 2, utility and validation metric were
ROC-AUC, aggregation was mean score, and receptor scores used train-only
min-max normalization.

After averaging out-of-fold scores across the three repetitions:

| Method | ROC-AUC | PR-AUC | BEDROC(alpha=20) |
|---|---:|---:|---:|
| train-selected single receptor | 0.813 | 0.841 | 0.986 |
| QUBO subset | 0.834 | 0.869 | 0.995 |
| all-receptor mean | 0.804 | 0.840 | 0.993 |

Paired bootstrap on the repeated out-of-fold scores gave QUBO minus single:

- ROC-AUC: `+0.0209`, 95% CI `[-0.0289, +0.0756]`
- PR-AUC: `+0.0279`, 95% CI `[-0.0158, +0.0913]`
- BEDROC: `+0.0119`, 95% CI `[-0.0041, +0.0674]`

The point estimates are encouraging and the all-receptor baseline is worse,
but all confidence intervals still include zero. This is a candidate method,
not a validated innovation.

## Next validation gate

The 100A/100D benchmark is being prepared with the same IDs and scaffold
split. It will test whether the normalized QUBO signal survives a larger
ligand set. Only after that result, and an independent receptor/conformer
comparison, should the method be described as an effective improvement.
