# QUBO Receptor Subset Sensitivity

## Grid

The prototype was rerun on the train split with:

- Target subset sizes `K=1,2,3`.
- Redundancy weights `0, 0.1, 0.25, 0.5, 1.0`.
- Count weight `0.10`.
- Size penalty `1.0`.

The validation and test scores were recorded for every train-selected subset,
but no validation or test value was used to select the subset.

## Selected Subsets

| Target size | Selected subset across redundancy weights |
|---:|---|
| K=1 | 1JVP for all weights |
| K=2 | 1AQ1 + 1JVP for all weights |
| K=3 | all three for weights 0-0.5; 1AQ1 + 1JVP at weight 1.0 |

The high redundancy penalty can override the nominal target size because the
size is implemented as a soft constraint. This is expected behavior and is
why the cost and size-penalty terms must be reported explicitly.

## Interpretation

The selected subset is stable for K=1 and K=2 over the tested redundancy
weights. This gives a small consistency check for the current prototype. It
does not establish that 1AQ1+1JVP is the correct biological ensemble because
the utility is based on only six train actives and redundancy is measured by
score correlation rather than pocket interactions.

The sensitivity output is generated locally at
`results/metrics/dude_cdk2_qubo_sensitivity.json` by
`scripts/sensitivity_qubo_receptor_subset.py`.

Before using a larger conformer pool, the next methodological improvement is
to replace the provisional utility and redundancy terms with explicitly
defined ligand-level early-enrichment and pocket-complementarity features,
then repeat the same train/validation/test sensitivity protocol.
