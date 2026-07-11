# CDK2 30A/30D Final Validation

## Data and docking completeness

- Benchmark: 30 DUD-E actives + 30 DUD-E decoys
- Receptors: 1AQ1, 1HCL, 1JVP, 2C68, 3RKB
- Protocol: AutoDock Vina 1.2.7, common aligned box, `exhaustiveness=1`,
  `num_modes=1`, `cpu=8`, base seed `20260727`
- Docking pairs: 5 x 60 = 300
- Successful pairs: 300/300
- Failed pairs: 0
- Matrix: `results/metrics/dude_cdk2_benchmark_30a_30d_e1_matrix.csv`
- Matrix summary: `results/metrics/dude_cdk2_benchmark_30a_30d_e1_summary.json`

## Leakage-aware evaluation

Five outer folds were used. For each fold, one fold was test, the next fold
was validation, and the remaining folds were train. Receptor selection used
train only; QUBO weights were not tuned on the outer test fold.

The QUBO utility and validation metric were BEDROC(alpha=20), and ensemble
scores were aggregated by mean docking score.

| Method | ROC-AUC | PR-AUC | BEDROC(alpha=20) | Top-10 active count |
|---|---:|---:|---:|---:|
| train-selected single receptor | 0.807 | 0.850 | 0.995 | 10 |
| coverage-aware QUBO subset | 0.768 | 0.813 | 0.985 | 10 |
| all-receptor mean ensemble | 0.782 | 0.815 | 0.980 | 9 |

The paired bootstrap QUBO-minus-single deltas were:

- ROC-AUC: `-0.0391`, 95% CI `[-0.0808, -0.0011]`
- PR-AUC: `-0.0375`, 95% CI `[-0.0889, -0.0005]`
- BEDROC: `-0.0152`, 95% CI `[-0.0667, 0.0000]`

The fixed-weight discriminative coverage variant produced the same aggregate
ranking on this benchmark. Therefore the intended QUBO innovation is not
validated by this experiment; the larger benchmark makes the negative result
more credible rather than hiding it.

## Local quantum simulation

The final five-variable coverage QUBO selected
`1JVP + 3RKB` under the exact classical solver. QAOA p=1 was simulated with
Qiskit Aer, 15 x 15 parameter grid, 4096 shots, and seeds `20260728`,
`20260729`, and `20260730`.

Across all three seeds, the best sampled energy matched the exact optimum and
the best sampled subset was `1JVP + 3RKB`. The most frequent bitstring was not
always the exact optimum, so this verifies sampling access to the optimum but
not reliable concentration or quantum advantage.

## Scientific conclusion

The current project has now passed an important negative-control milestone:
the full data path, multi-receptor docking, leakage-aware evaluation, QUBO
construction, bootstrap uncertainty, and local QAOA simulation are all
reproducible. However, the proposed receptor-subset selection method does not
yet improve early enrichment over the strongest single receptor on this CDK2
benchmark.

The next research decision should be structural rather than cosmetic:

1. expand beyond one small DUD-E target and use scaffold-aware splits;
2. add AF2/MD conformers rather than only additional crystal structures;
3. model per-ligand active coverage and decoy exposure with a validated
   structural/contact feature, not only docking-score correlation;
4. compare against greedy, clustering, random, and variable-cardinality
   baselines before attempting a hardware run.

No quantum hardware execution was performed.
