# Local QUBO Sampler Stability

The coverage-aware QUBO was sampled with the local
`dwave.samplers.SimulatedAnnealingSampler` using five independent seeds.

- Reads per seed: 100
- Sweeps per read: 1000
- Seeds: 11, 22, 33, 44, 55
- Exact reference: `dimod.ExactSolver`
- Exact energy: `-1.7852025544430568`

All five runs reached the exact reference energy and selected:

```text
1AQ1 + 1JVP
```

The exact-energy match rate was `1.0`. This validates local sampler
stability for the current three-variable toy-scale QUBO. It is not a
scalability or quantum-advantage result; a larger receptor pool is required
to make solver difficulty meaningful.

The machine-readable result is generated at
`results/metrics/dude_cdk2_qubo_sampler_benchmark.json` by
`scripts/benchmark_qubo_samplers.py`.
