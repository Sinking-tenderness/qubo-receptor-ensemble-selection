# QAOA Aer Simulation Record

## Environment

The QAOA simulation uses a separate environment so that Qiskit's NumPy
requirements do not alter the docking/ProDy environment:

- Environment file: `environment/qaoa-environment.yml`
- Qiskit: `2.5.0`
- Qiskit Aer: `0.17.2`
- Python: `3.11.15`

The main `qubo-receptor-ensemble` environment was restored after an initial
in-place installation attempted to upgrade NumPy. The separation is part of
the reproducibility record.

## Circuit

- Algorithm: QAOA, depth `p=1`.
- Problem: the coverage-aware three-receptor QUBO.
- Variables: 1AQ1, 1HCL, 1JVP.
- Backend: local noise-free `AerSimulator`.
- Parameters: fixed grid search over gamma and beta.
- Measurement: computational-basis shots.

## Results

### Seed 20260711

- Grid: 9 x 9.
- Shots: 4096.
- Most common bitstring: `101`.
- Selected subset: `1AQ1 + 1JVP`.
- Most common count: 3798/4096.
- Best sampled energy: `-1.7852025544430559`.

### Seed 20260712

- Grid: 15 x 15.
- Shots: 4096.
- Most common bitstring: `101`.
- Selected subset: `1AQ1 + 1JVP`.
- Most common count: 3623/4096.
- Best sampled energy: `-1.7852025544430559`.

The sampled optimum matches the exact and simulated-annealing solution. This
is evidence that a quantum circuit simulation can solve the current toy QUBO,
not evidence of quantum advantage or a real QPU execution.

## Reproducibility

Run with the separate environment:

```powershell
conda run -n qubo-qaoa python scripts/run_qaoa_aer.py `
  --qubo-json results/metrics/dude_cdk2_coverage_qubo_train.json `
  --grid-size 15 `
  --shots 4096 `
  --seed 20260712 `
  --output results/metrics/dude_cdk2_qaoa_aer_result.json
```

Official references: [IBM Qiskit installation](https://quantum.cloud.ibm.com/docs/en/guides/install-qiskit)
and [Qiskit Aer installation](https://qiskit.github.io/qiskit-aer/getting_started.html).
