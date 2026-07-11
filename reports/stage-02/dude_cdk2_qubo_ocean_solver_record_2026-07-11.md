# QUBO Ocean Solver Validation

## Local Ocean Stack

The environment now includes:

- `dimod 0.12.22`
- `dwave-samplers 1.7.0`
- `dwave-system 1.24.0`

The versions are pinned in `environment/environment.yml`.

## Local Solver Results

The explicit coverage-aware QUBO was converted to a
`dimod.BinaryQuadraticModel` using the coefficient convention recorded in
the QUBO JSON.

| Solver | Reads | Result | Energy |
|---|---:|---|---:|
| Exhaustive enumeration | all 8 states | 1AQ1 + 1JVP | -1.7852025544430565 |
| `dimod.ExactSolver` | all 8 states | 1AQ1 + 1JVP | -1.7852025544430568 |
| `SimulatedAnnealingSampler` | 100 | 1AQ1 + 1JVP | -1.7852025544430568 |

The tiny floating-point difference is below numerical precision for this
purpose. This validates the QUBO conversion and solver interface; it does not
demonstrate quantum speedup.

## Remote Interface

`scripts/solve_qubo_remote.py` supports:

- `leap_hybrid`: D-Wave Leap hybrid BQM solver.
- `qpu`: direct `DWaveSampler` interface.

Without `--run-remote`, the script performs a dry-run. With `--run-remote`,
it requires `DWAVE_API_TOKEN` to be present in the process environment. The
token is never printed, written to JSON, or committed.

The current verified remote result is therefore only:

```text
status = dry_run
token_present = false
credentials_written = false
```

No quantum hardware claim is made until a real authenticated submission
returns a solver name, sample, energy, and timing metadata.

Official references: [dimod API](https://docs.dwavequantum.com/en/latest/ocean/api_ref_dimod/index.html),
[D-Wave system samplers](https://docs.dwavequantum.com/en/latest/ocean/api_ref_system/),
and [simulated annealing](https://docs.dwavequantum.com/en/latest/ocean/api_ref_dimod/generated/dimod.reference.samplers.SimulatedAnnealingSampler.sample.html).
