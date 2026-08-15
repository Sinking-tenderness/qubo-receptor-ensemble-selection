# Stage78 Advantage2 External Execution

## Current Stop

Stage78 is frozen and independently audited. Local preparation made zero cloud
queries, zero QPU submissions, and zero new docking calls. The next operation is
a metadata-only Leap preflight. It selects a live Advantage2 Zephyr solver,
freezes the current working-graph identity, and constructs two physical
embeddings for every frozen variable count. It does not sample the QPU.

## Required Access

- D-Wave Leap project with an API token for arbitrary direct-QPU submissions.
- Direct access to a production Advantage2 QPU with Zephyr topology.
- At least 4,000 available solver qubits.
- Solver support for `initial_state`, `anneal_schedule`,
  `reinitialize_state`, `num_reads`, and raw answers.
- Project allowance of at least 20 seconds of QPU access time. The experiment
  has a client-side hard limit of 20 seconds and will stop if it is exceeded.

A demo-only or self-sign-up plan without an API token is insufficient. This is
not an AutoDL GPU rental. A small Linux client is enough: 2 CPU cores, 8 GB RAM,
10 GB free disk, Python 3.11, and outbound HTTPS. No GPU or CUDA is required.

## Environment

```bash
conda env create -f environment/stage78_dwave_advantage2.yml
conda activate qubo-receptor-stage78-qpu

python - <<'PY'
import importlib.metadata as metadata
print("ocean:", metadata.version("dwave-ocean-sdk"))
print("dwave-system:", metadata.version("dwave-system"))
print("dimod:", metadata.version("dimod"))
print("numpy:", metadata.version("numpy"))
PY
```

The frozen execution environment uses Ocean SDK 9.4.0, `dwave-system` 1.35.0,
`dimod` 0.12.22, and NumPy 2.x. Keep this environment separate from docking and
earlier analysis environments.

## Preflight Only

Set the token without writing it into the project or archive:

```bash
export DWAVE_API_TOKEN='REPLACE_WITH_THE_LEAP_PROJECT_TOKEN'
export STAGE78_PYTHON="$(command -v python)"

bash scripts/experimental/quantum/run_stage78_advantage2_preflight_remote.sh \
  2>&1 | tee stage78_preflight.log
```

If the project exposes several Zephyr QPUs, an exact solver can be requested:

```bash
export STAGE78_SOLVER_NAME='Advantage2_system1'
```

Do not use `dwave ping` as a metadata test because it submits a sample problem.
The supplied preflight script stops before all QPU sampling. Return these files
for review before proceeding:

- `stage78_advantage2_preflight_results_v1.tar.gz`
- `stage78_preflight.log`

The preflight must pass both embeddings for K38 and K40 with maximum chain
length at most 6 and at most 300 physical qubits. It also freezes `solver_id`,
`graph_id`, the complete working-graph hash, anneal limits, bias ranges, and
per-group coupling limits. A changed live graph invalidates the embeddings and
forces a new preflight.

## Paid Phases After Review

Calibration uses only the PPARG sub-resolution diagnostic: 18 conditions, 4
gauges, and 100 reads per gauge, for 72 QPU submissions and 7,200 reads.
Confirmation then uses two untouched PPARG positives and three cross-target
hard negatives: 2 embeddings by 8 gauges by 2 modes, for 160 submissions and
16,000 reads. Total planned use is 232 submissions and 23,200 reads.

Paid phases require both `--authorize-paid-qpu` and the exact environment
acknowledgement below. They are intentionally omitted from the preflight script:

```bash
export STAGE78_QPU_ACK=I_ACCEPT_STAGE78_QPU_CHARGES
```

The hardware outcome is a physical feasibility PoC. It cannot by itself support
a quantum-advantage, scaling, biological-generalization, or end-to-end speedup
claim.
