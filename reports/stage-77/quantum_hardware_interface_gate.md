# Stage77 Quantum-Hardware Interface Gate

## Question

Which current quantum-hardware form can stably accept the frozen dense variable-k receptor-selection problem without turning a local proof of concept into a quantum-advantage claim?

## Direct CQM-to-BQM Gate

- Frozen CQM identities recovered: `80/80`.
- Largest converted BQM: `117` variables and `6786` interactions.
- Ideal Zephyr upper-bound embedding: `1053` physical qubits, maximum chain length `9`.
- Worst coefficient dynamic range: `6.586e+07`.
- Maximum signed precision required to retain even the largest objective bias: `30` bits.
- Decision: full direct QPU BQM route authorized = `False`. Topology is not the limiting gate; penalty precision is.

## Feasibility-Preserving Local BQM

- Fixed-k frontier subproblems: `500` across four historical development targets.
- Move variables: mean `34.08`, maximum `40`.
- Every encoded move is a fixed-k receptor swap with nonpositive quality-deficit change.
- Maximum objective identity residual: `1.776e-15`.
- Ideal Zephyr upper-bound embedding: `160` physical qubits, maximum chain length `4`.
- Q10 retained-bias fraction: mean `0.9924`, minimum `0.9630`.
- Local improving reward-cells: `15` from `3` unique fixed-k instances; hardware-resolvable reward-cells at four Q10 LSBs: `10` from `2` unique fixed-k instances. Reward quantiles are repeated evaluation cells, not independent physical BQMs, because a fixed-k reward shift is constant across subsets.
- Q10 plus 1% coefficient-noise proxy: feasible reads `1.0000`, resolvable-opportunity recovery `1.0000`, guarded nonworse runs `1.0000`.
- Decision: Advantage2 local reverse-annealing PoC ready for a budget request = `True`.

## Hardware Route Review

| Route | Native interface | Status | Reason |
| --- | --- | --- | --- |
| D-Wave Leap Hybrid CQM | CQM with explicit constraints | recommended_application_route | Accepts the frozen constrained model directly and avoids manual penalty precision, but is explicitly quantum-classical hybrid and cannot isolate a pure-QPU contribution. |
| D-Wave Advantage2 reverse annealing | warm-started local BQM/Ising | preferred_physical_hardware_poc | Matches the Stage76 frontier-warm mechanism and the Stage77 feasibility-preserving 40-variable swap BQMs; requires physical embedding and schedule calibration. |
| IBM Heron warm-start QAOA | gate-model cost Hamiltonian and custom mixer | secondary_small_problem_ablation | Warm-start QAOA is maintained and Heron has enough nominal qubits, but the full dense interaction graph creates substantial two-qubit routing depth. |
| Quantinuum H2 | 56-qubit fully connected gate model | local_only_not_primary | Connectivity is favorable for a reduced local QAOA, but 56 qubits cannot hold the largest 117-variable direct BQM and no target-specific advantage evidence exists. |
| IonQ Forte Enterprise | 36-qubit all-to-all gate model | insufficient_for_frozen_cap | The 36-qubit count is below the frozen 40-variable local cap and far below the full model. |
| QuEra Aquila analog neutral atoms | unit-disk-graph MIS | not_native_to_current_problem | Aquila has 256 atoms, but the current arbitrary signed dense BQM is not a native unit-disk MIS; generic transformations can add substantial overhead and alter the experimental question. |

## Claim Boundary

Stage77 is a local encoding, ideal-topology, quantization, and coefficient-noise proxy study on consumed historical development data. The Gaussian noise model is not a calibration model for a specific QPU. No cloud solver or QPU was contacted. Passing the local gate authorizes only a small, budget-capped reverse-annealing PoC with matched classical controls; it does not establish end-to-end speedup, quantum scaling, biological generalization, or quantum advantage.
