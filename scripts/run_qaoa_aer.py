"""Run a small p=1 QAOA simulation for the explicit receptor QUBO."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def load_qubo(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    coeff = payload["qubo_coefficients"]
    variables = list(coeff["linear"])
    return variables, coeff


def energy(bits: tuple[int, ...], variables: list[str], coeff: dict[str, object]) -> float:
    selected = {variables[index] for index, bit in enumerate(bits) if bit}
    value = float(coeff["constant"])
    value += sum(
        float(coeff["linear"][variable]) for variable in selected
    )
    value += sum(
        float(value_)
        for key, value_ in coeff["quadratic"].items()
        if all(variable in selected for variable in key.split("__"))
    )
    return value


def build_qaoa_circuit(
    variables: list[str],
    coeff: dict[str, object],
    gamma: float,
    beta: float,
    shots: int,
    seed: int,
) -> QuantumCircuit:
    circuit = QuantumCircuit(len(variables), len(variables))
    for qubit in range(len(variables)):
        circuit.h(qubit)

    # Cost unitary for x=(1-Z)/2, ignoring only global phases.
    for index, variable in enumerate(variables):
        circuit.rz(-gamma * float(coeff["linear"][variable]), index)
    index_by_variable = {variable: index for index, variable in enumerate(variables)}
    for key, value in coeff["quadratic"].items():
        first, second = key.split("__", maxsplit=1)
        coupling = float(value)
        first_index = index_by_variable[first]
        second_index = index_by_variable[second]
        circuit.rz(-gamma * coupling / 2, first_index)
        circuit.rz(-gamma * coupling / 2, second_index)
        circuit.rzz(gamma * coupling / 2, first_index, second_index)

    for qubit in range(len(variables)):
        circuit.rx(2 * beta, qubit)
    circuit.measure(range(len(variables)), range(len(variables)))
    circuit.metadata = {"shots": shots, "seed": seed}
    return circuit


def expected_energy(counts: dict[str, int], variables: list[str], coeff: dict[str, object]) -> float:
    total = sum(counts.values())
    value = 0.0
    for bitstring, count in counts.items():
        bits = tuple(int(bit) for bit in bitstring[::-1])
        value += count * energy(bits, variables, coeff)
    return value / total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubo-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=15)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()

    variables, coeff = load_qubo(args.qubo_json)
    if len(variables) > 12:
        raise ValueError("this educational Aer implementation is limited to 12 qubits")
    simulator = AerSimulator(seed_simulator=args.seed)
    best_parameters = None
    best_expected = math.inf
    grid_size = max(3, args.grid_size)
    for gamma_index in range(grid_size):
        gamma = 2 * math.pi * gamma_index / grid_size
        for beta_index in range(grid_size):
            beta = math.pi * beta_index / grid_size
            circuit = build_qaoa_circuit(variables, coeff, gamma, beta, 256, args.seed)
            compiled = transpile(circuit, simulator, seed_transpiler=args.seed)
            counts = simulator.run(compiled, shots=256, seed_simulator=args.seed).result().get_counts()
            value = expected_energy(counts, variables, coeff)
            if value < best_expected:
                best_expected = value
                best_parameters = {"gamma": gamma, "beta": beta}

    circuit = build_qaoa_circuit(
        variables,
        coeff,
        best_parameters["gamma"],
        best_parameters["beta"],
        args.shots,
        args.seed,
    )
    compiled = transpile(circuit, simulator, seed_transpiler=args.seed)
    start = time.perf_counter()
    counts = simulator.run(compiled, shots=args.shots, seed_simulator=args.seed).result().get_counts()
    runtime = time.perf_counter() - start
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    most_common_bits = tuple(int(bit) for bit in ranked[0][0][::-1])
    most_common_selected = [
        variables[index] for index, bit in enumerate(most_common_bits) if bit
    ]
    energies = {
        bitstring: energy(tuple(int(bit) for bit in bitstring[::-1]), variables, coeff)
        for bitstring in counts
    }
    best_sample_bitstring = min(energies, key=energies.get)
    best_sample_bits = tuple(int(bit) for bit in best_sample_bitstring[::-1])
    result = {
        "qubo_json": str(args.qubo_json),
        "algorithm": "QAOA p=1 statevector/noise-free Aer simulation",
        "qiskit_version": __import__("qiskit").__version__,
        "qiskit_aer_version": __import__("qiskit_aer").__version__,
        "variables": variables,
        "grid_size": grid_size,
        "shots": args.shots,
        "seed": args.seed,
        "best_parameters": best_parameters,
        "grid_search_best_expected_energy": best_expected,
        "sample_expected_energy": expected_energy(counts, variables, coeff),
        "most_common_sample_count": ranked[0][1],
        "most_common_selected_subset": most_common_selected,
        "best_sampled_energy": energies[best_sample_bitstring],
        "best_sampled_subset": [
            variables[index] for index, bit in enumerate(best_sample_bits) if bit
        ],
        "counts": counts,
        "runtime_seconds": round(runtime, 6),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
