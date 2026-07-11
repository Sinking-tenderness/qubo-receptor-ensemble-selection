"""Solve an explicit QUBO with local Ocean exact or simulated-annealing samplers."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import dimod


def load_bqm(path: Path) -> tuple[dimod.BinaryQuadraticModel, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    coefficients = payload["qubo_coefficients"]
    linear = coefficients["linear"]
    quadratic = coefficients["quadratic"]
    qubo: dict[tuple[str, str], float] = {
        (receptor_id, receptor_id): float(value)
        for receptor_id, value in linear.items()
    }
    for key, value in quadratic.items():
        first, second = key.split("__", maxsplit=1)
        qubo[(first, second)] = float(value)
    bqm = dimod.BinaryQuadraticModel.from_qubo(
        qubo, offset=float(coefficients["constant"])
    )
    return bqm, list(linear)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubo-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sampler", choices=["exact", "simulated_annealing"], default="exact")
    parser.add_argument("--num-reads", type=int, default=100)
    parser.add_argument("--num-sweeps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()

    bqm, variables = load_bqm(args.qubo_json)
    start = time.perf_counter()
    if args.sampler == "exact":
        sampler = dimod.ExactSolver()
        sampleset = sampler.sample(bqm)
    else:
        from dwave.samplers import SimulatedAnnealingSampler

        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample(
            bqm,
            num_reads=args.num_reads,
            num_sweeps=args.num_sweeps,
            seed=args.seed,
        )
    runtime = time.perf_counter() - start
    best = sampleset.first
    selected = sorted(variable for variable in variables if best.sample[variable] == 1)
    expected = json.loads(args.qubo_json.read_text(encoding="utf-8"))["best_subset"]["subset"]
    result = {
        "qubo_json": str(args.qubo_json),
        "sampler": args.sampler,
        "dimod_version": dimod.__version__,
        "variables": variables,
        "selected_subset": selected,
        "energy": float(best.energy),
        "expected_exhaustive_subset": sorted(expected),
        "matches_exhaustive_subset": selected == sorted(expected),
        "runtime_seconds": round(runtime, 6),
        "num_reads": args.num_reads if args.sampler != "exact" else None,
        "num_sweeps": args.num_sweeps if args.sampler != "exact" else None,
        "seed": args.seed if args.sampler != "exact" else None,
        "sample_count": len(sampleset),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0 if result["matches_exhaustive_subset"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
