"""Benchmark local exact and simulated-annealing QUBO sampling stability."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import dimod
from dwave.samplers import SimulatedAnnealingSampler

from solve_qubo_with_ocean import load_bqm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubo-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-reads", type=int, default=100)
    parser.add_argument("--num-sweeps", type=int, default=1000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    args = parser.parse_args()

    bqm, variables = load_bqm(args.qubo_json)
    exact = dimod.ExactSolver().sample(bqm).first
    exact_energy = float(exact.energy)
    sampler = SimulatedAnnealingSampler()
    runs = []
    for seed in args.seeds:
        start = time.perf_counter()
        sampleset = sampler.sample(
            bqm,
            num_reads=args.num_reads,
            num_sweeps=args.num_sweeps,
            seed=seed,
        )
        best = sampleset.first
        selected = sorted(variable for variable in variables if best.sample[variable] == 1)
        runs.append(
            {
                "seed": seed,
                "best_energy": float(best.energy),
                "matches_exact_energy": abs(float(best.energy) - exact_energy) < 1e-10,
                "selected_subset": selected,
                "runtime_seconds": round(time.perf_counter() - start, 6),
                "sample_count": len(sampleset),
            }
        )
    result = {
        "qubo_json": str(args.qubo_json),
        "dimod_version": dimod.__version__,
        "variables": variables,
        "exact_energy": exact_energy,
        "runs": runs,
        "exact_energy_match_rate": sum(run["matches_exact_energy"] for run in runs) / len(runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
