"""Optionally submit a QUBO to a D-Wave remote sampler without storing credentials."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import dimod

try:
    from .solve_qubo_with_ocean import load_bqm
except ImportError:
    from solve_qubo_with_ocean import load_bqm


def credential_source() -> str | None:
    if os.environ.get("DWAVE_API_TOKEN"):
        return "environment"
    try:
        from dwave.cloud.config import load_config

        config = load_config()
        if config.get("token"):
            return "dwave_config"
    except Exception:
        return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubo-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sampler",
        choices=["leap_hybrid", "qpu"],
        default="leap_hybrid",
    )
    parser.add_argument(
        "--run-remote",
        action="store_true",
        help="Actually submit to D-Wave; requires DWAVE_API_TOKEN or configured credentials.",
    )
    args = parser.parse_args()

    source = credential_source()
    result: dict[str, object] = {
        "qubo_json": str(args.qubo_json),
        "sampler": args.sampler,
        "dimod_version": dimod.__version__,
        "run_remote": args.run_remote,
        "credential_source": source,
        "credentials_written": False,
    }
    if not args.run_remote:
        result["status"] = "dry_run"
        result["message"] = "Pass --run-remote with configured D-Wave credentials to submit."
    else:
        if source is None:
            raise RuntimeError(
                "No D-Wave credentials found in DWAVE_API_TOKEN or dwave.conf; refusing to submit"
            )
        bqm, variables = load_bqm(args.qubo_json)
        if args.sampler == "leap_hybrid":
            from dwave.system import LeapHybridSampler

            sampler = LeapHybridSampler()
            close_sampler = True
        else:
            from dwave.system import DWaveSampler

            sampler = DWaveSampler()
            close_sampler = True
        try:
            start = time.perf_counter()
            sampleset = sampler.sample(bqm)
            elapsed = time.perf_counter() - start
            best = sampleset.first
            result.update(
                {
                    "status": "completed",
                    "variables": variables,
                    "selected_subset": sorted(
                        variable for variable in variables if best.sample[variable] == 1
                    ),
                    "energy": float(best.energy),
                    "runtime_seconds": round(elapsed, 6),
                    "solver": str(sampleset.info.get("solver_name", "unknown")),
                    "sample_count": len(sampleset),
                }
            )
        finally:
            if close_sampler and hasattr(sampler, "close"):
                sampler.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
