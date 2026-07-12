"""Measure OpenMM throughput for a fixed prepared system without trajectory output."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


def ns_per_day(steps: int, timestep_fs: float, elapsed_seconds: float) -> float:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if timestep_fs <= 0.0:
        raise ValueError("timestep_fs must be positive")
    if elapsed_seconds <= 0.0:
        raise ValueError("elapsed_seconds must be positive")
    simulated_nanoseconds = steps * timestep_fs / 1_000_000.0
    return simulated_nanoseconds * 86_400.0 / elapsed_seconds


def build_platform_properties(platform_name: str, cpu_threads: int | None, precision: str | None) -> dict[str, str]:
    properties: dict[str, str] = {}
    if platform_name == "CPU" and cpu_threads is not None:
        if cpu_threads < 1:
            raise ValueError("cpu_threads must be >= 1")
        properties["Threads"] = str(cpu_threads)
    if platform_name in {"OpenCL", "CUDA"} and precision is not None:
        if precision not in {"single", "mixed", "double"}:
            raise ValueError("GPU precision must be one of: single, mixed, double")
        properties["Precision"] = precision
    return properties


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-xml", type=Path, required=True)
    parser.add_argument("--positions-pdb", type=Path, required=True)
    parser.add_argument("--platform", choices=("CPU", "OpenCL", "CUDA"), required=True)
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--gpu-precision", choices=("single", "mixed", "double"), default="mixed")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--benchmark-steps", type=int, default=1000)
    parser.add_argument("--timestep-fs", type=float, default=2.0)
    parser.add_argument("--temperature-kelvin", type=float, default=300.0)
    parser.add_argument("--friction-per-ps", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if args.benchmark_steps <= 0:
        raise ValueError("benchmark_steps must be positive")
    if args.seed <= 0:
        raise ValueError("seed must be positive")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; use --overwrite after review: {args.output}")

    from openmm import Context, LangevinMiddleIntegrator, Platform, XmlSerializer, unit
    from openmm.app import PDBFile
    import openmm

    system = XmlSerializer.deserialize(args.system_xml.read_text(encoding="utf-8"))
    positions = PDBFile(str(args.positions_pdb))
    properties = build_platform_properties(args.platform, args.cpu_threads, args.gpu_precision)
    platform = Platform.getPlatformByName(args.platform)
    integrator = LangevinMiddleIntegrator(
        args.temperature_kelvin * unit.kelvin,
        args.friction_per_ps / unit.picosecond,
        args.timestep_fs * unit.femtoseconds,
    )
    integrator.setRandomNumberSeed(args.seed)
    context = Context(system, integrator, platform, properties)
    try:
        context.setPositions(positions.positions)
        context.setVelocitiesToTemperature(args.temperature_kelvin * unit.kelvin, args.seed)
        if args.warmup_steps:
            integrator.step(args.warmup_steps)
        initial_energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        start = time.perf_counter()
        integrator.step(args.benchmark_steps)
        elapsed = time.perf_counter() - start
        final_energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        if not all(math.isfinite(value) for value in (initial_energy, final_energy)):
            raise RuntimeError("benchmark produced non-finite potential energy")
        resolved_properties = {
            name: context.getPlatform().getPropertyValue(context, name)
            for name in context.getPlatform().getPropertyNames()
        }
    finally:
        del context
        del integrator

    result = {
        "schema_version": "1.0",
        "operation": "fixed-step OpenMM throughput benchmark; no trajectory file was written",
        "system_xml": args.system_xml.as_posix(),
        "positions_pdb": args.positions_pdb.as_posix(),
        "openmm_version": openmm.version.version,
        "platform": args.platform,
        "requested_properties": properties,
        "resolved_properties": resolved_properties,
        "system_atom_count": system.getNumParticles(),
        "warmup_steps": args.warmup_steps,
        "benchmark_steps": args.benchmark_steps,
        "timestep_fs": args.timestep_fs,
        "temperature_kelvin": args.temperature_kelvin,
        "friction_per_ps": args.friction_per_ps,
        "seed": args.seed,
        "elapsed_seconds": round(elapsed, 6),
        "steps_per_second": round(args.benchmark_steps / elapsed, 4),
        "nanoseconds_per_day": round(ns_per_day(args.benchmark_steps, args.timestep_fs, elapsed), 6),
        "potential_energy_kj_per_mol": {
            "after_warmup": round(initial_energy, 6),
            "after_benchmark": round(final_energy, 6),
        },
        "interpretation_note": "A finite short benchmark verifies platform execution and estimates throughput. It is not an equilibration, stability, or sampling assessment.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
