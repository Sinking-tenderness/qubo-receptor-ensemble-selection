"""Run a resumable chunked OpenMM NPT production pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

try:
    from .run_openmm_equilibration import atomic_write_json
    from .run_openmm_equilibration_smoke import (
        ca_indices,
        coordinates_nm,
        load_smoke_config,
        platform_properties,
        state_record,
        steps_for_duration,
    )
except ImportError:
    from run_openmm_equilibration import atomic_write_json
    from run_openmm_equilibration_smoke import (
        ca_indices,
        coordinates_nm,
        load_smoke_config,
        platform_properties,
        state_record,
        steps_for_duration,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def production_steps(duration_ns: float, timestep_fs: float) -> int:
    return steps_for_duration(duration_ns * 1000.0, timestep_fs)


def validate_schedule(
    total_steps: int, metrics_steps: int, frame_steps: int, checkpoint_steps: int
) -> None:
    if not (total_steps % checkpoint_steps == 0):
        raise ValueError("production duration must be an exact multiple of checkpoint interval")
    if checkpoint_steps % metrics_steps or checkpoint_steps % frame_steps:
        raise ValueError("checkpoint interval must be a multiple of metric and frame intervals")
    if frame_steps % metrics_steps:
        raise ValueError("frame interval must be a multiple of metrics interval")


def chunk_filename(prefix: str, start_ps: float, end_ps: float) -> str:
    return f"{prefix}_{start_ps:010.3f}_{end_ps:010.3f}ps.dcd"


def initialize_progress(experiment_id: str) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "status": "running",
        "completed_steps": 0,
        "records": [],
        "completed_trajectory_chunks": [],
    }


def validate_progress(progress: dict[str, object], total_steps: int) -> None:
    completed = int(progress.get("completed_steps", -1))
    if completed < 0 or completed > total_steps:
        raise ValueError("invalid production completed_steps")
    if not isinstance(progress.get("records"), list):
        raise ValueError("production progress records must be a list")
    if not isinstance(progress.get("completed_trajectory_chunks"), list):
        raise ValueError("completed_trajectory_chunks must be a list")


def write_metrics(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        raise ValueError("cannot write empty production metrics")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    config = load_smoke_config(args.config)
    inputs = config["inputs"]
    platform_settings = config["platform"]
    dynamics = config["dynamics"]
    outputs = config["outputs"]
    assert isinstance(inputs, dict)
    assert isinstance(platform_settings, dict)
    assert isinstance(dynamics, dict)
    assert isinstance(outputs, dict)

    run_directory = Path(str(outputs["run_directory"]))
    progress_path = Path(str(outputs["progress_json"]))
    metrics_path = Path(str(outputs["metrics_csv"]))
    manifest_path = Path(str(outputs["manifest"]))
    checkpoint_path = Path(str(outputs["checkpoint"]))
    final_state_path = Path(str(outputs["final_state_xml"]))
    final_pdb_path = Path(str(outputs["final_pdb"]))
    prefix = str(outputs["trajectory_prefix"])
    run_directory.mkdir(parents=True, exist_ok=True)

    timestep_fs = float(dynamics["timestep_fs"])
    total_steps = production_steps(float(dynamics["production_duration_ns"]), timestep_fs)
    metrics_steps = steps_for_duration(float(dynamics["metrics_interval_ps"]), timestep_fs)
    frame_steps = steps_for_duration(float(dynamics["frame_interval_ps"]), timestep_fs)
    checkpoint_steps = steps_for_duration(float(dynamics["checkpoint_interval_ps"]), timestep_fs)
    validate_schedule(total_steps, metrics_steps, frame_steps, checkpoint_steps)

    existing_core = [
        path for path in (progress_path, metrics_path, manifest_path, checkpoint_path, final_state_path, final_pdb_path)
        if path.exists()
    ]
    existing_chunks = list(run_directory.glob(f"{prefix}_*.dcd"))
    if (existing_core or existing_chunks) and not (args.resume or args.overwrite):
        raise FileExistsError("production outputs exist; use --resume or --overwrite after review")
    if args.overwrite:
        for path in [*existing_core, *existing_chunks]:
            path.unlink()
    if args.resume and not progress_path.is_file():
        raise FileNotFoundError("--resume requires an existing production progress JSON")

    from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, Platform, XmlSerializer, unit
    from openmm.app import DCDReporter, PDBFile, Simulation
    import openmm

    start_wall = time.perf_counter()
    system_path = Path(str(inputs["system_xml"]))
    topology_path = Path(str(inputs["topology_pdb"]))
    initial_state_path = Path(str(inputs["equilibrated_state_xml"]))
    for path in (system_path, topology_path, initial_state_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    system = XmlSerializer.deserialize(system_path.read_text(encoding="utf-8"))
    system.addForce(
        MonteCarloBarostat(
            float(dynamics["pressure_bar"]) * unit.bar,
            float(dynamics["temperature_kelvin"]) * unit.kelvin,
            int(dynamics["barostat_frequency_steps"]),
        )
    )
    topology_pdb = PDBFile(str(topology_path))
    initial_state = XmlSerializer.deserialize(initial_state_path.read_text(encoding="utf-8"))
    ca_atom_indices = ca_indices(topology_pdb.topology)
    ca_reference_nm = coordinates_nm(initial_state.getPositions(), ca_atom_indices, unit)
    degrees_of_freedom = 3 * system.getNumParticles() - system.getNumConstraints() - 3
    properties = platform_properties(
        str(platform_settings["name"]),
        str(platform_settings["precision"]) if "precision" in platform_settings else None,
        int(platform_settings["cpu_threads"]) if "cpu_threads" in platform_settings else None,
    )
    platform = Platform.getPlatformByName(str(platform_settings["name"]))
    integrator = LangevinMiddleIntegrator(
        float(dynamics["temperature_kelvin"]) * unit.kelvin,
        float(dynamics["friction_per_ps"]) / unit.picosecond,
        timestep_fs * unit.femtoseconds,
    )
    integrator.setRandomNumberSeed(int(dynamics["seed"]))
    simulation = Simulation(topology_pdb.topology, system, integrator, platform, properties)
    try:
        if args.resume:
            progress = json.loads(progress_path.read_text(encoding="ascii"))
            validate_progress(progress, total_steps)
            if progress.get("status") == "ok" and int(progress["completed_steps"]) == total_steps:
                print(manifest_path.read_text(encoding="ascii"))
                return 0
            if not checkpoint_path.is_file():
                raise FileNotFoundError("production progress exists but checkpoint is missing")
            simulation.loadCheckpoint(str(checkpoint_path))
            print(f"production: resumed at {progress['completed_steps']}/{total_steps} steps", flush=True)
        else:
            progress = initialize_progress(str(config["experiment_id"]))
            simulation.context.setState(initial_state)
            atomic_write_json(progress_path, progress)
            print(f"production: start ({total_steps} steps)", flush=True)

        completed = int(progress["completed_steps"])
        while completed < total_steps:
            chunk_start = completed
            chunk_end = min(chunk_start + checkpoint_steps, total_steps)
            start_ps = chunk_start * timestep_fs / 1000.0
            end_ps = chunk_end * timestep_fs / 1000.0
            chunk_path = run_directory / chunk_filename(prefix, start_ps, end_ps)
            if chunk_path.exists():
                chunk_path.unlink()
            simulation.reporters = [
                DCDReporter(str(chunk_path), frame_steps, enforcePeriodicBox=True)
            ]
            chunk_records: list[dict[str, object]] = []
            while completed < chunk_end:
                current = min(metrics_steps, chunk_end - completed)
                simulation.step(current)
                completed += current
                state = simulation.context.getState(getPositions=True, getEnergy=True)
                elapsed_ps = completed * timestep_fs / 1000.0
                record = state_record(
                    state, "production", elapsed_ps,
                    ca_reference_nm, ca_atom_indices, degrees_of_freedom, unit,
                )
                record["total_elapsed_ps"] = round(elapsed_ps, 4)
                chunk_records.append(record)
            simulation.saveCheckpoint(str(checkpoint_path))
            progress["completed_steps"] = completed
            progress["records"].extend(chunk_records)
            progress["completed_trajectory_chunks"].append(chunk_path.as_posix())
            atomic_write_json(progress_path, progress)
            print(f"production checkpoint: {completed}/{total_steps}", flush=True)

        final_state = simulation.context.getState(
            getPositions=True, getVelocities=True, getEnergy=True
        )
        resolved_properties = {
            name: simulation.context.getPlatform().getPropertyValue(simulation.context, name)
            for name in simulation.context.getPlatform().getPropertyNames()
        }
    finally:
        del simulation
        del integrator

    final_state_path.write_text(XmlSerializer.serialize(final_state), encoding="utf-8")
    with final_pdb_path.open("w", encoding="ascii") as handle:
        PDBFile.writeFile(topology_pdb.topology, final_state.getPositions(), handle, keepIds=True)
    records = progress["records"]
    chunks = progress["completed_trajectory_chunks"]
    assert isinstance(records, list)
    assert isinstance(chunks, list)
    write_metrics(metrics_path, records)
    progress["status"] = "ok"
    atomic_write_json(progress_path, progress)
    expected_frames = total_steps // frame_steps
    manifest = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "ok",
        "operation": f"{float(dynamics['production_duration_ns']):g} ns NPT production pilot",
        "config": args.config.as_posix(),
        "openmm_version": openmm.version.version,
        "platform": str(platform_settings["name"]),
        "requested_properties": properties,
        "resolved_properties": resolved_properties,
        "system_atom_count": system.getNumParticles(),
        "total_steps": total_steps,
        "metric_record_count": len(records),
        "expected_frame_count": expected_frames,
        "completed_chunk_count": len(chunks),
        "trajectory_chunks": chunks,
        "resumed_invocation": args.resume,
        "invocation_runtime_seconds": round(time.perf_counter() - start_wall, 3),
        "final_record": records[-1],
        "outputs": {
            "progress_json": progress_path.as_posix(),
            "metrics_csv": metrics_path.as_posix(),
            "checkpoint": checkpoint_path.as_posix(),
            "final_state_xml": final_state_path.as_posix(),
            "final_pdb": final_pdb_path.as_posix(),
            "final_pdb_sha256": sha256(final_pdb_path),
        },
        "interpretation_note": config["interpretation_boundary"],
    }
    atomic_write_json(manifest_path, manifest)
    print("production: complete", flush=True)
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
