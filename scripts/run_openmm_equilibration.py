"""Run resumable minimization, NVT, and NPT equilibration with audited checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

try:
    from .run_openmm_equilibration_smoke import (
        ca_indices,
        coordinates_nm,
        load_smoke_config,
        periodic_volume_nm3,
        platform_properties,
        state_record,
        steps_for_duration,
    )
except ImportError:
    from run_openmm_equilibration_smoke import (
        ca_indices,
        coordinates_nm,
        load_smoke_config,
        periodic_volume_nm3,
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


def atomic_write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    temporary.replace(path)


def write_metrics(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        raise ValueError("cannot write an empty metrics table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def output_paths(outputs: dict[str, object]) -> dict[str, Path]:
    return {key: Path(str(value)) for key, value in outputs.items()}


def initialize_progress(experiment_id: str, minimized_record: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "status": "running",
        "phase": "NVT",
        "nvt_completed_steps": 0,
        "npt_completed_steps": 0,
        "records": [minimized_record],
    }


def validate_progress(progress: dict[str, object]) -> None:
    if progress.get("phase") not in {"NVT", "NPT", "complete"}:
        raise ValueError("progress phase must be NVT, NPT, or complete")
    for key in ("nvt_completed_steps", "npt_completed_steps"):
        if int(progress.get(key, -1)) < 0:
            raise ValueError(f"invalid progress counter: {key}")
    if not isinstance(progress.get("records"), list):
        raise ValueError("progress records must be a list")


def add_time_fields(record: dict[str, object], phase_offset_ps: float) -> dict[str, object]:
    result = dict(record)
    result["total_elapsed_ps"] = round(phase_offset_ps + float(record["elapsed_ps"]), 4)
    return result


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
    paths = output_paths(outputs)
    progress_path = paths["progress_json"]
    existing = [path for path in paths.values() if path.exists()]
    if existing and not (args.resume or args.overwrite):
        raise FileExistsError("equilibration outputs exist; use --resume or --overwrite after review")
    if args.overwrite:
        for path in existing:
            path.unlink()
    if args.resume and not progress_path.is_file():
        raise FileNotFoundError("--resume requires an existing progress JSON")

    from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, Platform, XmlSerializer, unit
    from openmm.app import PDBFile, Simulation
    import openmm

    start_wall = time.perf_counter()
    system_path = Path(str(inputs["system_xml"]))
    positions_path = Path(str(inputs["solvated_pdb"]))
    if not system_path.is_file() or not positions_path.is_file():
        raise FileNotFoundError("system XML or solvated PDB input is missing")
    system = XmlSerializer.deserialize(system_path.read_text(encoding="utf-8"))
    pdb = PDBFile(str(positions_path))
    timestep_fs = float(dynamics["timestep_fs"])
    nvt_duration_ps = float(dynamics["nvt_duration_ps"])
    npt_duration_ps = float(dynamics["npt_duration_ps"])
    nvt_total_steps = steps_for_duration(nvt_duration_ps, timestep_fs)
    npt_total_steps = steps_for_duration(npt_duration_ps, timestep_fs)
    checkpoint_steps = steps_for_duration(float(dynamics["checkpoint_interval_ps"]), timestep_fs)
    seed = int(dynamics["seed"])
    properties = platform_properties(
        str(platform_settings["name"]),
        str(platform_settings["precision"]) if "precision" in platform_settings else None,
        int(platform_settings["cpu_threads"]) if "cpu_threads" in platform_settings else None,
    )
    platform = Platform.getPlatformByName(str(platform_settings["name"]))
    ca_atom_indices = ca_indices(pdb.topology)
    if len(ca_atom_indices) < 3:
        raise RuntimeError("could not identify enough protein CA atoms")
    degrees_of_freedom = 3 * system.getNumParticles() - system.getNumConstraints() - 3

    def new_integrator(run_seed: int) -> object:
        integrator = LangevinMiddleIntegrator(
            float(dynamics["temperature_kelvin"]) * unit.kelvin,
            float(dynamics["friction_per_ps"]) / unit.picosecond,
            timestep_fs * unit.femtoseconds,
        )
        integrator.setRandomNumberSeed(run_seed)
        return integrator

    if args.resume:
        progress = json.loads(progress_path.read_text(encoding="ascii"))
        validate_progress(progress)
        if progress["phase"] == "complete":
            print(paths["manifest"].read_text(encoding="ascii"))
            return 0
        minimized_state = XmlSerializer.deserialize(paths["minimized_state_xml"].read_text(encoding="utf-8"))
    else:
        nvt_integrator = new_integrator(seed)
        nvt_simulation = Simulation(pdb.topology, system, nvt_integrator, platform, properties)
        try:
            print("minimization: start", flush=True)
            nvt_simulation.context.setPositions(pdb.positions)
            nvt_simulation.minimizeEnergy(
                tolerance=float(dynamics["minimization_tolerance_kj_per_mol_nm"])
                * unit.kilojoule_per_mole / unit.nanometer,
                maxIterations=int(dynamics["minimization_max_iterations"]),
            )
            minimized_state = nvt_simulation.context.getState(getPositions=True, getEnergy=True)
            paths["minimized_state_xml"].parent.mkdir(parents=True, exist_ok=True)
            paths["minimized_state_xml"].write_text(
                XmlSerializer.serialize(minimized_state), encoding="utf-8"
            )
            minimized_potential = minimized_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            minimized_record = {
                "phase": "minimized",
                "elapsed_ps": 0.0,
                "potential_energy_kj_per_mol": round(minimized_potential, 6),
                "kinetic_energy_kj_per_mol": 0.0,
                "instantaneous_temperature_kelvin": 0.0,
                "periodic_box_volume_nm3": round(periodic_volume_nm3(minimized_state, unit), 6),
                "protein_ca_centered_rmsd_to_minimized_start_angstrom": 0.0,
                "total_elapsed_ps": 0.0,
            }
            progress = initialize_progress(str(config["experiment_id"]), minimized_record)
            atomic_write_json(progress_path, progress)
        finally:
            del nvt_simulation
            del nvt_integrator
        print("minimization: complete", flush=True)

    ca_reference_nm = coordinates_nm(minimized_state.getPositions(), ca_atom_indices, unit)

    if progress["phase"] == "NVT":
        nvt_integrator = new_integrator(seed)
        nvt_simulation = Simulation(pdb.topology, system, nvt_integrator, platform, properties)
        try:
            nvt_completed = int(progress["nvt_completed_steps"])
            if nvt_completed:
                if not paths["nvt_checkpoint"].is_file():
                    raise FileNotFoundError("NVT progress exists but the checkpoint is missing")
                nvt_simulation.loadCheckpoint(str(paths["nvt_checkpoint"]))
                print(f"NVT: resumed at {nvt_completed}/{nvt_total_steps} steps", flush=True)
            else:
                nvt_simulation.context.setPositions(minimized_state.getPositions())
                nvt_simulation.context.setVelocitiesToTemperature(
                    float(dynamics["temperature_kelvin"]) * unit.kelvin, seed
                )
                print(f"NVT: start ({nvt_total_steps} steps)", flush=True)
            while nvt_completed < nvt_total_steps:
                current = min(checkpoint_steps, nvt_total_steps - nvt_completed)
                nvt_simulation.step(current)
                nvt_completed += current
                state = nvt_simulation.context.getState(getPositions=True, getEnergy=True)
                record = state_record(
                    state, "NVT", nvt_completed * timestep_fs / 1000.0,
                    ca_reference_nm, ca_atom_indices, degrees_of_freedom, unit,
                )
                progress["records"].append(add_time_fields(record, 0.0))
                progress["nvt_completed_steps"] = nvt_completed
                nvt_simulation.saveCheckpoint(str(paths["nvt_checkpoint"]))
                atomic_write_json(progress_path, progress)
                print(f"NVT checkpoint: {nvt_completed}/{nvt_total_steps}", flush=True)
            nvt_final_state = nvt_simulation.context.getState(getPositions=True, getVelocities=True)
            paths["nvt_final_state_xml"].write_text(
                XmlSerializer.serialize(nvt_final_state), encoding="utf-8"
            )
            progress["phase"] = "NPT"
            atomic_write_json(progress_path, progress)
        finally:
            del nvt_simulation
            del nvt_integrator

    system.addForce(
        MonteCarloBarostat(
            float(dynamics["pressure_bar"]) * unit.bar,
            float(dynamics["temperature_kelvin"]) * unit.kelvin,
            int(dynamics["barostat_frequency_steps"]),
        )
    )
    npt_integrator = new_integrator(seed + 1)
    npt_simulation = Simulation(pdb.topology, system, npt_integrator, platform, properties)
    try:
        npt_completed = int(progress["npt_completed_steps"])
        if npt_completed:
            if not paths["npt_checkpoint"].is_file():
                raise FileNotFoundError("NPT progress exists but the checkpoint is missing")
            npt_simulation.loadCheckpoint(str(paths["npt_checkpoint"]))
            print(f"NPT: resumed at {npt_completed}/{npt_total_steps} steps", flush=True)
        else:
            nvt_final_state = XmlSerializer.deserialize(
                paths["nvt_final_state_xml"].read_text(encoding="utf-8")
            )
            npt_simulation.context.setPositions(nvt_final_state.getPositions())
            npt_simulation.context.setVelocities(nvt_final_state.getVelocities())
            print(f"NPT: start ({npt_total_steps} steps)", flush=True)
        while npt_completed < npt_total_steps:
            current = min(checkpoint_steps, npt_total_steps - npt_completed)
            npt_simulation.step(current)
            npt_completed += current
            state = npt_simulation.context.getState(getPositions=True, getEnergy=True)
            record = state_record(
                state, "NPT", npt_completed * timestep_fs / 1000.0,
                ca_reference_nm, ca_atom_indices, degrees_of_freedom, unit,
            )
            progress["records"].append(add_time_fields(record, nvt_duration_ps))
            progress["npt_completed_steps"] = npt_completed
            npt_simulation.saveCheckpoint(str(paths["npt_checkpoint"]))
            atomic_write_json(progress_path, progress)
            print(f"NPT checkpoint: {npt_completed}/{npt_total_steps}", flush=True)
        final_state = npt_simulation.context.getState(
            getPositions=True, getVelocities=True, getEnergy=True
        )
        resolved_properties = {
            name: npt_simulation.context.getPlatform().getPropertyValue(npt_simulation.context, name)
            for name in npt_simulation.context.getPlatform().getPropertyNames()
        }
    finally:
        del npt_simulation
        del npt_integrator

    paths["final_state_xml"].write_text(XmlSerializer.serialize(final_state), encoding="utf-8")
    with paths["final_pdb"].open("w", encoding="ascii") as handle:
        PDBFile.writeFile(pdb.topology, final_state.getPositions(), handle, keepIds=True)
    records = progress["records"]
    assert isinstance(records, list)
    write_metrics(paths["metrics_csv"], records)
    progress["phase"] = "complete"
    progress["status"] = "ok"
    atomic_write_json(progress_path, progress)
    manifest = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "ok",
        "operation": f"bounded minimization, {nvt_duration_ps:g} ps NVT, and {npt_duration_ps:g} ps NPT; no production trajectory",
        "config": args.config.as_posix(),
        "openmm_version": openmm.version.version,
        "platform": str(platform_settings["name"]),
        "requested_properties": properties,
        "resolved_properties": resolved_properties,
        "system_atom_count": system.getNumParticles(),
        "protein_ca_atom_count": len(ca_atom_indices),
        "nvt_steps": nvt_total_steps,
        "npt_steps": npt_total_steps,
        "checkpoint_interval_steps": checkpoint_steps,
        "record_count": len(records),
        "resumed_invocation": args.resume,
        "invocation_runtime_seconds": round(time.perf_counter() - start_wall, 3),
        "final_record": records[-1],
        "outputs": {
            "progress_json": paths["progress_json"].as_posix(),
            "metrics_csv": paths["metrics_csv"].as_posix(),
            "final_state_xml": paths["final_state_xml"].as_posix(),
            "final_pdb": paths["final_pdb"].as_posix(),
            "final_pdb_sha256": sha256(paths["final_pdb"]),
        },
        "interpretation_note": config["interpretation_boundary"],
    }
    atomic_write_json(paths["manifest"], manifest)
    print("equilibration: complete", flush=True)
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
