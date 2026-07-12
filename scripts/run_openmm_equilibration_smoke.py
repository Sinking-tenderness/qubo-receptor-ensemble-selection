"""Run a short minimization, NVT, and NPT OpenMM stability smoke test."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

GAS_CONSTANT_KJ_PER_MOL_K = 0.00831446261815324
WATER_NAMES = {"HOH", "WAT"}
REQUIRED_CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "parent_protocol",
    "inputs",
    "platform",
    "dynamics",
    "outputs",
    "interpretation_boundary",
}


def load_smoke_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(config, dict):
        raise ValueError("smoke configuration must be a JSON object")
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"smoke configuration is missing required keys: {', '.join(missing)}")
    for name in ("inputs", "platform", "dynamics", "outputs"):
        if not isinstance(config[name], dict):
            raise ValueError(f"smoke configuration section must be an object: {name}")
    return config


def steps_for_duration(duration_ps: float, timestep_fs: float) -> int:
    if duration_ps <= 0.0 or timestep_fs <= 0.0:
        raise ValueError("duration_ps and timestep_fs must be positive")
    raw_steps = duration_ps * 1000.0 / timestep_fs
    rounded = round(raw_steps)
    if not math.isclose(raw_steps, rounded, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("duration_ps must be an exact multiple of timestep_fs")
    return int(rounded)


def temperature_from_kinetic_energy(kinetic_kj_per_mol: float, degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    return 2.0 * kinetic_kj_per_mol / (degrees_of_freedom * GAS_CONSTANT_KJ_PER_MOL_K)


def centered_rmsd(reference_nm: np.ndarray, mobile_nm: np.ndarray) -> float:
    if reference_nm.shape != mobile_nm.shape or reference_nm.ndim != 2 or reference_nm.shape[1] != 3:
        raise ValueError("reference and mobile coordinates must both have shape (n, 3)")
    if len(reference_nm) < 3:
        raise ValueError("at least three coordinates are required for centered RMSD")
    reference_center = reference_nm.mean(axis=0)
    mobile_center = mobile_nm.mean(axis=0)
    reference_zero = reference_nm - reference_center
    mobile_zero = mobile_nm - mobile_center
    return float(np.sqrt(np.mean(np.sum((mobile_zero - reference_zero) ** 2, axis=1))))


def ca_indices(topology: object) -> list[int]:
    return [
        atom.index for atom in topology.atoms()
        if atom.name == "CA" and atom.residue.name not in WATER_NAMES
    ]


def coordinates_nm(positions: object, indices: list[int], unit: object) -> np.ndarray:
    return np.array(
        [[component for component in positions[index].value_in_unit(unit.nanometer)] for index in indices],
        dtype=float,
    )


def periodic_volume_nm3(state: object, unit: object) -> float:
    vectors = state.getPeriodicBoxVectors()
    matrix = np.array(
        [[component for component in vector.value_in_unit(unit.nanometer)] for vector in vectors],
        dtype=float,
    )
    a, b, c = matrix
    determinant = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    return float(abs(determinant))


def state_record(
    state: object,
    phase: str,
    elapsed_ps: float,
    ca_reference_nm: np.ndarray,
    ca_atom_indices: list[int],
    degrees_of_freedom: int,
    unit: object,
) -> dict[str, object]:
    potential = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    ca_rmsd_angstrom = centered_rmsd(
        ca_reference_nm,
        coordinates_nm(state.getPositions(), ca_atom_indices, unit),
    ) * 10.0
    volume = periodic_volume_nm3(state, unit)
    values = (potential, kinetic, ca_rmsd_angstrom, volume)
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"non-finite state metric in {phase}")
    return {
        "phase": phase,
        "elapsed_ps": round(elapsed_ps, 4),
        "potential_energy_kj_per_mol": round(potential, 6),
        "kinetic_energy_kj_per_mol": round(kinetic, 6),
        "instantaneous_temperature_kelvin": round(
            temperature_from_kinetic_energy(kinetic, degrees_of_freedom), 4
        ),
        "periodic_box_volume_nm3": round(volume, 6),
        "protein_ca_centered_rmsd_to_minimized_start_angstrom": round(ca_rmsd_angstrom, 6),
    }


def run_phase(
    simulation: object,
    phase: str,
    total_steps: int,
    checkpoint_steps: int,
    timestep_fs: float,
    ca_reference_nm: np.ndarray,
    ca_atom_indices: list[int],
    degrees_of_freedom: int,
    unit: object,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    completed = 0
    while completed < total_steps:
        current_steps = min(checkpoint_steps, total_steps - completed)
        simulation.step(current_steps)
        completed += current_steps
        state = simulation.context.getState(getPositions=True, getEnergy=True)
        records.append(
            state_record(
                state, phase, completed * timestep_fs / 1000.0,
                ca_reference_nm, ca_atom_indices, degrees_of_freedom, unit,
            )
        )
        print(f"{phase} checkpoint: {completed}/{total_steps} steps", flush=True)
    return records


def platform_properties(platform_name: str, precision: str | None, cpu_threads: int | None) -> dict[str, str]:
    if platform_name in {"OpenCL", "CUDA"}:
        if precision not in {"single", "mixed", "double"}:
            raise ValueError("GPU precision must be single, mixed, or double")
        return {"Precision": str(precision)}
    if platform_name == "CPU":
        if cpu_threads is None or cpu_threads < 1:
            raise ValueError("CPU smoke runs require platform.cpu_threads >= 1")
        return {"Threads": str(cpu_threads)}
    raise ValueError("platform.name must be CPU, OpenCL, or CUDA")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_smoke_config(args.config)
    inputs = config["inputs"]
    platform_settings = config["platform"]
    dynamics = config["dynamics"]
    outputs = config["outputs"]
    assert isinstance(inputs, dict)
    assert isinstance(platform_settings, dict)
    assert isinstance(dynamics, dict)
    assert isinstance(outputs, dict)
    manifest_path = Path(str(outputs["manifest"]))
    final_pdb_path = Path(str(outputs["final_pdb"]))
    if (manifest_path.exists() or final_pdb_path.exists()) and not args.overwrite:
        raise FileExistsError("smoke outputs exist; use --overwrite after review")

    from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, Platform, XmlSerializer, unit
    from openmm.app import PDBFile, Simulation
    import openmm

    system = XmlSerializer.deserialize(Path(str(inputs["system_xml"])).read_text(encoding="utf-8"))
    pdb = PDBFile(str(inputs["solvated_pdb"]))
    timestep_fs = float(dynamics["timestep_fs"])
    seed = int(dynamics["seed"])
    nvt_steps = steps_for_duration(float(dynamics["nvt_duration_ps"]), timestep_fs)
    npt_steps = steps_for_duration(float(dynamics["npt_duration_ps"]), timestep_fs)
    checkpoint_steps = steps_for_duration(float(dynamics["checkpoint_interval_ps"]), timestep_fs)
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

    def integrator_for(run_seed: int) -> object:
        integrator = LangevinMiddleIntegrator(
            float(dynamics["temperature_kelvin"]) * unit.kelvin,
            float(dynamics["friction_per_ps"]) / unit.picosecond,
            timestep_fs * unit.femtoseconds,
        )
        integrator.setRandomNumberSeed(run_seed)
        return integrator

    nvt_integrator = integrator_for(seed)
    nvt = Simulation(pdb.topology, system, nvt_integrator, platform, properties)
    try:
        print("minimization: start", flush=True)
        nvt.context.setPositions(pdb.positions)
        nvt.minimizeEnergy(
            tolerance=float(dynamics["minimization_tolerance_kj_per_mol_nm"]) * unit.kilojoule_per_mole / unit.nanometer,
            maxIterations=int(dynamics["minimization_max_iterations"]),
        )
        minimized_state = nvt.context.getState(getPositions=True, getEnergy=True)
        print("minimization: complete", flush=True)
        print("minimization audit: CA coordinates", flush=True)
        ca_reference_nm = coordinates_nm(minimized_state.getPositions(), ca_atom_indices, unit)
        print("minimization audit: energy and volume", flush=True)
        minimized_potential = minimized_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        minimized_volume = periodic_volume_nm3(minimized_state, unit)
        if not all(math.isfinite(value) for value in (minimized_potential, minimized_volume)):
            raise RuntimeError("minimization produced a non-finite state metric")
        minimized_record = {
            "phase": "minimized",
            "elapsed_ps": 0.0,
            "potential_energy_kj_per_mol": round(minimized_potential, 6),
            "kinetic_energy_kj_per_mol": 0.0,
            "instantaneous_temperature_kelvin": 0.0,
            "periodic_box_volume_nm3": round(minimized_volume, 6),
            "protein_ca_centered_rmsd_to_minimized_start_angstrom": 0.0,
        }
        print("minimization audit: complete", flush=True)
        nvt.context.setVelocitiesToTemperature(float(dynamics["temperature_kelvin"]) * unit.kelvin, seed)
        print(f"NVT: start ({nvt_steps} steps)", flush=True)
        nvt_records = run_phase(
            nvt, "NVT", nvt_steps, checkpoint_steps, timestep_fs,
            ca_reference_nm, ca_atom_indices, degrees_of_freedom, unit,
        )
        nvt_final = nvt.context.getState(getPositions=True, getVelocities=True)
    finally:
        del nvt
        del nvt_integrator

    system.addForce(
        MonteCarloBarostat(
            float(dynamics["pressure_bar"]) * unit.bar,
            float(dynamics["temperature_kelvin"]) * unit.kelvin,
            int(dynamics["barostat_frequency_steps"]),
        )
    )
    npt_integrator = integrator_for(seed + 1)
    npt = Simulation(pdb.topology, system, npt_integrator, platform, properties)
    try:
        print(f"NPT: start ({npt_steps} steps)", flush=True)
        npt.context.setPositions(nvt_final.getPositions())
        npt.context.setVelocities(nvt_final.getVelocities())
        npt_records = run_phase(
            npt, "NPT", npt_steps, checkpoint_steps, timestep_fs,
            ca_reference_nm, ca_atom_indices, degrees_of_freedom, unit,
        )
        final_state = npt.context.getState(getPositions=True, getEnergy=True)
        resolved_properties = {
            name: npt.context.getPlatform().getPropertyValue(npt.context, name)
            for name in npt.context.getPlatform().getPropertyNames()
        }
    finally:
        del npt
        del npt_integrator

    final_pdb_path.parent.mkdir(parents=True, exist_ok=True)
    with final_pdb_path.open("w", encoding="ascii") as handle:
        PDBFile.writeFile(pdb.topology, final_state.getPositions(), handle, keepIds=True)
    records = [minimized_record, *nvt_records, *npt_records]
    manifest = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "ok",
        "operation": (
            "bounded minimization plus "
            f"{float(dynamics['nvt_duration_ps']):g} ps NVT and "
            f"{float(dynamics['npt_duration_ps']):g} ps NPT stability smoke; "
            "no production trajectory"
        ),
        "config": args.config.as_posix(),
        "parent_protocol": config["parent_protocol"],
        "openmm_version": openmm.version.version,
        "platform": str(platform_settings["name"]),
        "requested_properties": properties,
        "resolved_properties": resolved_properties,
        "system_atom_count": system.getNumParticles(),
        "protein_ca_atom_count": len(ca_atom_indices),
        "degrees_of_freedom_estimate": degrees_of_freedom,
        "records": records,
        "final_potential_energy_kj_per_mol": round(
            final_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole), 6
        ),
        "final_pdb": final_pdb_path.as_posix(),
        "interpretation_note": config["interpretation_boundary"],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print("smoke: complete", flush=True)
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
