"""OpenMM molecular-dynamics helpers.

Consolidated from ``scripts/build_openmm_system.py``,
``scripts/run_openmm_equilibration.py``, ``scripts/run_openmm_production.py``,
and ``scripts/analyze_md_trajectory.py``; behavior is identical to the
originals.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

REQUIRED_TOP_LEVEL_KEYS = {
    "schema_version",
    "experiment_id",
    "starting_structure",
    "protonation",
    "force_field",
    "solvation",
    "dynamics",
    "planned_outputs",
}

REQUIRED_CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "production_experiment_id",
    "purpose",
    "inputs",
    "frame_interval_ps",
    "expected_frame_count",
    "late_window_frame_count",
    "alignment_selection",
    "pocket_residue_numbers",
    "outputs",
    "interpretation_boundary",
}
REQUIRED_INPUT_KEYS = {"topology_pdb", "trajectory_glob"}
REQUIRED_OUTPUT_KEYS = {
    "frame_metrics_csv",
    "residue_rmsf_csv",
    "summary_json",
    "aligned_protein_pdb",
    "aligned_protein_dcd",
}


def load_protocol(path: Path) -> dict[str, object]:
    """Load and validate an OpenMM build protocol JSON."""
    protocol = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be a JSON object")
    missing = sorted(REQUIRED_TOP_LEVEL_KEYS - set(protocol))
    if missing:
        raise ValueError(f"protocol is missing required keys: {', '.join(missing)}")
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: dict[str, object]) -> None:
    starting = protocol["starting_structure"]
    protonation = protocol["protonation"]
    force_field = protocol["force_field"]
    solvation = protocol["solvation"]
    dynamics = protocol["dynamics"]
    outputs = protocol["planned_outputs"]
    if not all(isinstance(section, dict) for section in (starting, protonation, force_field, solvation, dynamics, outputs)):
        raise ValueError("protocol sections must be JSON objects")
    if not str(starting.get("pdb_path", "")).endswith(".pdb"):
        raise ValueError("starting_structure.pdb_path must name a PDB file")
    if float(protonation.get("target_ph", 0.0)) <= 0.0:
        raise ValueError("protonation.target_ph must be positive")
    xml = force_field.get("protein_and_water_xml")
    if not isinstance(xml, list) or len(xml) < 2 or not all(isinstance(item, str) for item in xml):
        raise ValueError("force_field.protein_and_water_xml must contain protein and water XML files")
    for key in ("padding_nm", "ionic_strength_molar"):
        if float(solvation.get(key, -1.0)) < 0.0:
            raise ValueError(f"solvation.{key} must be non-negative")
    for key in ("temperature_kelvin", "pressure_bar", "timestep_fs", "friction_per_ps", "frame_stride_ps"):
        if float(dynamics.get(key, 0.0)) <= 0.0:
            raise ValueError(f"dynamics.{key} must be positive")
    if int(dynamics.get("seed", 0)) <= 0:
        raise ValueError("dynamics.seed must be a positive integer")


def topology_counts(topology: object) -> dict[str, int]:
    residues = list(topology.residues())
    atoms = list(topology.atoms())
    return {
        "chain_count": sum(1 for _ in topology.chains()),
        "residue_count": len(residues),
        "atom_count": len(atoms),
        "water_residue_count": sum(residue.name in {"HOH", "WAT"} for residue in residues),
        "sodium_ion_count": sum(residue.name in {"NA", "Na+"} for residue in residues),
        "chloride_ion_count": sum(residue.name in {"CL", "Cl-"} for residue in residues),
    }


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
    """Equilibration progress record."""
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
    """Validate an equilibration progress record."""
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


def initialize_production_progress(experiment_id: str) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "status": "running",
        "completed_steps": 0,
        "records": [],
        "completed_trajectory_chunks": [],
    }


def validate_production_progress(progress: dict[str, object], total_steps: int) -> None:
    completed = int(progress.get("completed_steps", -1))
    if completed < 0 or completed > total_steps:
        raise ValueError("invalid production completed_steps")
    if not isinstance(progress.get("records"), list):
        raise ValueError("production progress records must be a list")
    if not isinstance(progress.get("completed_trajectory_chunks"), list):
        raise ValueError("completed_trajectory_chunks must be a list")


def load_config(path: Path) -> dict[str, object]:
    """Load and validate a trajectory QC configuration JSON."""
    config = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(config, dict):
        raise ValueError("trajectory QC configuration must be a JSON object")
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"trajectory QC configuration is missing keys: {', '.join(missing)}")
    if float(config["frame_interval_ps"]) <= 0.0:
        raise ValueError("frame_interval_ps must be positive")
    if int(config["expected_frame_count"]) <= 0:
        raise ValueError("expected_frame_count must be positive")
    late_window_frame_count = int(config["late_window_frame_count"])
    if not 2 <= late_window_frame_count <= int(config["expected_frame_count"]):
        raise ValueError("late_window_frame_count must be between 2 and expected_frame_count")
    inputs = config["inputs"]
    outputs = config["outputs"]
    if not isinstance(inputs, dict) or not REQUIRED_INPUT_KEYS.issubset(inputs):
        raise ValueError("inputs must define topology_pdb and trajectory_glob")
    if not isinstance(outputs, dict) or not REQUIRED_OUTPUT_KEYS.issubset(outputs):
        raise ValueError("outputs is missing one or more required trajectory QC paths")
    if not isinstance(config["alignment_selection"], str) or not config["alignment_selection"].strip():
        raise ValueError("alignment_selection must be a non-empty MDTraj selection")
    pocket = config["pocket_residue_numbers"]
    if (
        not isinstance(pocket, list)
        or not pocket
        or any(not isinstance(value, int) or value <= 0 for value in pocket)
        or len(set(pocket)) != len(pocket)
    ):
        raise ValueError("pocket_residue_numbers must be a non-empty unique list of positive integers")
    return config


def direct_rmsd_angstrom(
    frames_nm: np.ndarray, reference_nm: np.ndarray, atom_indices: np.ndarray
) -> np.ndarray:
    differences = frames_nm[:, atom_indices, :] - reference_nm[atom_indices, :]
    return np.sqrt(np.mean(np.sum(differences * differences, axis=2), axis=1)) * 10.0


def per_atom_rmsf_angstrom(frames_nm: np.ndarray, atom_indices: np.ndarray) -> np.ndarray:
    selected = frames_nm[:, atom_indices, :]
    mean_coordinates = selected.mean(axis=0)
    differences = selected - mean_coordinates
    return np.sqrt(np.mean(np.sum(differences * differences, axis=2), axis=0)) * 10.0


def finite_summary(values: np.ndarray) -> dict[str, float]:
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("summary values must be a finite one-dimensional array")
    return {
        "mean": round(float(values.mean()), 6),
        "sample_sd": round(float(values.std(ddof=1)), 6) if len(values) > 1 else 0.0,
        "minimum": round(float(values.min()), 6),
        "maximum": round(float(values.max()), 6),
        "final": round(float(values[-1]), 6),
    }


def distribution_summary(values: np.ndarray) -> dict[str, float]:
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("distribution values must be a finite one-dimensional array")
    return {
        "mean": round(float(values.mean()), 6),
        "sample_sd": round(float(values.std(ddof=1)), 6) if len(values) > 1 else 0.0,
        "median": round(float(np.median(values)), 6),
        "percentile_95": round(float(np.percentile(values, 95)), 6),
        "minimum": round(float(values.min()), 6),
        "maximum": round(float(values.max()), 6),
    }


def window_trend_summary(values: np.ndarray, frame_interval_ps: float) -> dict[str, float]:
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("window values must contain at least two finite observations")
    time_ns = np.arange(len(values), dtype=float) * frame_interval_ps / 1000.0
    slope = float(np.polyfit(time_ns, values, 1)[0])
    return {
        "mean": round(float(values.mean()), 6),
        "sample_sd": round(float(values.std(ddof=1)), 6),
        "minimum": round(float(values.min()), 6),
        "maximum": round(float(values.max()), 6),
        "first": round(float(values[0]), 6),
        "final": round(float(values[-1]), 6),
        "final_minus_first": round(float(values[-1] - values[0]), 6),
        "linear_slope_angstrom_per_ns": round(slope, 6),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
