"""Align an MD trajectory and calculate backbone, pocket, and RMSF quality metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


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


def load_config(path: Path) -> dict[str, object]:
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    inputs = config["inputs"]
    outputs = config["outputs"]
    assert isinstance(inputs, dict)
    assert isinstance(outputs, dict)
    topology_path = Path(str(inputs["topology_pdb"]))
    trajectory_files = sorted(Path().glob(str(inputs["trajectory_glob"])))
    if not topology_path.is_file():
        raise FileNotFoundError(topology_path)
    if not trajectory_files:
        raise FileNotFoundError("trajectory_glob matched no DCD chunks")
    output_paths = {key: Path(str(value)) for key, value in outputs.items()}
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("trajectory QC outputs exist; use --overwrite after review")
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    import mdtraj as md

    reference = md.load(str(topology_path))
    trajectory = md.load([str(path) for path in trajectory_files], top=str(topology_path))
    expected_frames = int(config["expected_frame_count"])
    if trajectory.n_frames != expected_frames:
        raise ValueError(
            f"trajectory frame count mismatch: expected {expected_frames}, got {trajectory.n_frames}"
        )
    alignment_indices = reference.topology.select(str(config["alignment_selection"]))
    protein_indices = reference.topology.select("protein")
    ca_indices = reference.topology.select("protein and name CA")
    if len(alignment_indices) < 3 or len(ca_indices) < 3:
        raise ValueError("alignment or protein CA selection contains too few atoms")
    pocket_numbers = {int(value) for value in config["pocket_residue_numbers"]}
    pocket_ca_atoms = [
        atom for atom in reference.topology.atoms
        if atom.name == "CA" and atom.residue.is_protein and atom.residue.resSeq in pocket_numbers
    ]
    pocket_ca_indices = np.array([atom.index for atom in pocket_ca_atoms], dtype=int)
    found_pocket_numbers = {atom.residue.resSeq for atom in pocket_ca_atoms}
    if found_pocket_numbers != pocket_numbers:
        missing = sorted(pocket_numbers - found_pocket_numbers)
        raise ValueError(f"pocket CA residues missing from topology: {missing}")
    if len(pocket_ca_atoms) != len(pocket_numbers):
        raise ValueError("each requested pocket residue number must identify exactly one protein CA atom")

    trajectory.superpose(
        reference, frame=0, atom_indices=alignment_indices,
        ref_atom_indices=alignment_indices,
    )
    reference_xyz = reference.xyz[0]
    backbone_rmsd = direct_rmsd_angstrom(trajectory.xyz, reference_xyz, alignment_indices)
    ca_rmsd = direct_rmsd_angstrom(trajectory.xyz, reference_xyz, ca_indices)
    pocket_rmsd = direct_rmsd_angstrom(trajectory.xyz, reference_xyz, pocket_ca_indices)
    ca_rmsf = per_atom_rmsf_angstrom(trajectory.xyz, ca_indices)
    arrays = (backbone_rmsd, ca_rmsd, pocket_rmsd, ca_rmsf)
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise RuntimeError("trajectory QC produced a non-finite metric")

    frame_interval_ps = float(config["frame_interval_ps"])
    late_window_frame_count = int(config["late_window_frame_count"])
    frame_rows = [
        {
            "frame_index": index,
            "time_ps": round((index + 1) * frame_interval_ps, 4),
            "aligned_backbone_rmsd_angstrom": round(float(backbone_rmsd[index]), 6),
            "aligned_ca_rmsd_angstrom": round(float(ca_rmsd[index]), 6),
            "aligned_pocket_ca_rmsd_angstrom": round(float(pocket_rmsd[index]), 6),
        }
        for index in range(trajectory.n_frames)
    ]
    ca_atoms = [reference.topology.atom(int(index)) for index in ca_indices]
    residue_rows = [
        {
            "chain_index": atom.residue.chain.index,
            "chain_id": atom.residue.chain.chain_id or "",
            "residue_number": atom.residue.resSeq,
            "residue_name": atom.residue.name,
            "ca_rmsf_angstrom": round(float(ca_rmsf[index]), 6),
            "is_reference_pocket_residue": atom.residue.resSeq in pocket_numbers,
        }
        for index, atom in enumerate(ca_atoms)
    ]
    write_csv(output_paths["frame_metrics_csv"], frame_rows)
    write_csv(output_paths["residue_rmsf_csv"], residue_rows)
    protein_trajectory = trajectory.atom_slice(protein_indices)
    protein_reference = reference.atom_slice(protein_indices)
    protein_reference.save_pdb(str(output_paths["aligned_protein_pdb"]), force_overwrite=True)
    protein_trajectory.save_dcd(str(output_paths["aligned_protein_dcd"]), force_overwrite=True)

    top_flexible = sorted(residue_rows, key=lambda row: float(row["ca_rmsf_angstrom"]), reverse=True)[:10]
    pocket_residue_rows = [row for row in residue_rows if row["is_reference_pocket_residue"]]
    ca_position_by_atom_index = {
        int(atom_index): position for position, atom_index in enumerate(ca_indices)
    }
    pocket_ca_rmsf = ca_rmsf[
        [ca_position_by_atom_index[atom.index] for atom in pocket_ca_atoms]
    ]
    top_flexible_pocket = sorted(
        pocket_residue_rows,
        key=lambda row: float(row["ca_rmsf_angstrom"]),
        reverse=True,
    )[:10]
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "ok",
        "production_experiment_id": config["production_experiment_id"],
        "topology_pdb": topology_path.as_posix(),
        "topology_pdb_sha256": sha256(topology_path),
        "trajectory_chunk_count": len(trajectory_files),
        "trajectory_chunks": [
            {"path": path.as_posix(), "sha256": sha256(path)} for path in trajectory_files
        ],
        "frame_count": trajectory.n_frames,
        "frame_interval_ps": frame_interval_ps,
        "total_observed_time_ps": round(trajectory.n_frames * frame_interval_ps, 4),
        "topology_atom_count": reference.n_atoms,
        "protein_atom_count": len(protein_indices),
        "protein_ca_count": len(ca_indices),
        "pocket_ca_count": len(pocket_ca_indices),
        "alignment_selection": config["alignment_selection"],
        "aligned_backbone_rmsd_angstrom": finite_summary(backbone_rmsd),
        "aligned_ca_rmsd_angstrom": finite_summary(ca_rmsd),
        "aligned_pocket_ca_rmsd_angstrom": finite_summary(pocket_rmsd),
        "late_window": {
            "frame_count": late_window_frame_count,
            "first_frame_time_ps": round(
                (trajectory.n_frames - late_window_frame_count + 1) * frame_interval_ps, 4
            ),
            "last_frame_time_ps": round(trajectory.n_frames * frame_interval_ps, 4),
            "sample_span_ps": round((late_window_frame_count - 1) * frame_interval_ps, 4),
            "aligned_backbone_rmsd_angstrom": window_trend_summary(
                backbone_rmsd[-late_window_frame_count:], frame_interval_ps
            ),
            "aligned_ca_rmsd_angstrom": window_trend_summary(
                ca_rmsd[-late_window_frame_count:], frame_interval_ps
            ),
            "aligned_pocket_ca_rmsd_angstrom": window_trend_summary(
                pocket_rmsd[-late_window_frame_count:], frame_interval_ps
            ),
        },
        "ca_rmsf_angstrom": distribution_summary(ca_rmsf),
        "pocket_ca_rmsf_angstrom": distribution_summary(pocket_ca_rmsf),
        "top_10_ca_rmsf_residues": top_flexible,
        "top_10_pocket_ca_rmsf_residues": top_flexible_pocket,
        "outputs": {key: path.as_posix() for key, path in output_paths.items()},
        "interpretation_note": config["interpretation_boundary"],
    }
    output_paths["summary_json"].write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
