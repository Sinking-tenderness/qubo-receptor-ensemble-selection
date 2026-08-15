"""Freeze PPARG multi-start MD structures and generate per-start OpenMM configs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage21_structure_aware_qubo import (
    descriptor,
    file_sha256,
    read_csv,
    read_json,
    rooted,
    write_csv,
    write_json,
)


def config_paths(run_root: str, config_root: str, conformer_id: str) -> dict[str, str]:
    slug = conformer_id.lower()
    base = f"{run_root}/{slug}"
    configs = f"{config_root}/{slug}"
    return {
        "slug": slug,
        "run_directory": base,
        "protocol_config": f"{configs}_protocol.json",
        "equilibration_config": f"{configs}_equilibration.json",
        "production_config": f"{configs}_production.json",
        "trajectory_qc_config": f"{configs}_trajectory_qc.json",
        "system_manifest": f"{base}/system_build_manifest.json",
        "solvated_pdb": f"{base}/solvated_system.pdb",
        "system_xml": f"{base}/system.xml",
        "equilibration_manifest": f"{base}/equilibration_manifest.json",
        "equilibration_progress": f"{base}/equilibration_progress.json",
        "equilibrated_state": f"{base}/equilibrated_state.xml",
        "equilibrated_pdb": f"{base}/equilibrated_state.pdb",
        "production_directory": f"{base}/production",
        "production_manifest": f"{base}/production/manifest.json",
        "production_progress": f"{base}/production/progress.json",
        "trajectory_qc_summary": f"{base}/trajectory_qc/summary.json",
        "aligned_protein_pdb": f"{base}/trajectory_qc/aligned_protein_topology.pdb",
        "aligned_protein_dcd": f"{base}/trajectory_qc/aligned_protein.dcd",
    }


def protocol(master: dict[str, Any], conformer_id: str, pdb_path: str, paths: dict[str, str], seed: int) -> dict[str, Any]:
    system = master["system"]
    dynamics = master["dynamics"]
    prefix = master.get("run_experiment_prefix", "stage28-pparg")
    return {
        "schema_version": "1.0",
        "experiment_id": f"{prefix}-{paths['slug']}-system-v1",
        "starting_structure": {
            "conformer_id": conformer_id,
            "pdb_path": pdb_path,
            "chain_id": master["target"]["protein_chain_id"],
            "state_description": "pre-existing hard-gate-passing PPARG max-min starting structure",
            "ligands_retained": False,
            "crystal_waters_retained": False,
            "metals_retained": False,
            "cofactors_retained": False,
        },
        "protonation": {
            "target_ph": system["target_ph"],
            "method": "OpenMM Modeller.addHydrogens with force-field templates",
            "limitation": "Model-based protonation; no claim of exhaustive microstate sampling.",
        },
        "force_field": {
            "protein_and_water_xml": system["protein_and_water_xml"],
            "water_geometry_model": system["water_geometry_model"],
            "nonbonded_method": "PME",
            "nonbonded_cutoff_nm": system["nonbonded_cutoff_nm"],
            "constraints": "HBonds",
        },
        "solvation": {
            "padding_nm": system["padding_nm"],
            "ionic_strength_molar": system["ionic_strength_molar"],
            "positive_ion": system["positive_ion"],
            "negative_ion": system["negative_ion"],
            "neutralize": system["neutralize"],
        },
        "dynamics": {
            "temperature_kelvin": dynamics["temperature_kelvin"],
            "pressure_bar": dynamics["pressure_bar"],
            "timestep_fs": dynamics["timestep_fs"],
            "friction_per_ps": dynamics["friction_per_ps"],
            "minimization_tolerance_kj_per_mol_nm": dynamics["minimization_tolerance_kj_per_mol_nm"],
            "minimization_max_iterations": dynamics["minimization_max_iterations"],
            "nvt_duration_ps": dynamics["nvt_duration_ps"],
            "npt_duration_ps": dynamics["npt_duration_ps"],
            "pilot_production_duration_ns": dynamics["production_duration_ns_per_start"],
            "frame_stride_ps": dynamics["production_frame_interval_ps"],
            "seed": seed,
        },
        "planned_outputs": {
            "system_manifest": paths["system_manifest"],
            "solvated_pdb": paths["solvated_pdb"],
            "system_xml": paths["system_xml"],
            "trajectory_directory": paths["production_directory"],
        },
        "sampling_boundary": master["interpretation_boundary"],
    }


def equilibration(master: dict[str, Any], paths: dict[str, str], seed: int) -> dict[str, Any]:
    dynamics = master["dynamics"]
    base = paths["run_directory"]
    prefix = master.get("run_experiment_prefix", "stage28-pparg")
    return {
        "schema_version": "1.0",
        "experiment_id": f"{prefix}-{paths['slug']}-equilibration-v1",
        "parent_protocol": paths["protocol_config"],
        "purpose": "Resumable bounded minimization, NVT, and NPT equilibration before the Stage28 structural-pool trajectory.",
        "inputs": {"system_xml": paths["system_xml"], "solvated_pdb": paths["solvated_pdb"]},
        "platform": {"name": dynamics["platform"], "precision": dynamics["precision"]},
        "dynamics": {
            "temperature_kelvin": dynamics["temperature_kelvin"],
            "pressure_bar": dynamics["pressure_bar"],
            "timestep_fs": dynamics["timestep_fs"],
            "friction_per_ps": dynamics["friction_per_ps"],
            "minimization_tolerance_kj_per_mol_nm": dynamics["minimization_tolerance_kj_per_mol_nm"],
            "minimization_max_iterations": dynamics["minimization_max_iterations"],
            "nvt_duration_ps": dynamics["nvt_duration_ps"],
            "npt_duration_ps": dynamics["npt_duration_ps"],
            "checkpoint_interval_ps": dynamics["equilibration_checkpoint_interval_ps"],
            "barostat_frequency_steps": dynamics["barostat_frequency_steps"],
            "seed": seed,
        },
        "outputs": {
            "progress_json": paths["equilibration_progress"],
            "metrics_csv": f"{base}/equilibration_metrics.csv",
            "manifest": paths["equilibration_manifest"],
            "minimized_state_xml": f"{base}/minimized_state.xml",
            "nvt_checkpoint": f"{base}/nvt.chk",
            "nvt_final_state_xml": f"{base}/nvt_final_state.xml",
            "npt_checkpoint": f"{base}/npt.chk",
            "final_state_xml": paths["equilibrated_state"],
            "final_pdb": paths["equilibrated_pdb"],
        },
        "interpretation_boundary": master["interpretation_boundary"],
    }


def production(master: dict[str, Any], paths: dict[str, str], seed: int) -> dict[str, Any]:
    dynamics = master["dynamics"]
    directory = paths["production_directory"]
    prefix = master.get("run_experiment_prefix", "stage28-pparg")
    return {
        "schema_version": "1.0",
        "experiment_id": f"{prefix}-{paths['slug']}-production-v1",
        "parent_protocol": paths["protocol_config"],
        "purpose": "Resumable short NPT production for a structure-only PPARG solver-scaling pool.",
        "inputs": {"system_xml": paths["system_xml"], "topology_pdb": paths["equilibrated_pdb"], "equilibrated_state_xml": paths["equilibrated_state"]},
        "platform": {"name": dynamics["platform"], "precision": dynamics["precision"]},
        "dynamics": {
            "temperature_kelvin": dynamics["temperature_kelvin"],
            "pressure_bar": dynamics["pressure_bar"],
            "timestep_fs": dynamics["timestep_fs"],
            "friction_per_ps": dynamics["friction_per_ps"],
            "production_duration_ns": dynamics["production_duration_ns_per_start"],
            "metrics_interval_ps": dynamics["production_metrics_interval_ps"],
            "frame_interval_ps": dynamics["production_frame_interval_ps"],
            "checkpoint_interval_ps": dynamics["production_checkpoint_interval_ps"],
            "barostat_frequency_steps": dynamics["barostat_frequency_steps"],
            "seed": seed + 1,
        },
        "outputs": {
            "run_directory": directory,
            "progress_json": paths["production_progress"],
            "metrics_csv": f"{directory}/metrics.csv",
            "manifest": paths["production_manifest"],
            "checkpoint": f"{directory}/production.chk",
            "final_state_xml": f"{directory}/final_state.xml",
            "final_pdb": f"{directory}/final_state.pdb",
            "trajectory_prefix": f"{paths['slug']}_production",
        },
        "interpretation_boundary": master["interpretation_boundary"],
    }


def trajectory_qc(master: dict[str, Any], paths: dict[str, str]) -> dict[str, Any]:
    expected = int(master["sampling"]["expected_frames_per_start"])
    prefix = master.get("run_experiment_prefix", "stage28-pparg")
    return {
        "schema_version": "1.0",
        "experiment_id": f"{prefix}-{paths['slug']}-trajectory-qc-v1",
        "production_experiment_id": f"{prefix}-{paths['slug']}-production-v1",
        "purpose": "Align and quality-control one Stage28 PPARG trajectory before cross-start feature aggregation.",
        "inputs": {"topology_pdb": paths["equilibrated_pdb"], "trajectory_glob": f"{paths['production_directory']}/*.dcd"},
        "frame_interval_ps": master["dynamics"]["production_frame_interval_ps"],
        "expected_frame_count": expected,
        "late_window_frame_count": max(10, expected // 4),
        "alignment_selection": "protein and backbone",
        "pocket_residue_numbers": master["target"]["pocket_residue_numbers"],
        "outputs": {
            "frame_metrics_csv": f"{paths['run_directory']}/trajectory_qc/frame_metrics.csv",
            "residue_rmsf_csv": f"{paths['run_directory']}/trajectory_qc/residue_ca_rmsf.csv",
            "summary_json": paths["trajectory_qc_summary"],
            "aligned_protein_pdb": paths["aligned_protein_pdb"],
            "aligned_protein_dcd": paths["aligned_protein_dcd"],
        },
        "interpretation_boundary": master["interpretation_boundary"],
    }


def prepare(config_path: Path, root: Path, overwrite: bool = False) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    master = read_json(config_path)
    source_path = rooted(root, master["target"]["source_selection_manifest"])
    source_rows = sorted(read_csv(source_path), key=lambda row: int(row["selection_rank"]))
    count = int(master["target"]["starting_structure_count"])
    selected = [
        row for row in source_rows
        if int(float(row.get("global_incomplete_standard_amino_acid_residue_count", "0") or 0)) == 0
    ][:count]
    if len(selected) != count:
        raise ValueError("fewer than eight hard-gate-passing max-min starts")
    aligned_directory = rooted(root, master["target"]["aligned_structure_directory"])
    config_directory = rooted(root, master["runtime"]["generated_config_directory"])
    manifest_path = rooted(root, master["runtime"]["start_manifest"])
    result_path = rooted(root, master["outputs"]["preparation_result_json"])
    existing = [path for path in (manifest_path, result_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Stage28 preparation outputs exist: {existing}")
    config_directory.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    generated_paths: list[Path] = []
    for position, row in enumerate(selected):
        conformer_id = row["conformer_id"]
        matches = sorted(aligned_directory.glob(f"{conformer_id}*.pdb"))
        if len(matches) != 1:
            raise ValueError(f"{conformer_id}: expected one aligned PDB, found {len(matches)}")
        pdb_path = matches[0]
        if int(float(row.get("global_incomplete_standard_amino_acid_residue_count", "0") or 0)) != 0:
            raise ValueError(f"{conformer_id}: globally incomplete start is forbidden")
        paths = config_paths(master["runtime"]["run_root"], master["runtime"]["generated_config_directory"], conformer_id)
        seed = int(master["dynamics"]["base_seed"]) + position * 100
        payloads = {
            paths["protocol_config"]: protocol(master, conformer_id, pdb_path.relative_to(root).as_posix(), paths, seed),
            paths["equilibration_config"]: equilibration(master, paths, seed + 10),
            paths["production_config"]: production(master, paths, seed + 20),
            paths["trajectory_qc_config"]: trajectory_qc(master, paths),
        }
        for relative, payload in payloads.items():
            output = rooted(root, relative)
            write_json(output, payload)
            generated_paths.append(output)
        manifest_rows.append({
            "start_index": position,
            "selection_rank": int(row["selection_rank"]),
            "conformer_id": conformer_id,
            "starting_pdb": pdb_path.relative_to(root).as_posix(),
            "starting_pdb_sha256": file_sha256(pdb_path),
            "seed": seed,
            "protocol_config": paths["protocol_config"],
            "equilibration_config": paths["equilibration_config"],
            "production_config": paths["production_config"],
            "trajectory_qc_config": paths["trajectory_qc_config"],
            "system_manifest": paths["system_manifest"],
            "equilibration_manifest": paths["equilibration_manifest"],
            "production_manifest": paths["production_manifest"],
            "trajectory_qc_summary": paths["trajectory_qc_summary"],
            "aligned_protein_pdb": paths["aligned_protein_pdb"],
            "aligned_protein_dcd": paths["aligned_protein_dcd"],
            "expected_frame_count": master["sampling"]["expected_frames_per_start"],
        })
    write_csv(manifest_path, manifest_rows)
    result = {
        "schema_version": "1.0",
        "status": "stage28_pparg_multistart_md_inputs_ready",
        "config": descriptor(root, config_path),
        "source_selection_manifest": descriptor(root, source_path),
        "start_manifest": descriptor(root, manifest_path),
        "starting_structure_count": len(manifest_rows),
        "expected_total_frames": sum(int(row["expected_frame_count"]) for row in manifest_rows),
        "starting_structures": [{"conformer_id": row["conformer_id"], "selection_rank": row["selection_rank"], "pdb": descriptor(root, rooted(root, row["starting_pdb"]))} for row in manifest_rows],
        "generated_configs": [descriptor(root, path) for path in sorted(generated_paths)],
        "data_boundary": {"docking_scores_read": 0, "ligand_labels_read": 0, "stage27_qubo_subsets_read": 0, "new_md_jobs_started": 0, "quantum_hardware_jobs": 0},
        "interpretation_boundary": master["interpretation_boundary"],
    }
    write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage28_pparg_multistart_md_ensemble.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
