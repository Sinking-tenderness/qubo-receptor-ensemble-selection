"""Independently audit Stage 56 PPARD panels and Amendment01 coordinates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def checked(root: Path, value: dict[str, Any]) -> Path:
    path = root / str(value["path"])
    if not path.is_file() or sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage 56b evidence identity differs: {path}")
    return path


def run(config_path: Path, root: Path, output_path: Path) -> dict[str, Any]:
    root = root.resolve()
    config = read_json(config_path)
    implementation = checked(root, config["implementation"])
    if implementation.resolve() != Path(__file__).resolve():
        raise ValueError("Stage 56b audit implementation differs")
    inputs = {key: checked(root, value) for key, value in config["inputs"].items()}

    allocation = read_json(inputs["allocation_summary"])
    selected = read_csv(inputs["selected_panel_manifest"])
    pilot = read_csv(inputs["pilot_manifest"])
    if allocation["status"] != "stage56_ppard_ligand_panels_and_pilot_frozen":
        raise ValueError("Stage56 ligand allocation did not pass")
    observed = Counter((row["split"], row["label"]) for row in selected)
    expected = Counter(
        {
            ("train", "active"): 120,
            ("train", "decoy"): 120,
            ("validation", "active"): 60,
            ("validation", "decoy"): 1200,
            ("test", "active"): 60,
            ("test", "decoy"): 1200,
        }
    )
    if len(selected) != 2760 or observed != expected:
        raise ValueError("Stage56 selected panel counts differ")
    for key in (
        "source_molecule_id",
        "canonical_smiles",
        "scaffold_smiles",
        "split_group_id",
    ):
        splits: dict[str, set[str]] = defaultdict(set)
        for row in selected:
            splits[row[key]].add(row["split"])
        if any(len(values) != 1 for values in splits.values()):
            raise ValueError(f"Stage56 {key} crosses panel boundaries")
    if len(pilot) != 96 or any(row["split"] != "train" for row in pilot):
        raise ValueError("Stage56 pilot membership differs")
    fold_counts = Counter((int(row["pilot_outer_fold"]), row["label"]) for row in pilot)
    if fold_counts != Counter(
        {(fold, label): 12 for fold in range(4) for label in ("active", "decoy")}
    ):
        raise ValueError("Stage56 pilot fold counts differ")
    scaffold_folds: dict[str, set[int]] = defaultdict(set)
    for row in pilot:
        scaffold_folds[row["scaffold_smiles"]].add(int(row["pilot_outer_fold"]))
    if any(len(values) != 1 for values in scaffold_folds.values()):
        raise ValueError("Stage56 pilot scaffold crosses folds")

    original_config = read_json(inputs["original_coordinate_config"])
    original = read_json(inputs["original_coordinate_summary"])
    adjudication = read_json(inputs["numbering_adjudication"])
    remapping = read_json(inputs["remapping_result"])
    amended_config = read_json(inputs["amended_coordinate_config"])
    amended = read_json(inputs["amended_coordinate_summary"])
    if original["status"] != "stage56_ppard_coordinate_pool_insufficient_stop":
        raise ValueError("Stage56 original coordinate failure differs")
    if original["counts"]["coordinate_eligible_count"] != 17:
        raise ValueError("Stage56 original eligible count differs")
    if not adjudication["decision"]["sequence_correspondence_amendment_authorized"]:
        raise ValueError("Stage56a amendment was not authorized")
    if adjudication["decision"]["threshold_lowering_authorized"]:
        raise ValueError("Stage56a authorized threshold lowering")
    if adjudication["counts"]["sequence_mapping_pass_count"] != 51:
        raise ValueError("Stage56a sequence mapping count differs")
    if remapping["status"] != "stage56b_ppard_sequence_remapped_coordinates_ready":
        raise ValueError("Stage56b remapped coordinates are incomplete")
    if remapping["raw_coordinates_modified"] or remapping["thresholds_changed"]:
        raise ValueError("Stage56b remapping changed protected information")

    unchanged_keys = (
        "minimum_matched_ca_count",
        "maximum_aligned_global_ca_rmsd_angstrom",
        "minimum_pocket_residue_fraction",
        "require_all_anchor_residues",
        "require_reference_residue_name_match",
        "minimum_pocket_heavy_atom_completeness_fraction",
        "require_complete_reference_heavy_atom_template_for_each_anchor",
        "require_qualifying_ligand_within_reference_ligand_angstrom",
        "hidden_covalency",
        "missing_nonanchor_feature_imputation",
        "record_global_incomplete_standard_amino_acid_residues",
        "global_incomplete_residue_policy",
    )
    for key in unchanged_keys:
        if original_config["coordinate_gate"][key] != amended_config["coordinate_gate"][key]:
            raise ValueError(f"Stage56b coordinate threshold differs: {key}")
    if original_config["structural_pool"]["minimum_coordinate_eligible_count"] != amended_config[
        "structural_pool"
    ]["minimum_coordinate_eligible_count"]:
        raise ValueError("Stage56b minimum pool size differs")

    remap_manifest = read_csv(inputs["remapped_coordinate_manifest"])
    if len(remap_manifest) != 51 or len({row["pdb_id"] for row in remap_manifest}) != 51:
        raise ValueError("Stage56b remapping manifest differs")
    for row in remap_manifest:
        source = root / row["source_mmcif_path"]
        remapped = root / row["remapped_mmcif_path"]
        if sha256(source) != row["source_mmcif_sha256"] or sha256(remapped) != row[
            "remapped_mmcif_sha256"
        ]:
            raise ValueError(f"Stage56b remapped coordinate hash differs: {row['pdb_id']}")

    for value in amended["artifacts"].values():
        checked(root, value)
    coordinate_rows = read_csv(inputs["amended_coordinate_audit"])
    redocking_rows = read_csv(inputs["amended_redocking_pool"])
    if (
        amended["status"] != "stage56_ppard_coordinate_pool_hard_gate_ok"
        or len(coordinate_rows) != 51
        or len(redocking_rows) != 51
        or any(row["status"] != "coordinate_eligible" for row in coordinate_rows)
    ):
        raise ValueError("Stage56b amended coordinate result differs")
    for row in coordinate_rows:
        pdb_path = root / row["aligned_protein_pdb_path"]
        if not pdb_path.is_file() or sha256(pdb_path) != row[
            "aligned_protein_pdb_sha256"
        ]:
            raise ValueError(f"Stage56b aligned PDB identity differs: {row['pdb_id']}")
    if amended["selection_policy"] != {
        "hard_gate_only": True,
        "all_hard_gate_passing_structures_retained": True,
        "max_min_or_outcome_informed_compression_used": False,
    }:
        raise ValueError("Stage56b structural selection policy differs")
    if any(
        amended["decision"][key]
        for key in (
            "pilot_production_docking_authorized",
            "full_training_matrix_authorized",
            "fresh_validation_release_authorized",
            "quantum_hardware_authorized",
        )
    ):
        raise ValueError("Stage56b authorization boundary differs")

    audit = {
        "schema_version": "1.0",
        "status": "stage56b_ppard_allocation_and_coordinates_independent_audit_ok",
        "config": descriptor(root, config_path),
        "implementation": descriptor(root, implementation),
        "ligand_panels": {
            "selected_row_count": len(selected),
            "pilot_row_count": len(pilot),
            "outer_fold_count": 4,
            "panel_disjointness_exact": True,
            "pilot_fold_balance_exact": True,
        },
        "coordinate_adjudication": {
            "original_eligible_count": 17,
            "systematic_numbering_failure_confirmed": True,
            "sequence_mapping_pass_count": 51,
            "raw_coordinates_modified": False,
            "coordinate_thresholds_exact": True,
            "amended_eligible_count": 51,
            "all_hard_gate_passing_structures_retained": True,
        },
        "decision": {
            "ligand_allocation_gate_passed": True,
            "coordinate_gate_passed": True,
            "cognate_redocking_input_preparation_authorized": True,
            "pilot_production_docking_authorized": False,
            "full_training_matrix_authorized": False,
            "fresh_validation_release_authorized": False,
            "quantum_hardware_authorized": False,
        },
        "data_boundary": {
            "docking_scores_read": 0,
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    run(config_path, root, args.output if args.output.is_absolute() else root / args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
