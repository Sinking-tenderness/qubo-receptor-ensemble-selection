"""Select PPARG starts that pass an explicit MD-topology feasibility gate."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage21_structure_aware_qubo import (
    descriptor,
    read_csv,
    read_json,
    rooted,
    write_csv,
    write_json,
)
from scripts.select_mk14_rcsb_coordinate_pool import maxmin_select


def atom_records(path: Path, chain_id: str) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="ascii").splitlines():
        if not line.startswith("ATOM  ") or line[21:22] != chain_id:
            continue
        records.append({
            "name": line[12:16].strip(),
            "chain": line[21:22],
            "resseq": int(line[22:26]),
            "icode": line[26:27].strip(),
            "xyz": tuple(float(line[start:start + 8]) for start in (30, 38, 46)),
        })
    if not records:
        raise ValueError(f"no ATOM records for chain {chain_id}: {path}")
    return records


def distance(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def preflight(path: Path, row: dict[str, str], gate: dict[str, Any], root: Path) -> dict[str, Any]:
    chain_id = str(gate["required_atom_chain_id"])
    records = atom_records(path, chain_id)
    residue_order: list[tuple[int, str]] = []
    atoms: dict[tuple[int, str], dict[str, tuple[float, ...]]] = {}
    for atom in records:
        key = (int(atom["resseq"]), str(atom["icode"]))
        if key not in atoms:
            residue_order.append(key)
            atoms[key] = {}
        atoms[key][str(atom["name"])] = tuple(atom["xyz"])
    reasons = []
    incomplete = int(float(row["global_incomplete_standard_amino_acid_residue_count"] or 0))
    if bool(gate["require_global_complete_standard_residues"]) and incomplete != 0:
        reasons.append("globally_incomplete_standard_residue")
    if len(residue_order) < int(gate["minimum_residue_count"]):
        reasons.append("too_few_residues")
    if bool(gate["require_blank_insertion_codes"]) and any(code for _, code in residue_order):
        reasons.append("nonblank_insertion_code")
    gaps = []
    peptide_distances = []
    for first, second in zip(residue_order, residue_order[1:]):
        if second[0] != first[0] + 1 or first[1] or second[1]:
            gaps.append(f"{first[0]}{first[1]}-{second[0]}{second[1]}")
            continue
        if "C" not in atoms[first] or "N" not in atoms[second]:
            reasons.append("adjacent_peptide_backbone_atom_missing")
            continue
        peptide_distances.append(distance(atoms[first]["C"], atoms[second]["N"]))
    if bool(gate["require_strictly_consecutive_residue_numbers"]) and gaps:
        reasons.append("internal_residue_number_gap")
    lower = float(gate["minimum_adjacent_peptide_cn_distance_angstrom"])
    upper = float(gate["maximum_adjacent_peptide_cn_distance_angstrom"])
    if any(value < lower or value > upper for value in peptide_distances):
        reasons.append("adjacent_peptide_cn_distance_out_of_range")
    first_missing = sorted(set(gate["required_first_residue_atoms"]) - set(atoms[residue_order[0]]))
    last_missing = sorted(set(gate["required_last_residue_atoms"]) - set(atoms[residue_order[-1]]))
    if first_missing:
        reasons.append("first_residue_terminal_atom_missing")
    if last_missing:
        reasons.append("last_residue_terminal_atom_missing")
    reasons = sorted(set(reasons))
    return {
        "conformer_id": row["conformer_id"],
        "status": "md_ready" if not reasons else "md_ineligible",
        "exclusion_reasons": ";".join(reasons),
        "aligned_protein_pdb_path": path.relative_to(root).as_posix(),
        "global_incomplete_standard_amino_acid_residue_count": incomplete,
        "residue_count": len(residue_order),
        "first_residue_number": residue_order[0][0],
        "last_residue_number": residue_order[-1][0],
        "internal_gap_count": len(gaps),
        "internal_gaps": ";".join(gaps),
        "minimum_adjacent_peptide_cn_distance_angstrom": min(peptide_distances) if peptide_distances else "",
        "maximum_adjacent_peptide_cn_distance_angstrom": max(peptide_distances) if peptide_distances else "",
        "first_residue_missing_atoms": ";".join(first_missing),
        "last_residue_missing_atoms": ";".join(last_missing),
    }


def run(config_path: Path, root: Path, overwrite: bool = False) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    eligible_path = rooted(root, config["inputs"]["coordinate_eligible_pool"])
    distance_path = rooted(root, config["inputs"]["pairwise_structural_distances"])
    prior_path = rooted(root, config["inputs"]["prior_maxmin_selection"])
    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Stage28b selection outputs exist; pass --overwrite")
    eligible = read_csv(eligible_path)
    audit_rows = []
    source_by_id = {row["conformer_id"]: row for row in eligible}
    for row in eligible:
        source = rooted(root, row["aligned_protein_pdb_path"])
        audit_rows.append(preflight(source, row, config["preflight_gate"], root))
    ready_rows = [row for row in audit_rows if row["status"] == "md_ready"]
    ready_ids = {row["conformer_id"] for row in ready_rows}
    target_count = int(config["selection"]["target_start_count"])
    if len(ready_ids) < target_count:
        raise ValueError("fewer than eight PPARG structures pass MD preflight")
    prior = sorted(read_csv(prior_path), key=lambda row: int(row["selection_rank"]))
    seed_candidates = [row["conformer_id"] for row in prior if row["conformer_id"] in ready_ids]
    if not seed_candidates:
        raise ValueError("no prior max-min structure passes MD preflight")
    seed = seed_candidates[0]
    distances: dict[tuple[str, str], float] = {}
    for row in read_csv(distance_path):
        first, second = row["conformer_id_a"], row["conformer_id_b"]
        if first in ready_ids and second in ready_ids:
            distances[tuple(sorted((first, second)))] = float(row["standardized_pocket_distance"])
    expected_pairs = len(ready_ids) * (len(ready_ids) - 1) // 2
    if len(distances) != expected_pairs:
        raise ValueError("MD-ready structural distance matrix is incomplete")
    additions = maxmin_select(sorted(ready_ids), [seed], distances, target_count - 1)
    selected_ids = [seed] + [str(row["conformer_id"]) for row in additions]
    selected_distance = {seed: ""}
    selected_distance.update({
        str(row["conformer_id"]): row["minimum_standardized_distance_to_selected_pool"]
        for row in additions
    })
    audit_by_id = {row["conformer_id"]: row for row in audit_rows}
    selected_rows = []
    for rank, conformer_id in enumerate(selected_ids, start=1):
        source = source_by_id[conformer_id]
        selected_rows.append({
            "pool_role": "md_ready_seed" if rank == 1 else "md_ready_maxmin_addition",
            "selection_rank": rank,
            "minimum_standardized_distance_to_selected_pool": selected_distance[conformer_id],
            "conformer_id": conformer_id,
            "pdb_id": source["pdb_id"],
            "chain": source["chain"],
            "status": source["status"],
            "aligned_protein_pdb_path": audit_by_id[conformer_id]["aligned_protein_pdb_path"],
            "aligned_protein_pdb_sha256": source["aligned_protein_pdb_sha256"],
            "protein_ca_count": source["protein_ca_count"],
            "protein_heavy_atom_count": source["protein_heavy_atom_count"],
            "pocket_present_count": source["pocket_present_count"],
            "pocket_residue_fraction": source["pocket_residue_fraction"],
            "pocket_heavy_atom_completeness_fraction": source["pocket_heavy_atom_completeness_fraction"],
            "global_incomplete_standard_amino_acid_residue_count": source["global_incomplete_standard_amino_acid_residue_count"],
            "resolution_angstrom": source["resolution_angstrom"],
            "md_preflight_status": audit_by_id[conformer_id]["status"],
            "md_preflight_residue_count": audit_by_id[conformer_id]["residue_count"],
        })
    write_csv(outputs["preflight_audit_csv"], audit_rows)
    write_csv(outputs["selected_start_manifest_csv"], selected_rows)
    result = {
        "schema_version": "1.0",
        "status": "stage28b_pparg_md_ready_start_selection_ok",
        "config": descriptor(root, config_path),
        "inputs": {
            "coordinate_eligible_pool": descriptor(root, eligible_path),
            "pairwise_structural_distances": descriptor(root, distance_path),
            "prior_maxmin_selection": descriptor(root, prior_path),
        },
        "counts": {
            "coordinate_eligible": len(eligible),
            "md_ready": len(ready_rows),
            "md_ineligible": len(eligible) - len(ready_rows),
            "selected": len(selected_rows),
            "md_ready_distance_pairs": len(distances),
        },
        "seed_conformer_id": seed,
        "selected_conformer_ids": selected_ids,
        "quarantined_stage28_conformer_ids": ["PPARG_2GTK_reference"],
        "outputs": {key: descriptor(root, path) for key, path in outputs.items() if key != "summary_json"},
        "data_boundary": {
            "docking_scores_read": 0,
            "ligand_labels_read": 0,
            "stage28_trajectory_metrics_used_for_selection": 0,
            "quantum_hardware_jobs": 0,
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["summary_json"], result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage28b_pparg_md_ready_start_selection.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
