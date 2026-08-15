"""Select one nonredundant replacement for the admitted MAPK14 pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .select_stage08_mk14_expanded16_pool import (
        checked_record,
        deterministic_maxmin,
        file_sha256,
        load_complete_distances,
        read_csv,
        read_json,
        write_csv,
        write_json,
    )
except ImportError:
    from select_stage08_mk14_expanded16_pool import (
        checked_record,
        deterministic_maxmin,
        file_sha256,
        load_complete_distances,
        read_csv,
        read_json,
        write_csv,
        write_json,
    )


def run_selection(config_path: Path, overwrite: bool = False) -> dict[str, object]:
    config = read_json(config_path)
    boundary = dict(config["data_boundary"])
    if any(int(value) != 0 for value in boundary.values()):
        raise ValueError("Stage 08c selection crossed a data boundary")
    paths = {
        key: checked_record(value) for key, value in dict(config["inputs"]).items()
    }
    current_summary = read_json(paths["current15_summary"])
    failure = read_json(paths["stage08b_failure_adjudication"])
    if current_summary.get("status") != "stage08c_current15_manifest_ok":
        raise ValueError("current 15-receptor manifest did not pass")
    if failure.get("status") != "stage08b_replacement_gate_failed_one_receptor":
        raise ValueError("Stage 08b failure adjudication did not pass")
    current_rows = read_csv(paths["current15_manifest"])
    current_ids = [row["conformer_id"] for row in current_rows]
    expected = dict(config["expected"])
    if len(current_ids) != int(expected["current_receptor_count"]):
        raise ValueError("current admitted receptor count differs")

    eligible_rows = read_csv(paths["eligible_pool"])
    eligible_by_id = {row["conformer_id"]: row for row in eligible_rows}
    eligible_ids = sorted(eligible_by_id)
    distances = load_complete_distances(
        read_csv(paths["pairwise_distances"]), eligible_ids
    )
    rule = dict(config["replacement_rule"])
    direct_excluded = [str(value) for value in rule["directly_failed_receptor_ids"]]
    if direct_excluded[-1:] != [str(value) for value in failure["failed_receptor_ids"]]:
        raise ValueError("latest direct exclusion differs from Stage 08b failure")
    equivalence = dict(rule["technical_equivalence_exclusion"])
    tolerance = float(equivalence["maximum_standardized_pocket_distance"])
    propagated: set[str] = set()
    for failed_id in direct_excluded:
        failed_ligand = eligible_by_id[failed_id]["selected_ligand_resname"]
        for candidate_id in eligible_ids:
            if candidate_id == failed_id or candidate_id in current_ids:
                continue
            pair = tuple(sorted((failed_id, candidate_id)))
            if distances[pair] > tolerance:
                continue
            if bool(equivalence["require_same_selected_ligand_resname"]) and eligible_by_id[
                candidate_id
            ]["selected_ligand_resname"] != failed_ligand:
                continue
            propagated.add(candidate_id)
    propagated_ids = sorted(propagated)
    if propagated_ids != [str(value) for value in expected["propagated_equivalence_exclusions"]]:
        raise ValueError(f"technical-equivalence exclusion set differs: {propagated_ids}")

    all_excluded = set(direct_excluded) | propagated
    candidates = sorted(set(eligible_ids) - all_excluded)
    replacements = deterministic_maxmin(
        candidates,
        current_ids,
        distances,
        int(rule["replacement_count"]),
    )
    replacement_ids = [str(row["conformer_id"]) for row in replacements]
    if replacement_ids != [str(value) for value in expected["replacement_receptor_ids"]]:
        raise ValueError(f"Stage 08c replacement differs: {replacement_ids}")
    audit_by_id = {
        row["conformer_id"]: row for row in read_csv(paths["candidate_audit"])
    }
    output_rows: list[dict[str, object]] = []
    for replacement in replacements:
        receptor_id = str(replacement["conformer_id"])
        audit = audit_by_id[receptor_id]
        if audit["status"] != "coordinate_eligible":
            raise ValueError("Stage 08c replacement is not coordinate eligible")
        output_rows.append(
            {
                "replacement_rank": replacement["selection_rank"],
                "conformer_id": receptor_id,
                "minimum_standardized_distance_to_admitted_pool": replacement[
                    "minimum_standardized_distance_to_selected_pool"
                ],
                **eligible_by_id[receptor_id],
                "selected_ligand_heavy_atom_count": audit[
                    "selected_ligand_heavy_atom_count"
                ],
                "pocket_heavy_atom_completeness_fraction": audit[
                    "pocket_heavy_atom_completeness_fraction"
                ],
                "incomplete_standard_amino_acid_residue_count": audit[
                    "incomplete_standard_amino_acid_residue_count"
                ],
                "polymer_like_hetero_residue_count": audit[
                    "polymer_like_hetero_residue_count"
                ],
            }
        )
    outputs = dict(config["outputs"])
    csv_path = Path(str(outputs["replacement_selection_csv"]))
    summary_path = Path(str(outputs["summary_json"]))
    if not overwrite and (csv_path.exists() or summary_path.exists()):
        raise FileExistsError("Stage 08c replacement outputs already exist")
    write_csv(csv_path, output_rows)
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage08c_final_replacement_selection_ok",
        "config": {
            "path": config_path.as_posix(),
            "sha256": file_sha256(config_path),
        },
        "current_receptor_count": len(current_ids),
        "directly_failed_receptor_ids": direct_excluded,
        "propagated_equivalence_exclusions": propagated_ids,
        "replacement_selection": replacements,
        "replacement_receptor_ids": replacement_ids,
        "final_receptor_count_if_pass": len(current_ids) + len(replacements),
        "data_used_for_selection": {
            "structural_distances": True,
            "binary_technical_admission": True,
            "docking_affinities": False,
            "rmsd_magnitudes_for_ranking": False,
            "benchmark_labels": False,
        },
        "data_boundary": {str(key): int(value) for key, value in boundary.items()},
        "outputs": {
            "replacement_selection_csv": {
                "path": csv_path.as_posix(),
                "sha256": file_sha256(csv_path),
            }
        },
        "next_gate": "prepare and three-seed cognate-redock 1OZ1-FPH under the unchanged frozen protocol",
        "decision_boundary": config["decision_boundary"],
    }
    write_json(summary_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_selection(args.config, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
