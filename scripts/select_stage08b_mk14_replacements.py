"""Select deterministic structural replacements for failed Stage 08 receptors."""

from __future__ import annotations

import argparse
import csv
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
        raise ValueError("Stage 08b replacement selection crossed a data boundary")
    inputs = dict(config["inputs"])
    paths = {key: checked_record(value) for key, value in inputs.items()}
    adjudication = read_json(paths["failure_adjudication"])
    if adjudication.get("status") != "stage08_redocking_gate_failed_two_receptors":
        raise ValueError("Stage 08 failure was not formally adjudicated")
    if read_json(paths["structural_summary"]).get("status") != "expanded16_structural_selection_ok":
        raise ValueError("source structural selection did not pass")
    if read_json(paths["structural_audit"]).get("status") != "independent_expanded16_structural_audit_ok":
        raise ValueError("source structural audit did not pass")

    rule = dict(config["replacement_rule"])
    admitted = [str(value) for value in rule["admitted_receptor_ids"]]
    excluded = [str(value) for value in rule["permanently_excluded_receptor_ids"]]
    if excluded != [str(value) for value in adjudication["failed_receptor_ids"]]:
        raise ValueError("replacement exclusions differ from adjudicated failures")
    adjudicated_admitted = {str(value) for value in adjudication["admitted_receptor_ids"]}
    prior_new = {
        row["conformer_id"]
        for row in read_csv(paths["structural_manifest"])
        if row["prefix_status"] == "new_expanded16_addition"
    }
    if set(excluded) | adjudicated_admitted != prior_new:
        raise ValueError("adjudicated new-receptor partition is incomplete")

    eligible_rows = read_csv(paths["eligible_pool"])
    eligible_by_id = {row["conformer_id"]: row for row in eligible_rows}
    eligible_ids = sorted(eligible_by_id)
    expected = dict(config["expected"])
    if len(eligible_ids) != int(expected["eligible_pool_count"]):
        raise ValueError("eligible pool count differs")
    if len(admitted) != int(expected["admitted_receptor_count_before_replacement"]):
        raise ValueError("admitted receptor count differs")
    if not set(admitted).issubset(eligible_ids):
        raise ValueError("an admitted receptor is absent from the eligible pool")

    distance_rows = read_csv(paths["pairwise_distances"])
    distances = load_complete_distances(distance_rows, eligible_ids)
    if len(distances) != int(expected["pairwise_distance_count"]):
        raise ValueError("pairwise distance count differs")
    candidate_universe = sorted(set(eligible_ids) - set(excluded))
    replacements = deterministic_maxmin(
        candidate_universe,
        admitted,
        distances,
        int(rule["replacement_count"]),
    )
    replacement_ids = [str(row["conformer_id"]) for row in replacements]
    if replacement_ids != [str(value) for value in expected["replacement_receptor_ids"]]:
        raise ValueError(f"replacement IDs differ: {replacement_ids}")

    audit_by_id = {
        row["conformer_id"]: row for row in read_csv(paths["candidate_audit"])
    }
    rows: list[dict[str, object]] = []
    for replacement in replacements:
        receptor_id = str(replacement["conformer_id"])
        audit = audit_by_id[receptor_id]
        if audit["status"] != "coordinate_eligible":
            raise ValueError(f"replacement is not coordinate eligible: {receptor_id}")
        rows.append(
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
    summary_path = Path(str(outputs["replacement_selection_summary_json"]))
    if not overwrite and (csv_path.exists() or summary_path.exists()):
        raise FileExistsError("Stage 08b replacement outputs already exist")
    write_csv(csv_path, rows)
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage08b_replacement_selection_ok",
        "config": {
            "path": config_path.as_posix(),
            "sha256": file_sha256(config_path),
        },
        "admitted_receptor_count_before_replacement": len(admitted),
        "permanently_excluded_receptor_ids": excluded,
        "replacement_selection": replacements,
        "replacement_receptor_ids": replacement_ids,
        "final_receptor_count_if_both_pass": len(admitted) + len(replacements),
        "data_used_for_selection": {
            "structural_distances": True,
            "binary_redocking_admission_status": True,
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
        "next_gate": "prepare and three-seed cognate-redock 3ITZ-P66 and 2BAK-AQZ under the unchanged Stage 08 Uni-Dock protocol",
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
