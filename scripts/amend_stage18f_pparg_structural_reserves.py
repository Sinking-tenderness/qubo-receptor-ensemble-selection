"""Replace a technically incompatible PPARG reserve before reserve docking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gemmi

try:
    from scripts.select_mk14_rcsb_coordinate_pool import maxmin_select
    from scripts.select_stage13_egfr_coordinate_pool import (
        file_sha256,
        read_csv,
        read_json,
        write_csv,
        write_json,
    )
except ModuleNotFoundError:
    from select_mk14_rcsb_coordinate_pool import maxmin_select
    from select_stage13_egfr_coordinate_pool import (
        file_sha256,
        read_csv,
        read_json,
        write_csv,
        write_json,
    )


def verified(root: Path, descriptor: dict[str, object]) -> Path:
    path = root / str(descriptor["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"SHA-256 differs: {path}")
    return path


def ligand_elements(path: Path, chain: str, resname: str, resseq: int) -> list[str]:
    structure = gemmi.read_structure(str(path))
    elements: set[str] = set()
    for model in structure:
        for model_chain in model:
            if model_chain.name != chain:
                continue
            for residue in model_chain:
                if residue.name.strip().upper() != resname.upper():
                    continue
                if int(residue.seqid.num) != resseq:
                    continue
                elements.update(atom.element.name.upper() for atom in residue)
    if not elements:
        raise ValueError(f"selected ligand was not found in {path}: {resname} {resseq}")
    return sorted(elements)


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    if file_sha256(Path(__file__)) != str(config["implementation"]["sha256"]).upper():
        raise ValueError("Stage 18f amendment implementation SHA-256 differs")
    inputs = {
        key: verified(root, dict(value))
        for key, value in dict(config["inputs"]).items()
    }

    original = read_json(inputs["original_selection_summary"])
    if original["status"] != "stage18f_pparg_structural_reserves_selected":
        raise ValueError("original Stage 18f selection did not pass")
    if any(
        int(original["data_use_audit"][key]) != 0
        for key in ("activity_labels_read", "benchmark_docking_scores_read", "fresh_validation_rows_read", "test_rows_read")
    ):
        raise ValueError("original Stage 18f selection crossed a protected boundary")
    original_rows = read_csv(inputs["original_reserve_manifest"])
    original_ids = [row["conformer_id"] for row in original_rows]
    if original_ids != [str(value) for value in original["reserve_receptor_ids"]]:
        raise ValueError("original Stage 18f reserve order differs")

    eligible_rows = read_csv(inputs["eligible_pool"])
    ready_rows = [
        row for row in eligible_rows
        if int(row["global_incomplete_standard_amino_acid_residue_count"]) == 0
    ]
    by_id = {row["conformer_id"]: row for row in ready_rows}
    ready_ids = sorted(by_id)
    selected_rows = read_csv(inputs["selected24_manifest"])
    selected_ids = [row["conformer_id"] for row in selected_rows]
    if len(selected_ids) != 24 or not set(selected_ids).issubset(ready_ids):
        raise ValueError("Stage 18f amendment selected-24 structural seed differs")

    distances: dict[tuple[str, str], float] = {}
    ready_set = set(ready_ids)
    for row in read_csv(inputs["pairwise_distances"]):
        first = row["conformer_id_a"]
        second = row["conformer_id_b"]
        if first not in ready_set or second not in ready_set:
            continue
        pair = tuple(sorted((first, second)))
        if pair in distances:
            raise ValueError("duplicate PPARG structural-distance pair")
        distances[pair] = float(row["standardized_pocket_distance"])
    expected_pairs = len(ready_ids) * (len(ready_ids) - 1) // 2
    if len(distances) != expected_pairs:
        raise ValueError("PPARG preparation-ready distance matrix is incomplete")

    amendment = dict(config["technical_amendment"])
    excluded_id = str(amendment["excluded_conformer_id"])
    if excluded_id not in original_ids:
        raise ValueError("technically excluded conformer is not an original reserve")
    excluded_row = by_id[excluded_id]
    excluded_mmcif = verified(
        root,
        {"path": excluded_row["mmcif_path"], "sha256": excluded_row["mmcif_sha256"]},
    )
    excluded_elements = ligand_elements(
        excluded_mmcif,
        excluded_row["chain"],
        excluded_row["selected_ligand_resname"],
        int(excluded_row["selected_ligand_resseq"]),
    )
    unsupported = {str(value).upper() for value in amendment["observed_unsupported_elements"]}
    if not unsupported.intersection(excluded_elements):
        raise ValueError("excluded PPARG ligand lacks the observed unsupported element")

    additions = maxmin_select(ready_ids, selected_ids, distances, len(original_ids) + 1)
    if [str(row["conformer_id"]) for row in additions[: len(original_ids)]] != original_ids:
        raise ValueError("original Stage 18f max-min order does not reproduce")
    replacement_addition = additions[len(original_ids)]
    replacement_id = str(replacement_addition["conformer_id"])
    if replacement_id != str(amendment["replacement_conformer_id"]):
        raise ValueError("replacement is not the next frozen max-min candidate")
    replacement_row = by_id[replacement_id]
    replacement_mmcif = verified(
        root,
        {"path": replacement_row["mmcif_path"], "sha256": replacement_row["mmcif_sha256"]},
    )
    replacement_elements = ligand_elements(
        replacement_mmcif,
        replacement_row["chain"],
        replacement_row["selected_ligand_resname"],
        int(replacement_row["selected_ligand_resseq"]),
    )
    if unsupported.intersection(replacement_elements):
        raise ValueError("replacement ligand contains an observed unsupported element")

    replacement_rank = int(replacement_addition["selection_rank"])
    retained = [row for row in original_rows if row["conformer_id"] != excluded_id]
    replacement = {
        "pool_role": "posthoc_structural_reserve_technical_replacement",
        "reserve_rank": replacement_rank,
        "global_structural_rank": len(selected_ids) + replacement_rank,
        "minimum_standardized_distance_to_selected_pool": replacement_addition[
            "minimum_standardized_distance_to_selected_pool"
        ],
        **replacement_row,
    }
    amended_rows = sorted(retained + [replacement], key=lambda row: int(row["reserve_rank"]))
    output_rows = [
        {
            "recovery_test_rank": rank,
            "technical_amendment_status": (
                "retained_from_original_stage18f"
                if row["conformer_id"] != replacement_id
                else "replacement_for_unsupported_ligand_atom_type"
            ),
            **row,
        }
        for rank, row in enumerate(amended_rows, start=1)
    ]
    if len(output_rows) != len(original_rows) or excluded_id in {row["conformer_id"] for row in output_rows}:
        raise ValueError("Stage 18f amended reserve count differs")

    output_manifest = root / str(config["outputs"]["amended_reserve_manifest_csv"])
    output_summary = root / str(config["outputs"]["summary_json"])
    if not overwrite and (output_manifest.exists() or output_summary.exists()):
        raise FileExistsError("Stage 18f amendment outputs exist; pass --overwrite")
    write_csv(output_manifest, output_rows)
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        # The preparation runner consumes this selection status while the explicit
        # amendment fields preserve why and when the member changed.
        "status": "stage18f_pparg_structural_reserves_selected",
        "amendment_status": "stage18f_pparg_structural_reserves_amendment01_applied",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "amendment_timing": config["amendment_timing"],
        "technical_failure": {
            "excluded_conformer_id": excluded_id,
            "ligand_resname": excluded_row["selected_ligand_resname"],
            "ligand_elements": excluded_elements,
            "observed_error_signature": amendment["observed_error_signature"],
            "reserve_redocking_jobs_started_before_amendment": 0,
        },
        "replacement": {
            "conformer_id": replacement_id,
            "original_maxmin_reserve_rank": replacement_rank,
            "ligand_resname": replacement_row["selected_ligand_resname"],
            "ligand_elements": replacement_elements,
            "local_meeko_preflight_status": amendment["replacement_meeko_preflight_status"],
        },
        "selection_rule": config["selection_rule"],
        "reserve_receptor_ids": [row["conformer_id"] for row in output_rows],
        "counts": {
            "original_reserve_count": len(original_rows),
            "technically_excluded_count": 1,
            "replacement_count": 1,
            "amended_reserve_count": len(output_rows),
        },
        "data_use_audit": {
            "reserve_input_preparation_error_read": True,
            "reserve_redocking_jobs_started": 0,
            "reserve_redocking_scores_read": 0,
            "activity_labels_read": 0,
            "benchmark_docking_scores_read": 0,
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
            "structural_coordinates_used": True,
        },
        "recovery_gate": config["recovery_gate"],
        "outputs": {
            "amended_reserve_manifest_csv": {
                "path": output_manifest.relative_to(root).as_posix(),
                "sha256": file_sha256(output_manifest),
            }
        },
        "next_gate": "prepare and three-seed redock the eight technically compatible reserves",
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(output_summary, result)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
