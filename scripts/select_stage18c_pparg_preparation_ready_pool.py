"""Apply the frozen PPARG preparation-readiness correction and reselect 24 structures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def verified(root: Path, record: dict[str, object]) -> Path:
    path = root / str(record["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(record["sha256"]).upper():
        raise ValueError(f"SHA-256 differs: {path}")
    return path


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    if file_sha256(Path(__file__)) != str(config["implementation"]["sha256"]).upper():
        raise ValueError("Stage 18c amendment implementation SHA-256 differs")
    summary_path = verified(root, dict(config["inputs"]["structural_summary"]))
    eligible_path = verified(root, dict(config["inputs"]["eligible_pool"]))
    distances_path = verified(root, dict(config["inputs"]["pairwise_distances"]))
    source = read_json(summary_path)
    if source["status"] != "stage18c_pparg_structural_pool_ok":
        raise ValueError("Stage 18c structural pool did not pass")
    if any(int(value) != 0 for value in source["data_boundary"].values()):
        raise ValueError("Stage 18c crossed a protected data boundary")

    eligible = read_csv(eligible_path)
    ready = [
        row for row in eligible
        if int(row["global_incomplete_standard_amino_acid_residue_count"]) == 0
    ]
    by_id = {row["conformer_id"]: row for row in ready}
    ready_ids = sorted(by_id)
    if len(by_id) != len(ready):
        raise ValueError("preparation-ready PPARG IDs are not unique")
    reference_id = str(config["selection"]["reference_conformer_id"])
    target_count = int(config["selection"]["target_receptor_count"])
    if reference_id not in by_id or len(ready_ids) < target_count:
        raise ValueError("PPARG preparation-ready pool is insufficient")

    ready_set = set(ready_ids)
    distances: dict[tuple[str, str], float] = {}
    for row in read_csv(distances_path):
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
        raise ValueError("preparation-ready PPARG distance matrix is incomplete")

    additions = maxmin_select(ready_ids, [reference_id], distances, target_count - 1)
    selected_ids = [reference_id] + [str(row["conformer_id"]) for row in additions]
    rank_distance = {reference_id: ""}
    rank_distance.update({
        str(row["conformer_id"]): row["minimum_standardized_distance_to_selected_pool"]
        for row in additions
    })
    output_rows = [
        {
            "pool_role": "reference_seed" if rank == 1 else "maxmin_addition",
            "selection_rank": rank,
            "minimum_standardized_distance_to_selected_pool": rank_distance[value],
            **by_id[value],
        }
        for rank, value in enumerate(selected_ids, start=1)
    ]
    output_manifest = root / str(config["outputs"]["selected24_manifest_csv"])
    output_summary = root / str(config["outputs"]["summary_json"])
    if not overwrite and (output_manifest.exists() or output_summary.exists()):
        raise FileExistsError("Stage 18c amendment outputs exist; pass --overwrite")
    write_csv(output_manifest, output_rows)
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage18c_pparg_preparation_ready_pool_ok",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "trigger": config["trigger"],
        "counts": {
            "source_coordinate_eligible_count": len(eligible),
            "preparation_ready_count": len(ready),
            "excluded_incomplete_count": len(eligible) - len(ready),
            "pairwise_distance_count": len(distances),
            "selected_receptor_count": len(output_rows),
        },
        "selected_receptor_ids": selected_ids,
        "data_boundary": {
            "ligand_labels_read": 0,
            "docking_scores_read": 0,
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
        },
        "outputs": {
            "selected24_manifest_csv": {
                "path": output_manifest.relative_to(root).as_posix(),
                "sha256": file_sha256(output_manifest),
            }
        },
        "next_gate": "prepare the complete PPARG receptors and cognate ligands for three-seed redocking",
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
