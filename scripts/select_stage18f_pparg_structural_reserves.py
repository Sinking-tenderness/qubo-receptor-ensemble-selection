"""Select post-hoc PPARG reserve receptors by continuing the frozen max-min order."""

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


def verified(root: Path, descriptor: dict[str, object]) -> Path:
    path = root / str(descriptor["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"SHA-256 differs: {path}")
    return path


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    if file_sha256(Path(__file__)) != str(config["implementation"]["sha256"]).upper():
        raise ValueError("Stage 18f implementation SHA-256 differs")
    inputs = {
        key: verified(root, dict(value))
        for key, value in dict(config["inputs"]).items()
    }
    adjudication = read_json(inputs["failure_adjudication"])
    if adjudication["status"] != "stage18e_pparg_confirmatory_technical_gate_closed":
        raise ValueError("Stage 18e failure adjudication is not closed")
    if adjudication["decision"]["posthoc_structural_reserve_recovery_authorized"] is not True:
        raise ValueError("PPARG structural reserve recovery is not authorized")
    if any(int(value) != 0 for value in adjudication["data_boundary"].values()):
        raise ValueError("Stage 18e adjudication crossed a protected boundary")

    eligible = read_csv(inputs["eligible_pool"])
    ready = [
        row for row in eligible
        if int(row["global_incomplete_standard_amino_acid_residue_count"]) == 0
    ]
    by_id = {row["conformer_id"]: row for row in ready}
    ready_ids = sorted(by_id)
    selected = read_csv(inputs["selected24_manifest"])
    selected_ids = [row["conformer_id"] for row in selected]
    if len(selected_ids) != 24 or not set(selected_ids).issubset(ready_ids):
        raise ValueError("Stage 18f selected-24 structural seed differs")

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

    reserve_count = int(config["selection"]["reserve_receptor_count"])
    additions = maxmin_select(ready_ids, selected_ids, distances, reserve_count)
    rows = [
        {
            "pool_role": "posthoc_structural_reserve",
            "reserve_rank": int(addition["selection_rank"]),
            "global_structural_rank": len(selected_ids) + int(addition["selection_rank"]),
            "minimum_standardized_distance_to_selected_pool": addition[
                "minimum_standardized_distance_to_selected_pool"
            ],
            **by_id[str(addition["conformer_id"])],
        }
        for addition in additions
    ]
    output_manifest = root / str(config["outputs"]["reserve_manifest_csv"])
    output_summary = root / str(config["outputs"]["summary_json"])
    if not overwrite and (output_manifest.exists() or output_summary.exists()):
        raise FileExistsError("Stage 18f outputs exist; pass --overwrite")
    write_csv(output_manifest, rows)
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage18f_pparg_structural_reserves_selected",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "selection_timing": config["selection_timing"],
        "selection_rule": config["selection"]["selection_rule"],
        "counts": {
            "preparation_ready_count": len(ready_ids),
            "frozen_primary_count": len(selected_ids),
            "reserve_count": len(rows),
            "pairwise_distance_count": len(distances),
        },
        "reserve_receptor_ids": [row["conformer_id"] for row in rows],
        "data_use_audit": {
            "redocking_outcome_used_only_to_trigger_extension": True,
            "redocking_scores_used_to_rank_reserves": False,
            "structural_coordinates_used": True,
            "activity_labels_read": 0,
            "benchmark_docking_scores_read": 0,
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0
        },
        "outputs": {
            "reserve_manifest_csv": {
                "path": output_manifest.relative_to(root).as_posix(),
                "sha256": file_sha256(output_manifest),
            }
        },
        "next_gate": "prepare and three-seed redock all eight reserves; at least two must pass to form an exploratory final-16 PPARG pool",
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
