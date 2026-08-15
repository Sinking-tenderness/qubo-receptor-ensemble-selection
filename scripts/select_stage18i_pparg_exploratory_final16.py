"""Freeze the post-hoc exploratory PPARG final-16 receptor pool."""

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


def truth(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


def verified(root: Path, descriptor: dict[str, object]) -> Path:
    path = Path(str(descriptor["path"]))
    path = path if path.is_absolute() else root / path
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
        raise ValueError("Stage 18i implementation SHA-256 differs")
    inputs = {
        key: verified(root, dict(value))
        for key, value in dict(config["inputs"]).items()
    }

    stage18e = read_json(inputs["stage18e_failure_adjudication"])
    stage18h = read_json(inputs["stage18h_recovery_adjudication"])
    if stage18e["status"] != "stage18e_pparg_confirmatory_technical_gate_closed":
        raise ValueError("Stage 18e failure adjudication is not closed")
    if stage18e["outcome"]["confirmatory_gate_pass"] is not False:
        raise ValueError("Stage 18e confirmatory failure boundary differs")
    if stage18h["status"] != "stage18h_pparg_posthoc_reserve_recovery_independently_adjudicated_ok":
        raise ValueError("Stage 18h reserve recovery was not independently adjudicated")
    if any(int(value) != 0 for value in dict(stage18h["data_boundary"]).values()):
        raise ValueError("Stage 18h adjudication crossed a protected boundary")

    primary_ids = [str(value) for value in stage18e["outcome"]["passing_receptors"]]
    reserve_ids = [str(value) for value in stage18h["outcome"]["passing_reserve_receptors"]]
    selection = dict(config["selection"])
    if len(primary_ids) != int(selection["frozen_primary_passing_count"]):
        raise ValueError("Stage 18i primary passing count differs")
    if len(reserve_ids) != int(selection["eligible_reserve_passing_count"]):
        raise ValueError("Stage 18i reserve passing count differs")
    if set(primary_ids).intersection(reserve_ids):
        raise ValueError("Stage 18i primary and reserve pools overlap")

    primary_gate = {row["conformer_id"]: row for row in read_csv(inputs["stage18e_gate_results"])}
    reserve_gate = {row["conformer_id"]: row for row in read_csv(inputs["stage18h_gate_results"])}
    if any(not truth(primary_gate[receptor_id]["gate_pass"]) for receptor_id in primary_ids):
        raise ValueError("Stage 18i primary pool contains a failed receptor")
    if any(not truth(reserve_gate[receptor_id]["gate_pass"]) for receptor_id in reserve_ids):
        raise ValueError("Stage 18i reserve pool contains a failed receptor")

    structural_rows = read_csv(inputs["eligible_structural_pool"])
    structural_by_id = {row["conformer_id"]: row for row in structural_rows}
    candidate_ids = primary_ids + reserve_ids
    if not set(candidate_ids).issubset(structural_by_id):
        raise ValueError("Stage 18i candidate is absent from the structural pool")
    distances: dict[tuple[str, str], float] = {}
    candidate_set = set(candidate_ids)
    for row in read_csv(inputs["pairwise_distances"]):
        first = row["conformer_id_a"]
        second = row["conformer_id_b"]
        if first not in candidate_set or second not in candidate_set:
            continue
        pair = tuple(sorted((first, second)))
        if pair in distances:
            raise ValueError("duplicate Stage 18i structural-distance pair")
        distances[pair] = float(row["standardized_pocket_distance"])
    expected_pairs = len(candidate_ids) * (len(candidate_ids) - 1) // 2
    if len(distances) != expected_pairs:
        raise ValueError("Stage 18i candidate distance matrix is incomplete")

    addition_count = int(selection["recovery_addition_count"])
    additions = maxmin_select(candidate_ids, primary_ids, distances, addition_count)
    addition_ids = [str(row["conformer_id"]) for row in additions]
    if addition_ids != [str(value) for value in selection["expected_recovery_additions"]]:
        raise ValueError("Stage 18i deterministic recovery additions differ")
    final_ids = primary_ids + addition_ids
    if len(final_ids) != int(selection["final_receptor_count"]) or len(set(final_ids)) != len(final_ids):
        raise ValueError("Stage 18i final receptor count differs")

    stage18d_prepared = {row["conformer_id"]: row for row in read_csv(inputs["stage18d_prepared_receptors"])}
    stage18g_prepared = {row["conformer_id"]: row for row in read_csv(inputs["stage18g_prepared_receptors"])}
    prepared_by_id = {**stage18d_prepared, **stage18g_prepared}
    if not set(final_ids).issubset(prepared_by_id):
        raise ValueError("Stage 18i final receptor lacks a prepared PDBQT")
    for receptor_id in final_ids:
        row = prepared_by_id[receptor_id]
        if row["status"] != "ok":
            raise ValueError(f"Stage 18i prepared receptor failed: {receptor_id}")
        verified(root, {"path": row["receptor_pdbqt"], "sha256": row["receptor_pdbqt_sha256"]})

    distance_by_addition = {
        str(row["conformer_id"]): row["minimum_standardized_distance_to_selected_pool"]
        for row in additions
    }
    final_structural_rows = []
    final_prepared_rows = []
    for rank, receptor_id in enumerate(final_ids, start=1):
        is_recovery = receptor_id in addition_ids
        role = "posthoc_recovery_addition" if is_recovery else "stage18e_primary_passer"
        source_stage = "stage18h" if is_recovery else "stage18e"
        prefix = {
            "final_pool_rank": rank,
            "final_pool_role": role,
            "technical_gate_source": source_stage,
            "recovery_minimum_standardized_distance_to_selected_pool": (
                distance_by_addition.get(receptor_id, "")
            ),
        }
        final_structural_rows.append({**prefix, **structural_by_id[receptor_id]})
        final_prepared_rows.append({**prefix, **prepared_by_id[receptor_id]})

    output_structural = root / str(config["outputs"]["final_structural_manifest_csv"])
    output_prepared = root / str(config["outputs"]["final_prepared_receptor_manifest_csv"])
    output_summary = root / str(config["outputs"]["summary_json"])
    if not overwrite and any(path.exists() for path in (output_structural, output_prepared, output_summary)):
        raise FileExistsError("Stage 18i outputs exist; pass --overwrite")
    write_csv(output_structural, final_structural_rows)
    write_csv(output_prepared, final_prepared_rows)
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage18i_pparg_exploratory_final16_selected",
        "experiment_class": "posthoc_exploratory",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "stage18e_confirmatory_gate": "closed_failed_14_of_24",
        "stage18h_exploratory_recovery_gate": "passed_7_of_8",
        "selection_rule": selection["selection_rule"],
        "primary_receptor_ids": primary_ids,
        "passing_reserve_receptor_ids": reserve_ids,
        "selected_recovery_additions": [
            {
                "selection_rank": int(row["selection_rank"]),
                "conformer_id": str(row["conformer_id"]),
                "minimum_standardized_distance_to_selected_pool": row[
                    "minimum_standardized_distance_to_selected_pool"
                ],
            }
            for row in additions
        ],
        "final_receptor_ids": final_ids,
        "counts": {
            "primary_passing_receptor_count": len(primary_ids),
            "passing_reserve_receptor_count": len(reserve_ids),
            "selected_recovery_addition_count": len(addition_ids),
            "final_receptor_count": len(final_ids),
        },
        "data_use_audit": {
            "redocking_pass_fail_used": True,
            "redocking_rmsd_magnitudes_used_for_selection": False,
            "structural_distances_used": True,
            "activity_labels_read": 0,
            "benchmark_docking_scores_read": 0,
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
        },
        "outputs": {
            "final_structural_manifest_csv": {
                "path": output_structural.relative_to(root).as_posix(),
                "sha256": file_sha256(output_structural),
            },
            "final_prepared_receptor_manifest_csv": {
                "path": output_prepared.relative_to(root).as_posix(),
                "sha256": file_sha256(output_prepared),
            },
        },
        "next_gate": "prepare the frozen PPARG benchmark ligand panels, then generate a separate-engine Uni-Dock training matrix for the exploratory final 16",
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
