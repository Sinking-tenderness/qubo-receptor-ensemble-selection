"""Finalize the preparation-ready EGFR receptor pool and reserve."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
from itertools import combinations
from pathlib import Path

import numpy as np

try:
    from .select_stage13_egfr_coordinate_pool import (
        file_sha256,
        maxmin_select,
        read_csv,
        read_json,
        verified,
        write_csv,
        write_json,
    )
except ImportError:
    from select_stage13_egfr_coordinate_pool import (
        file_sha256,
        maxmin_select,
        read_csv,
        read_json,
        verified,
        write_csv,
        write_json,
    )


def preparation_ready_rows(
    audit_rows: list[dict[str, str]], amendment: dict[str, object]
) -> list[dict[str, str]]:
    expected_ids = [str(value) for value in amendment["selection"]["candidate_pdb_ids"]]
    rows = sorted(
        (
            row
            for row in audit_rows
            if row["status"] == "coordinate_eligible"
            and int(row["global_incomplete_standard_amino_acid_residue_count"]) == 0
        ),
        key=lambda row: row["pdb_id"],
    )
    if [row["pdb_id"] for row in rows] != expected_ids:
        raise ValueError("Stage 13d preparation-ready candidate IDs differ")
    return rows


def write_report(path: Path, summary: dict[str, object]) -> None:
    counts = dict(summary["counts"])
    lines = [
        "# Stage 13d EGFR Preparation-Ready Pool",
        "",
        "## Result",
        "",
        f"- Fully observed candidates: {counts['preparation_ready_candidate_count']}",
        f"- Selected receptors: {counts['selected_receptor_count']}",
        f"- Reserve receptors: {counts['reserve_receptor_count']}",
        f"- Status: {summary['status']}",
        "",
        "2RGP is retained only as the common-frame and max-min seed.",
        "No receptor heavy atom was modeled or deleted, and no benchmark label or docking score was read.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def run(config_path: Path, overwrite: bool = False) -> dict[str, object]:
    config = read_json(config_path)
    implementation = dict(config["implementation"])
    script_path = verified(
        Path(str(implementation["path"])), str(implementation["sha256"])
    )
    if script_path.resolve() != Path(__file__).resolve():
        raise ValueError("Stage 13d implementation path differs")
    verified(
        Path(str(implementation["dependency_path"])),
        str(implementation["dependency_sha256"]),
    )
    amendment_record = dict(config["preparation_readiness_amendment"])
    amendment_path = verified(
        Path(str(amendment_record["path"])), str(amendment_record["sha256"])
    )
    amendment = read_json(amendment_path)
    boundary = dict(amendment["data_boundary"])
    if any(int(value) != 0 for value in boundary.values()):
        raise ValueError("Stage 13d data boundary differs")

    expected_runtime = {
        key: str(value) for key, value in dict(config["runtime"]).items()
    }
    runtime = {
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    if runtime != expected_runtime:
        raise RuntimeError(f"Stage 13d runtime differs: {runtime} != {expected_runtime}")

    inputs: dict[str, Path] = {}
    for key, value in dict(config["inputs"]).items():
        record = dict(value)
        inputs[key] = verified(Path(str(record["path"])), str(record["sha256"]))
    structural_summary = read_json(inputs["structural_summary"])
    if structural_summary.get("status") != amendment["trigger"]["required_status"]:
        raise ValueError("Stage 13d trigger status differs")
    if any(int(value) != 0 for value in structural_summary["data_boundary"].values()):
        raise ValueError("Stage 13d upstream data boundary differs")

    outputs = {key: Path(str(value)) for key, value in dict(config["outputs"]).items()}
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Stage 13d outputs exist; pass --overwrite")
    if overwrite:
        for path in existing:
            path.unlink()

    audit_rows = read_csv(inputs["coordinate_audit_csv"])
    candidates = preparation_ready_rows(audit_rows, amendment)
    candidate_by_id = {row["conformer_id"]: row for row in candidates}
    feature_rows = read_csv(inputs["feature_matrix_csv"])
    feature_by_id = {row["conformer_id"]: row for row in feature_rows}
    reference_id = str(amendment["selection"]["reference_seed"])
    population_ids = sorted([reference_id, *candidate_by_id])
    if not set(population_ids).issubset(feature_by_id):
        raise ValueError("Stage 13d feature population is incomplete")
    feature_names = [name for name in feature_rows[0] if name != "conformer_id"]
    matrix = np.vstack(
        [
            [float(feature_by_id[conformer_id][name]) for name in feature_names]
            for conformer_id in population_ids
        ]
    )
    standard_deviations = matrix.std(axis=0)
    keep = standard_deviations >= float(
        config["selection_parameters"]["minimum_variable_feature_sd_angstrom"]
    )
    variable_feature_count = int(keep.sum())
    if variable_feature_count < 3:
        raise ValueError("too few variable Stage 13d features")
    means = matrix.mean(axis=0)
    standardized = (matrix[:, keep] - means[keep]) / standard_deviations[keep]
    standardized /= math.sqrt(variable_feature_count)
    distance_by_pair: dict[tuple[str, str], float] = {}
    distance_rows: list[dict[str, object]] = []
    for first_index, second_index in combinations(range(len(population_ids)), 2):
        first = population_ids[first_index]
        second = population_ids[second_index]
        distance = float(
            np.linalg.norm(standardized[first_index] - standardized[second_index])
        )
        distance_by_pair[(first, second)] = distance
        distance_rows.append(
            {
                "conformer_id_a": first,
                "conformer_id_b": second,
                "standardized_pocket_distance": distance,
            }
        )
    target_count = int(amendment["selection"]["target_receptor_count"])
    selected = maxmin_select(
        population_ids,
        [reference_id],
        distance_by_pair,
        target_count,
    )
    selected_ids = [str(row["conformer_id"]) for row in selected]
    if reference_id in selected_ids or not set(selected_ids).issubset(candidate_by_id):
        raise ValueError("Stage 13d selected a non-docking seed")
    selected_rows = [
        {
            "selection_rank": int(selection["selection_rank"]),
            "minimum_standardized_distance_to_selected_pool": selection[
                "minimum_standardized_distance_to_selected_pool"
            ],
            **candidate_by_id[str(selection["conformer_id"])],
        }
        for selection in selected
    ]
    reserve_ids = sorted(set(candidate_by_id) - set(selected_ids))
    expected_reserve_count = int(amendment["selection"]["reserve_receptor_count"])
    if len(reserve_ids) != expected_reserve_count:
        raise ValueError("Stage 13d reserve count differs")
    reserve_rows = [
        {
            "reserve_rank": rank,
            **candidate_by_id[conformer_id],
        }
        for rank, conformer_id in enumerate(reserve_ids, start=1)
    ]
    write_csv(outputs["selected_receptor_manifest_csv"], selected_rows)
    write_csv(outputs["reserve_receptor_manifest_csv"], reserve_rows)
    write_csv(outputs["selection_distances_csv"], distance_rows)

    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage13d_egfr_preparation_ready_pool_ok",
        "config": {"path": config_path.as_posix(), "sha256": file_sha256(config_path)},
        "preparation_readiness_amendment": {"path": amendment_path.as_posix(), "sha256": file_sha256(amendment_path)},
        "runtime": runtime,
        "counts": {
            "selection_seed_count": 1,
            "preparation_ready_candidate_count": len(candidates),
            "selected_receptor_count": len(selected_rows),
            "reserve_receptor_count": len(reserve_rows),
            "raw_feature_count": len(feature_names),
            "variable_feature_count": variable_feature_count,
            "pairwise_distance_count": len(distance_rows),
        },
        "selection_seed": reference_id,
        "selected_receptor_ids": selected_ids,
        "reserve_receptor_ids": reserve_ids,
        "data_boundary": {
            "ligand_labels_read": 0,
            "benchmark_docking_scores_read": 0,
            "MAPK14_stage11_rows_read": 0,
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
        },
        "outputs": {
            key: {"path": path.as_posix(), "sha256": file_sha256(path)}
            for key, path in outputs.items()
            if key not in {"summary_json", "report_md"}
        },
        "next_gate": "prepare all 16 receptors and cognate ligands, audit one common ATP-site box, and run three-seed Uni-Dock redocking",
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["summary_json"], summary)
    write_report(outputs["report_md"], summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
