"""Audit and aggregate paired docking seed replicates into score matrices.

Thin CLI wrapper; the core logic lives in ``qubo_receptor_ensemble.matrix``.
"""

from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import argparse
import csv
import json
import statistics
from pathlib import Path

from qubo_receptor_ensemble.io import file_sha256, write_csv
from qubo_receptor_ensemble.matrix import (
    aggregate_seed_rows,
    audit_ligand_manifest,
    build_matrix,
    load_config,
)

__all__ = [
    "file_sha256",
    "read_csv",
    "write_csv",
    "audit_ligand_manifest",
    "aggregate_seed_rows",
    "build_matrix",
    "load_config",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    ligand_spec = config["ligand_manifest"]
    expected = config["expected"]
    aggregation = config["aggregation"]
    outputs = config["outputs"]
    assert isinstance(ligand_spec, dict)
    assert isinstance(expected, dict)
    assert isinstance(aggregation, dict)
    assert isinstance(outputs, dict)

    ligand_path = Path(str(ligand_spec["path"]))
    if not ligand_path.is_file() or file_sha256(ligand_path) != str(
        ligand_spec["sha256"]
    ).upper():
        raise ValueError("ligand manifest is missing or its hash differs")
    expected_role_counts = {
        str(key): int(value)
        for key, value in expected["role_label_counts"].items()
    }
    ligand_by_id = audit_ligand_manifest(
        read_csv(ligand_path),
        int(expected["ligand_count"]),
        expected_role_counts,
        set(str(role) for role in expected["allowed_selection_roles"]),
    )

    seed_groups: list[tuple[str, list[dict[str, str]]]] = []
    seed_evidence: list[dict[str, object]] = []
    for run in config["seed_runs"]:
        seed_id = str(run["seed_id"])
        run_config_path = Path(str(run["config_path"]))
        summary_path = Path(str(run["summary_path"]))
        representative_path = Path(str(run["representative_scores_path"]))
        if not run_config_path.is_file() or file_sha256(run_config_path) != str(
            run["config_sha256"]
        ).upper():
            raise ValueError(f"seed {seed_id} config is missing or its hash differs")
        if not summary_path.is_file() or not representative_path.is_file():
            raise FileNotFoundError(f"seed {seed_id} outputs are incomplete")
        summary = json.loads(summary_path.read_text(encoding="ascii"))
        if summary.get("status") not in {"ok", "ok_with_search_warning"}:
            raise ValueError(f"seed {seed_id} run did not pass")
        if int(summary["docking_parameters"]["base_seed"]) != int(run["base_seed"]):
            raise ValueError(f"seed {seed_id} base seed differs")
        if int(summary["failed_receptor_ligand_pairs"]) != 0:
            raise ValueError(f"seed {seed_id} contains failed pairs")
        if int(summary["observed_receptor_ligand_pairs"]) != int(
            expected["receptor_ligand_pairs_per_seed"]
        ):
            raise ValueError(f"seed {seed_id} pair count differs")
        summary_output = summary["outputs"]["representative_long_csv"]
        if Path(str(summary_output["path"])).as_posix() != representative_path.as_posix():
            raise ValueError(f"seed {seed_id} representative path differs")
        representative_hash = file_sha256(representative_path)
        if representative_hash != str(summary_output["sha256"]).upper():
            raise ValueError(f"seed {seed_id} representative hash differs")
        seed_groups.append((seed_id, read_csv(representative_path)))
        seed_evidence.append(
            {
                "seed_id": seed_id,
                "base_seed": int(run["base_seed"]),
                "summary_path": summary_path.as_posix(),
                "summary_sha256": file_sha256(summary_path),
                "representative_scores_path": representative_path.as_posix(),
                "representative_scores_sha256": representative_hash,
                "search_quality_warning_count": int(
                    summary["search_quality_warning_count"]
                ),
            }
        )

    if len(seed_groups) != int(expected["seed_count"]):
        raise ValueError("seed count differs")
    combined = aggregate_seed_rows(
        seed_groups,
        ligand_by_id,
        int(expected["receptor_count"]),
        str(aggregation["representative_method"]),
    )
    primary_matrix = build_matrix(combined, "median_representative_score")
    sensitivity_matrix = build_matrix(combined, "minimum_representative_score")
    output_paths = {key: Path(str(value)) for key, value in outputs.items()}
    materialized = [
        output_paths["aggregated_long_csv"],
        output_paths["primary_median_matrix_csv"],
        output_paths["sensitivity_minimum_matrix_csv"],
        output_paths["summary_json"],
    ]
    if not args.overwrite and any(path.exists() for path in materialized):
        raise FileExistsError("aggregation outputs exist; use --overwrite")
    write_csv(output_paths["aggregated_long_csv"], combined)
    write_csv(output_paths["primary_median_matrix_csv"], primary_matrix)
    write_csv(output_paths["sensitivity_minimum_matrix_csv"], sensitivity_matrix)
    ranges = [float(row["seed_score_range"]) for row in combined]
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "ok",
        "config": {"path": args.config.as_posix(), "sha256": file_sha256(args.config)},
        "ligand_count": len(ligand_by_id),
        "receptor_count": int(expected["receptor_count"]),
        "seed_count": len(seed_groups),
        "aggregated_pair_count": len(combined),
        "locked_test_manifest_rows": 0,
        "aggregation": aggregation,
        "seed_evidence": seed_evidence,
        "seed_score_range_kcal_per_mol": {
            "median": statistics.median(ranges),
            "maximum": max(ranges),
        },
        "outputs": {
            key: {"path": path.as_posix(), "sha256": file_sha256(path)}
            for key, path in output_paths.items()
            if key != "summary_json"
        },
        "interpretation_note": config["interpretation_boundary"],
    }
    output_paths["summary_json"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["summary_json"].write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
