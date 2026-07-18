"""Audit the expanded MAPK14 train matrix against its frozen admission gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .aggregate_seed_replicates import file_sha256, read_csv
    from .audit_stage05_development_matrix import (
        audit_aggregated_rows,
        validate_matrix,
        write_flags,
    )
except ImportError:
    from aggregate_seed_replicates import file_sha256, read_csv
    from audit_stage05_development_matrix import (
        audit_aggregated_rows,
        validate_matrix,
        write_flags,
    )


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return value


def require_hash(path: Path, expected: object, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(expected).upper():
        raise ValueError(f"{name} SHA-256 differs")


def verify_seed_evidence(
    evidence: list[object],
    expected_seed_count: int,
    expected_pairs: int,
    expected_ligands: int,
    expected_receptors: int,
    required_failed_pairs: int,
) -> list[str]:
    if len(evidence) != expected_seed_count:
        raise ValueError("aggregation seed evidence count differs")
    seed_ids: list[str] = []
    for item in evidence:
        if not isinstance(item, dict):
            raise ValueError("seed evidence must contain objects")
        seed_id = str(item["seed_id"])
        seed_ids.append(seed_id)
        summary_path = Path(str(item["summary_path"]))
        scores_path = Path(str(item["representative_scores_path"]))
        require_hash(summary_path, item["summary_sha256"], f"{seed_id} summary")
        require_hash(
            scores_path,
            item["representative_scores_sha256"],
            f"{seed_id} representative scores",
        )
        seed_summary = load_json(summary_path)
        if seed_summary.get("status") not in {"ok", "ok_with_search_warning"}:
            raise ValueError(f"{seed_id} execution status did not pass")
        if int(seed_summary["docking_parameters"]["base_seed"]) != int(
            item["base_seed"]
        ):
            raise ValueError(f"{seed_id} base seed differs")
        expected_values = {
            "expected_receptor_ligand_pairs": expected_pairs,
            "observed_receptor_ligand_pairs": expected_pairs,
            "successful_receptor_ligand_pairs": expected_pairs,
            "failed_receptor_ligand_pairs": required_failed_pairs,
            "ligand_count": expected_ligands,
            "receptor_count": expected_receptors,
        }
        for field, expected in expected_values.items():
            if int(seed_summary.get(field, -1)) != expected:
                raise ValueError(f"{seed_id} {field} differs")
        if len(read_csv(scores_path)) != expected_pairs:
            raise ValueError(f"{seed_id} representative score row count differs")
    if len(seed_ids) != len(set(seed_ids)):
        raise ValueError("aggregation contains duplicate seed IDs")
    return seed_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_json(args.config)
    inputs = config["inputs"]
    hashes = config["input_sha256"]
    outputs = config["outputs"]
    assert isinstance(inputs, dict)
    assert isinstance(hashes, dict)
    assert isinstance(outputs, dict)
    input_paths = {key: Path(str(value)) for key, value in inputs.items()}
    for key, path in input_paths.items():
        require_hash(path, hashes[key], key)
    require_hash(
        args.source_archive,
        config["source_archive"]["sha256"],
        "source archive",
    )

    output_summary = Path(str(outputs["summary_json"]))
    flagged_output = Path(str(outputs["flagged_pairs_csv"]))
    for path in (output_summary, flagged_output):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"output exists; use --overwrite: {path}")

    preregistration = load_json(input_paths["preregistration"])
    aggregate_summary = load_json(input_paths["aggregation_summary"])
    if aggregate_summary.get("status") != "ok":
        raise ValueError("seed aggregation did not pass")
    if int(aggregate_summary.get("locked_test_manifest_rows", -1)) != 0:
        raise ValueError("aggregation contains locked test rows")

    admission = preregistration["matrix_admission"]
    frozen = preregistration["frozen_inputs"]
    docking = preregistration["docking"]
    assert isinstance(admission, dict)
    assert isinstance(frozen, dict)
    assert isinstance(docking, dict)
    receptor_spec = frozen["receptor_manifest"]
    ligand_spec = frozen["train_ligand_manifest"]
    assert isinstance(receptor_spec, dict)
    assert isinstance(ligand_spec, dict)
    receptor_path = Path(str(receptor_spec["path"]))
    ligand_path = Path(str(ligand_spec["path"]))
    require_hash(receptor_path, receptor_spec["sha256"], "receptor manifest")
    require_hash(ligand_path, ligand_spec["sha256"], "ligand manifest")

    receptor_rows = read_csv(receptor_path)
    ligand_rows = read_csv(ligand_path)
    expected_receptors = int(receptor_spec["row_count"])
    expected_ligands = int(ligand_spec["row_count"])
    expected_pairs = expected_receptors * expected_ligands
    if len(receptor_rows) != expected_receptors or len(ligand_rows) != expected_ligands:
        raise ValueError("frozen manifest row count differs")
    if any(row.get("status") != "ok" for row in receptor_rows):
        raise ValueError("receptor manifest contains a failed receptor")
    if any(
        row.get("selection_role") != "development_train" or row.get("split") != "train"
        for row in ligand_rows
    ):
        raise ValueError("ligand manifest is not development-train-only")
    if expected_pairs != int(docking["receptor_ligand_pairs_per_seed"]):
        raise ValueError("frozen docking pair count differs")
    if int(aggregate_summary.get("aggregated_pair_count", -1)) != expected_pairs:
        raise ValueError("aggregated pair count differs")
    if int(aggregate_summary.get("ligand_count", -1)) != expected_ligands:
        raise ValueError("aggregated ligand count differs")
    if int(aggregate_summary.get("receptor_count", -1)) != expected_receptors:
        raise ValueError("aggregated receptor count differs")

    evidence = aggregate_summary.get("seed_evidence")
    if not isinstance(evidence, list):
        raise ValueError("aggregation summary has no seed evidence")
    seed_ids = verify_seed_evidence(
        evidence,
        int(admission["required_seed_count"]),
        expected_pairs,
        expected_ligands,
        expected_receptors,
        int(admission["required_failed_pairs_per_seed"]),
    )
    observed_base_seeds = [int(item["base_seed"]) for item in evidence]
    if observed_base_seeds != [int(value) for value in docking["paired_base_seeds"]]:
        raise ValueError("aggregation base seeds differ from preregistration")

    output_specs = aggregate_summary.get("outputs")
    if not isinstance(output_specs, dict):
        raise ValueError("aggregation summary has no output evidence")
    required_outputs = {
        "aggregated_long_csv",
        "primary_median_matrix_csv",
        "sensitivity_minimum_matrix_csv",
    }
    if not required_outputs.issubset(output_specs):
        raise ValueError("aggregation output evidence is incomplete")
    aggregate_paths: dict[str, Path] = {}
    for key in required_outputs:
        spec = output_specs[key]
        if not isinstance(spec, dict):
            raise ValueError(f"invalid aggregate output evidence: {key}")
        path = Path(str(spec["path"]))
        require_hash(path, spec["sha256"], key)
        aggregate_paths[key] = path

    aggregated_rows = read_csv(aggregate_paths["aggregated_long_csv"])
    flags, audit = audit_aggregated_rows(
        aggregated_rows,
        seed_ids,
        expected_ligands,
        expected_receptors,
        {"development_train"},
        int(admission["maximum_allowed_nonnegative_score_pairs"]),
        float(admission["maximum_allowed_seed_score_range_kcal_per_mol"]),
    )
    receptor_ids = [row["conformer_id"] for row in receptor_rows]
    validate_matrix(
        read_csv(aggregate_paths["primary_median_matrix_csv"]),
        aggregated_rows,
        receptor_ids,
        "median_representative_score",
    )
    validate_matrix(
        read_csv(aggregate_paths["sensitivity_minimum_matrix_csv"]),
        aggregated_rows,
        receptor_ids,
        "minimum_representative_score",
    )

    write_flags(flagged_output, flags)
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": (
            "matrix_admission_passed"
            if audit["admission_passed"]
            else "matrix_admission_rejected_pending_label_blind_diagnostics"
        ),
        "config": {"path": args.config.as_posix(), "sha256": file_sha256(args.config)},
        "source_archive": {
            "filename": args.source_archive.name,
            "sha256": file_sha256(args.source_archive),
        },
        "preregistration": {
            "path": input_paths["preregistration"].as_posix(),
            "sha256": file_sha256(input_paths["preregistration"]),
        },
        "aggregation_summary": {
            "path": input_paths["aggregation_summary"].as_posix(),
            "sha256": file_sha256(input_paths["aggregation_summary"]),
        },
        "thresholds": admission,
        "audit": audit,
        "original_matrix_cells_replaced": 0,
        "qubo_fitted": False,
        "enrichment_metrics_calculated": False,
        "validation_rows_read": 0,
        "test_rows_read": 0,
        "outputs": {
            "flagged_pairs_csv": {
                "path": flagged_output.as_posix(),
                "sha256": file_sha256(flagged_output),
            }
        },
        "interpretation_note": config["interpretation_boundary"],
    }
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0 if audit["admission_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
